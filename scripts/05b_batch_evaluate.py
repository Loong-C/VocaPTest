#!/usr/bin/env python
"""Batch evaluation: song-level top-k recall using precomputed embeddings."""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from vpstyle.retrieval.similarity import score_song_against_all
from vpstyle.retrieval.build_profiles import load_profiles
from vpstyle.utils.logging import setup_logging

logger = setup_logging()


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation of retrieval")
    parser.add_argument("--profile", required=True, help="Path to profiles pickle file")
    parser.add_argument("--embeddings", required=True, help="Path to embedding manifest JSONL")
    parser.add_argument("--top-k", type=int, default=3, help="Top-K for recall/accuracy")
    parser.add_argument("--limit", type=int, default=0, help="Limit songs queried (0=all)")
    args = parser.parse_args()

    profiles = load_profiles(args.profile)
    logger.info("Loaded %d profiles", len(profiles["producers"]))

    records = []
    with open(args.embeddings, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Loaded %d embedding records", len(records))

    embedding_dir = Path(args.embeddings).parent

    # Group by (producer_slug, song_id) -> list[segment_id]
    song_groups = defaultdict(list)
    for r in records:
        key = (r["producer_slug"], r["song_id"])
        song_groups[key].append(r["segment_id"])

    logger.info("Grouped into %d songs", len(song_groups))

    songs = list(song_groups.items())
    if args.limit > 0:
        songs = songs[: args.limit]

    top1_correct = 0
    topk_correct = 0
    producer_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    confusion = defaultdict(lambda: defaultdict(int))

    start = time.time()
    for (true_producer, song_id), seg_ids in songs:
        seg_embs = []
        for sid in seg_ids:
            emb_path = embedding_dir / f"{sid}.npy"
            if emb_path.exists():
                emb = np.load(emb_path)
                seg_embs.append(emb)

        if not seg_embs:
            continue

        seg_embs = np.stack(seg_embs, axis=0)
        scores = score_song_against_all(seg_embs, profiles)

        top_slugs = [s["producer_slug"] for s in scores[: args.top_k]]
        top1 = top_slugs[0] if top_slugs else None

        if top1 == true_producer:
            top1_correct += 1
        if true_producer in top_slugs:
            topk_correct += 1

        producer_stats[true_producer]["total"] += 1
        if top1 == true_producer:
            producer_stats[true_producer]["correct"] += 1

        confusion[true_producer][top1] += 1

    elapsed = time.time() - start
    total = sum(v["total"] for v in producer_stats.values())

    print(f"\n{'='*60}")
    print(f"  Song-Level Batch Evaluation (top-{args.top_k})")
    print(f"{'='*60}")
    print(f"  Total songs:    {total}")
    print(f"  Top-1 accuracy:  {top1_correct}/{total} = {top1_correct/total*100:.1f}%")
    print(f"  Top-{args.top_k} recall:   {topk_correct}/{total} = {topk_correct/total*100:.1f}%")
    print(f"  Time: {elapsed:.1f}s ({elapsed*1000/total:.1f}ms/song)")
    print(f"{'='*60}")

    print(f"\n  Per-Producer Top-1 Accuracy:")
    print(f"  {'Producer':<20s} {'Correct':>8s} {'Total':>6s} {'Acc%':>8s}")
    print(f"  {'-'*44}")
    for p in sorted(producer_stats):
        r = producer_stats[p]
        acc = r["correct"] / r["total"] * 100 if r["total"] > 0 else 0
        bar = "█" * max(1, int(acc / 5))
        print(f"  {p:<20s} {r['correct']:>8d} {r['total']:>6d} {acc:>7.1f}% {bar}")

    print(f"\n  Top Misclassifications (top-1 wrong):")
    all_wrong = []
    for p in sorted(confusion):
        for op, count in confusion[p].items():
            if op != p:
                all_wrong.append((p, op, count))
    all_wrong.sort(key=lambda x: -x[2])
    for p, op, count in all_wrong[:10]:
        print(f"    {p:<20s} -> {op:<20s} ({count}x)")


if __name__ == "__main__":
    main()
