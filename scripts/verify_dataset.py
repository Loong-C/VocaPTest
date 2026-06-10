#!/usr/bin/env python
"""Verify dataset quality after download.
Checks:
  1. Per-producer song count (target: ~12)
  2. Audio duration range (60-600s)
  3. Duplicate detection (same title, same producer)
  4. File size / bitrate sanity
"""
import json
import subprocess
from pathlib import Path
from collections import defaultdict, Counter

DATA_DIR = Path("data/audio")
SONGS_JSONL = Path("data/interim/youtube_songs.jsonl")

print("=" * 60)
print("DATASET QUALITY VERIFICATION")
print("=" * 60)

# --- 1. Count per producer from JSONL ---
print("\n[1] Songs per producer (JSONL):")
producer_songs = defaultdict(list)
with open(SONGS_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            s = json.loads(line)
            producer_songs[s["producer_slug"]].append(s)

for slug, songs in sorted(producer_songs.items()):
    flag = " ✅" if len(songs) >= 10 else " ⚠️ LOW" if len(songs) < 6 else ""
    print(f"  {slug:22s}: {len(songs):3d} songs{flag}")

print(f"\n  TOTAL: {sum(len(v) for v in producer_songs.values())} songs across {len(producer_songs)} producers")

# --- 2. Check audio files exist and get durations ---
print("\n[2] Audio file check:")
files = list(DATA_DIR.rglob("*.mp3")) + list(DATA_DIR.rglob("*.m4a")) + list(DATA_DIR.rglob("*.webm"))
print(f"  Audio files found: {len(files)}")

short_songs = []
long_songs = []
missing_audio = []

for slug, songs in sorted(producer_songs.items()):
    for song in songs:
        path = song.get("local_audio_path")
        if not path or not Path(path).exists():
            missing_audio.append(song["song_id"])
            continue
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", path],
                capture_output=True, text=True, timeout=30,
            )
            dur = float(result.stdout.strip())
            if dur < 60:
                short_songs.append((song["title"][:60], dur, slug))
            elif dur > 600:
                long_songs.append((song["title"][:60], dur, slug))
        except Exception:
            pass

if missing_audio:
    print(f"\n  ⚠️ Missing audio files ({len(missing_audio)}):")
    for sid in missing_audio[:5]:
        print(f"    - {sid}")
    if len(missing_audio) > 5:
        print(f"    ... and {len(missing_audio) - 5} more")

if short_songs:
    print(f"\n  ⚠️ Songs < 60s ({len(short_songs)}):")
    for title, dur, slug in short_songs:
        print(f"    [{slug}] {title}: {dur:.0f}s")

if long_songs:
    print(f"\n  ⚠️ Songs > 600s ({len(long_songs)}):")
    for title, dur, slug in long_songs:
        print(f"    [{slug}] {title}: {dur:.0f}s")

if not short_songs and not long_songs:
    print("  ✅ All songs within 60-600s range")

# --- 3. Duplicate titles within same producer ---
print("\n[3] Duplicate check:")
dups = []
for slug, songs in producer_songs.items():
    titles = Counter(s["title"] for s in songs)
    for title, count in titles.items():
        if count > 1:
            dups.append((slug, title, count))

if dups:
    print(f"  ⚠️ Duplicate titles ({len(dups)}):")
    for slug, title, count in dups:
        print(f"    [{slug}] \"{title[:60]}\" x{count}")
else:
    print("  ✅ No duplicate titles found")

# --- 4. File size sanity ---
print("\n[4] File size check:")
tiny_files = []
for f in files:
    size_mb = f.stat().st_size / (1024 * 1024)
    if size_mb < 1.0:
        tiny_files.append((f.name, size_mb))

if tiny_files:
    print(f"  ⚠️ Files < 1MB ({len(tiny_files)}):")
    for name, size in tiny_files[:5]:
        print(f"    {name}: {size:.2f}MB")
else:
    print("  ✅ All files >= 1MB")

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
