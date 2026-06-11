import numpy as np

from vocaptest.data.curation import curate_embedding_records
from vocaptest.data.metadata_schema import EmbeddingRecord
from vocaptest.features.extract_embeddings import load_all_embeddings_aligned


def make_record(tmp_path, segment_id, song_id, producer_slug, value):
    path = tmp_path / f"{segment_id}.npy"
    np.save(path, np.asarray(value, dtype=np.float32))
    return EmbeddingRecord(
        segment_id=segment_id,
        song_id=song_id,
        producer_slug=producer_slug,
        model_backend="test_backend",
        embedding_path=str(path),
        embedding_dim=len(value),
    )


def test_embedding_loader_preserves_record_alignment(tmp_path):
    missing = EmbeddingRecord(
        segment_id="missing",
        song_id="song_missing",
        producer_slug="wrong_label",
        model_backend="test_backend",
        embedding_path=str(tmp_path / "missing.npy"),
        embedding_dim=2,
    )
    valid = make_record(tmp_path, "valid", "song_valid", "right_label", [1.0, 2.0])

    embeddings, loaded_records = load_all_embeddings_aligned([missing, valid])

    assert embeddings.shape == (1, 2)
    assert loaded_records == [valid]
    assert np.allclose(embeddings[0], [1.0, 2.0])


def test_curation_excludes_bad_song_and_noncanonical_duplicate(tmp_path):
    records = [
        make_record(tmp_path, "a0", "song_a", "producer", [1.0, 0.0]),
        make_record(tmp_path, "b0", "song_b", "producer", [1.0, 0.0]),
        make_record(tmp_path, "bad0", "song_bad", "producer", [0.0, 1.0]),
    ]
    config = {
        "curation": {"canonical_only": True, "minimum_songs_per_producer": 1},
        "exclude_songs": {
            "song_bad": {"category": "cover", "reason": "test exclusion"}
        },
        "work_groups": [{
            "work_id": "shared_work",
            "canonical_song_id": "song_a",
            "members": ["song_a", "song_b"],
            "reason": "same work",
        }],
    }

    result = curate_embedding_records(
        records,
        {"song_a": "A", "song_b": "B", "song_bad": "Bad"},
        config,
    )

    assert {record.song_id for record in result.records} == {"song_a"}
    assert result.records[0].work_id == "shared_work"
    assert result.summary["accepted_songs"] == 1
    assert result.summary["excluded_songs"] == 2
