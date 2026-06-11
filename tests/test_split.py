from vocaptest.data.metadata_schema import EmbeddingRecord
from vocaptest.training.split import split_by_song


def make_records():
    return [
        EmbeddingRecord(
            segment_id=f"{producer}_{song}_segment",
            song_id=f"{producer}_{song}",
            producer_slug=producer,
            model_backend="test",
            embedding_path="unused.npy",
            embedding_dim=2,
        )
        for producer in ["a", "b", "c"]
        for song in range(10)
    ]


def test_split_by_song_is_deterministic_and_stratified():
    records = make_records()
    first = split_by_song(records, seed=123)
    second = split_by_song(list(reversed(records)), seed=123)

    assert first == second
    for split in first:
        assert {song_id.split("_")[0] for song_id in split} == {"a", "b", "c"}
    assert not (set(first[0]) & set(first[1]))
    assert not (set(first[0]) & set(first[2]))
