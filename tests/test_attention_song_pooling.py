"""Tests for the supervised attention pooling head."""
import torch

from vocaptest.models.attention_song_pooling import AttentionSongClassifier


def test_attention_pooling_masks_padding_and_normalizes_weights():
    model = AttentionSongClassifier(input_dim=4, hidden_dim=3, class_count=2)
    features = torch.randn(2, 4, 4)
    masks = torch.tensor([
        [True, True, False, False],
        [True, True, True, False],
    ])

    logits, weights = model(features, masks)

    assert logits.shape == (2, 2)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2))
    assert torch.equal(weights[~masks], torch.zeros_like(weights[~masks]))
