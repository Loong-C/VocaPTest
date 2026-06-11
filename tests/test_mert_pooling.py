import torch

from vocaptest.models.mert_embedder import mean_pool_hidden


def test_mean_pool_hidden_resizes_raw_sample_mask():
    hidden = torch.tensor([[
        [1.0, 0.0],
        [3.0, 0.0],
        [9.0, 0.0],
    ]])
    raw_sample_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0]])

    pooled = mean_pool_hidden(hidden, raw_sample_mask, model=object())

    assert pooled.shape == (1, 2)
    assert torch.allclose(pooled, torch.tensor([[2.0, 0.0]]))
