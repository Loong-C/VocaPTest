import torch

from vocaptest.models.projection_head import (
    ProjectionHead,
    supervised_contrastive_loss,
)


def test_projection_head_shapes_and_contrastive_loss():
    model = ProjectionHead(input_dim=8, projection_dim=4, class_count=3)
    inputs = torch.randn(6, 8)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    logits, projected = model(inputs)
    loss = supervised_contrastive_loss(projected, labels)

    assert logits.shape == (6, 3)
    assert projected.shape == (6, 4)
    assert torch.isfinite(loss)
    assert loss >= 0
