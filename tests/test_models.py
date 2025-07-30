"""
Tests for OpiClaw models
"""

import torch
import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models import PanopticOpiClaw, EvidentialHead, InstanceHead, MarineConvViTBlock
from utils import project_to_polar_bev, generate_marine_scene, evidential_loss


def test_evidential_head():
    """Test EvidentialHead forward pass"""
    head = EvidentialHead(in_channels=32, num_classes=4)
    x = torch.randn(2, 32, 64, 64)

    prob, u, alpha = head(x)

    assert prob.shape == (2, 4, 64, 64)
    assert u.shape == (2, 1, 64, 64)
    assert alpha.shape == (2, 4, 64, 64)
    assert torch.all(prob >= 0) and torch.all(prob <= 1)
    assert torch.all(u >= 0)


def test_instance_head():
    """Test InstanceHead forward pass"""
    head = InstanceHead(in_channels=32)
    x = torch.randn(2, 32, 64, 64)

    center, offset, embedding = head(x)

    assert center.shape == (2, 1, 64, 64)
    assert offset.shape == (2, 2, 64, 64)
    assert embedding.shape == (2, 8, 64, 64)
    assert torch.all(center >= 0) and torch.all(center <= 1)


def test_marine_convvit_block():
    """Test MarineConvViTBlock forward pass"""
    block = MarineConvViTBlock(in_channels=32, embed_dim=64)
    x = torch.randn(2, 32, 64, 64)

    output = block(x)

    assert output.shape == (2, 64, 64, 64)


def test_panoptic_opiclaw():
    """Test complete PanopticOpiClaw model"""
    model = PanopticOpiClaw(in_channels=1, num_classes=4, embed_dim=64)
    x = torch.randn(2, 1, 128, 128)

    # Test without prompt
    outputs = model(x)

    assert "semantic_prob" in outputs
    assert "semantic_uncertainty" in outputs
    assert "instance_center" in outputs
    assert "instance_offset" in outputs
    assert "instance_embedding" in outputs

    # Test with prompt
    outputs = model(x, prompt="find hydrothermal_vent")

    assert "semantic_prob" in outputs
    assert "semantic_uncertainty" in outputs


def test_polar_bev_projection():
    """Test polar BEV projection"""
    points = torch.randn(1000, 3) * 10
    bev = project_to_polar_bev(points, bev_shape=(64, 64))

    assert bev.shape == (1, 1, 64, 64)
    assert not torch.isnan(bev).any()


def test_marine_scene_generation():
    """Test marine scene generation"""
    points = generate_marine_scene(1000, "hydrothermal_field")

    assert points.shape == (1000, 3)
    assert not torch.isnan(points).any()


def test_evidential_loss():
    """Test evidential loss calculation"""
    alpha = torch.randn(2, 4, 64, 64) + 1  # Ensure positive
    target = torch.randint(0, 4, (2, 64, 64))

    loss = evidential_loss(alpha, target)

    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0


if __name__ == "__main__":
    pytest.main([__file__])
