"""
OpiClaw Utilities
Helper functions for data processing, training, and evaluation
"""

import torch
import torch.nn.functional as F
import numpy as np


def project_to_polar_bev(points, bev_shape=(128, 128), rho_max=50):
    """Enhanced polar BEV projection with optimized memory operations"""
    # Input validation
    if not isinstance(points, torch.Tensor):
        raise TypeError("points must be a torch.Tensor")
    if points.dim() != 2 or points.size(1) != 3:
        raise ValueError(
            f"Expected points to be 2D tensor with shape (N, 3), got {points.shape}"
        )
    if points.size(0) == 0:
        raise ValueError("points tensor cannot be empty")

    # Ensure points are on the same device
    device = points.device

    rho = torch.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    theta = torch.atan2(points[:, 1], points[:, 0])
    theta = (theta + np.pi) / (2 * np.pi)  # Normalize [0,1]
    rho = rho / rho_max  # Normalize [0,1]

    # Bin indices with proper bounds checking
    i = torch.clamp((theta * (bev_shape[1] - 1)), 0, bev_shape[1] - 1).long()
    j = torch.clamp((rho * (bev_shape[0] - 1)), 0, bev_shape[0] - 1).long()

    # Use scatter_add for better memory efficiency
    linear_idx = j * bev_shape[1] + i

    # Initialize tensors on the correct device
    bev_flat = torch.zeros(
        bev_shape[0] * bev_shape[1], dtype=torch.float32, device=device
    )
    count_flat = torch.zeros(
        bev_shape[0] * bev_shape[1], dtype=torch.float32, device=device
    )

    # Use scatter_add_ for more efficient accumulation
    bev_flat.scatter_add_(0, linear_idx, points[:, 2])
    count_flat.scatter_add_(0, linear_idx, torch.ones_like(points[:, 2]))

    # Reshape back to 2D
    bev = bev_flat.reshape(bev_shape)
    count = count_flat.reshape(bev_shape)

    # Avoid division by zero with in-place operations
    mask = count > 0
    bev[mask] = bev[mask] / count[mask]

    return bev.unsqueeze(0).unsqueeze(0)  # (1,1,H,W) for batch


def generate_marine_scene(N=1000, scene_type="hydrothermal_field"):
    """Generate diverse marine scenes for testing"""
    if scene_type == "hydrothermal_field":
        # Clustered vents with elevated seafloor
        centers = torch.randn(5, 2) * 10  # 5 vent clusters
        points = []
        for center in centers:
            cluster_points = torch.randn(N // 5, 2) * 2 + center
            x, y = cluster_points[:, 0], cluster_points[:, 1]
            z = -15 + 8 * torch.exp(-(cluster_points**2).sum(1) / 10)  # Elevated vent
            points.append(torch.stack([x, y, z], dim=1))
        return torch.cat(points, dim=0)

    elif scene_type == "debris_field":
        # Scattered debris on flat seafloor
        x = torch.rand(N) * 50 - 25
        y = torch.rand(N) * 50 - 25
        z = torch.ones(N) * -20  # Flat seafloor
        # Add random debris
        debris_mask = torch.rand(N) < 0.1
        z[debris_mask] += torch.rand(debris_mask.sum()) * 3
        return torch.stack([x, y, z], dim=1)

    else:  # default flat seafloor
        x = torch.rand(N) * 50 - 25
        y = torch.rand(N) * 50 - 25
        z = torch.ones(N) * -10 + torch.randn(N) * 0.5  # Slight noise
        return torch.stack([x, y, z], dim=1)


def evidential_loss(alpha, target, lambda_reg=0.1):
    """Optimized evidential loss with vectorized operations"""
    # Ensure alpha has correct shape and avoid unnecessary computations
    device = alpha.device

    # Dirichlet strength
    S = alpha.sum(1, keepdim=True)  # (B, 1, H, W)

    # Convert target to one-hot if needed
    if target.dim() == 3:  # (B, H, W)
        # More efficient one-hot encoding
        num_classes = alpha.shape[1]
        target_one_hot = torch.zeros_like(alpha, device=device)
        target_one_hot.scatter_(1, target.long().unsqueeze(1), 1.0)
    else:
        target_one_hot = target

    # Vectorized digamma computation - compute once and reuse
    digamma_S = torch.digamma(S)
    digamma_alpha = torch.digamma(alpha)

    # Cross-entropy term - fully vectorized
    ce_term = (target_one_hot * (digamma_S - digamma_alpha)).sum(dim=1)

    # KL regularization - vectorized computation
    kl_term = lambda_reg * ((alpha - target_one_hot) * (digamma_alpha - digamma_S)).sum(
        dim=1
    )

    # Return mean loss
    return (ce_term + kl_term).mean()


def marine_contrastive_loss(embeddings, centers, targets, margin=1.0):
    """Contrastive loss for instance embeddings in marine context"""
    # Simplified implementation - use proper contrastive/triplet loss in production
    return F.mse_loss(embeddings, targets)  # Placeholder


def panoptic_fusion(sem_pred, center, offset, threshold=0.5):
    """Combine semantic and instance predictions into panoptic output"""
    # Find object centers
    center_mask = center > threshold

    # Simple clustering based on predicted centers (for prototype)
    # In practice, you'd use more sophisticated clustering like watershed
    instance_map = torch.zeros_like(sem_pred)
    instance_id = 1

    # This is a simplified version - real implementation would use
    # proper clustering algorithms like DBSCAN or watershed
    for i in range(center.shape[-2]):
        for j in range(center.shape[-1]):
            if center_mask[0, 0, i, j]:
                instance_map[0, i, j] = instance_id
                instance_id += 1

    return instance_map


def calculate_panoptic_quality(semantic_pred, instance_pred, semantic_gt, instance_gt):
    """Calculate Panoptic Quality (PQ) metric"""
    # Simplified PQ calculation
    # In practice, use proper PQ implementation from COCO or similar
    semantic_acc = (semantic_pred == semantic_gt).float().mean()
    instance_iou = torch.tensor(0.5)  # Placeholder
    return semantic_acc * instance_iou


def calculate_uncertainty_metrics(uncertainty, accuracy):
    """Calculate uncertainty calibration metrics"""
    # Expected Calibration Error (ECE)
    n_bins = 10
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (uncertainty > bin_lower) & (uncertainty <= bin_upper)
        if in_bin.sum() > 0:
            bin_acc = accuracy[in_bin].mean()
            bin_conf = uncertainty[in_bin].mean()
            ece += (bin_conf - bin_acc).abs() * in_bin.float().mean()

    return ece
