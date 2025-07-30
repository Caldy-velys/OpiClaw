"""
OpiClaw Training & Visualization Demo
Enhanced panoptic segmentation for deep-sea navigation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm

# Import our enhanced OpiClaw model
from opiclaw_upgrade import (
    PanopticOpiClaw,
    project_to_polar_bev,
    generate_marine_scene,
    evidential_loss,
    MarineLGRSFusion,
)


class MarineDataset(Dataset):
    """Synthetic marine dataset for training"""

    def __init__(self, num_samples=1000, scene_types=None):
        if scene_types is None:
            scene_types = ["hydrothermal_field", "debris_field", "flat_seafloor"]

        self.scene_types = scene_types
        self.num_samples = num_samples
        self.samples = []

        print(f"🌊 Generating {num_samples} synthetic marine scenes...")
        for i in tqdm(range(num_samples)):
            scene_type = np.random.choice(scene_types)
            points = generate_marine_scene(1000, scene_type)
            bev = project_to_polar_bev(points)

            # Generate ground truth labels (simplified for demo)
            gt_semantic = self._generate_semantic_gt(bev, scene_type)
            gt_instance = self._generate_instance_gt(bev, scene_type)

            # Generate marine prompts
            prompt = self._generate_prompt(scene_type)

            self.samples.append(
                {
                    "bev": bev,
                    "semantic_gt": gt_semantic,
                    "instance_gt": gt_instance,
                    "prompt": prompt,
                    "scene_type": scene_type,
                }
            )

    def _generate_semantic_gt(self, bev, scene_type):
        """Generate synthetic semantic ground truth"""
        B, C, H, W = bev.shape
        gt = torch.zeros(H, W, dtype=torch.long)

        # Base seafloor (class 1)
        gt.fill_(1)

        # Add features based on scene type
        if scene_type == "hydrothermal_field":
            # Add vents (class 2) where depth is elevated
            depth = bev[0, 0]
            vent_mask = depth > depth.mean() + depth.std()
            gt[vent_mask] = 2

        elif scene_type == "debris_field":
            # Add debris (class 3) randomly
            debris_mask = torch.rand(H, W) < 0.1
            gt[debris_mask] = 3

        return gt

    def _generate_instance_gt(self, bev, scene_type):
        """Generate synthetic instance ground truth"""
        B, C, H, W = bev.shape
        instance_gt = torch.zeros(H, W, dtype=torch.long)

        if scene_type in ["hydrothermal_field", "debris_field"]:
            # Simple clustered instances
            centers = torch.randint(0, min(H, W), (5, 2))
            for i, center in enumerate(centers):
                r, c = center
                instance_gt[
                    max(0, r - 3) : min(H, r + 4), max(0, c - 3) : min(W, c + 4)
                ] = (i + 1)

        return instance_gt

    def _generate_prompt(self, scene_type):
        """Generate marine-specific prompts"""
        prompts = {
            "hydrothermal_field": [
                "find hydrothermal_vent",
                "detect vent",
                "map hydrothermal_vent",
            ],
            "debris_field": ["detect debris wreck", "find wreck", "avoid debris"],
            "flat_seafloor": ["map seafloor", "navigate seafloor", "explore seafloor"],
        }
        return np.random.choice(prompts[scene_type])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        # Fix batch tensor shapes
        bev = torch.cat([sample["bev"] for sample in batch], dim=0).to(
            device
        )  # Proper batching
        semantic_gt = torch.stack(
            [sample["semantic_gt"] for sample in batch], dim=0
        ).to(device)
        prompts = [sample["prompt"] for sample in batch]

        optimizer.zero_grad()

        # Forward pass with first prompt (simplified for demo)
        outputs = model(bev, prompts[0])

        # Compute evidential loss
        alpha = outputs["semantic_alpha"]
        loss = evidential_loss(alpha, semantic_gt)

        # Add instance loss (simplified)
        center_loss = F.mse_loss(
            outputs["instance_center"], torch.zeros_like(outputs["instance_center"])
        )
        total_loss_batch = loss + 0.1 * center_loss

        total_loss_batch.backward()
        optimizer.step()

        total_loss += total_loss_batch.item()

    return total_loss / len(dataloader)


def evaluate_model(model, dataloader, device):
    """Evaluate model performance"""
    model.eval()
    total_accuracy = 0
    total_uncertainty = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Fix batch tensor shapes
            bev = torch.cat([sample["bev"] for sample in batch], dim=0).to(device)
            semantic_gt = torch.stack(
                [sample["semantic_gt"] for sample in batch], dim=0
            ).to(device)
            prompts = [sample["prompt"] for sample in batch]

            outputs = model(bev, prompts[0])

            # Calculate accuracy
            pred = outputs["semantic_prob"].argmax(1)
            accuracy = (pred == semantic_gt).float().mean()
            total_accuracy += accuracy.item()

            # Calculate uncertainty
            uncertainty = outputs["semantic_uncertainty"].mean()
            total_uncertainty += uncertainty.item()

    return total_accuracy / len(dataloader), total_uncertainty / len(dataloader)


# Custom collate function for proper batching
def marine_collate_fn(batch):
    """Custom collate function for marine dataset"""
    return batch  # Return list of samples, we'll handle batching manually


def visualize_predictions(model, dataset, device, num_samples=4):
    """Visualize model predictions"""
    model.eval()

    fig, axes = plt.subplots(num_samples, 5, figsize=(20, num_samples * 4))

    for i in range(num_samples):
        sample = dataset[i * (len(dataset) // num_samples)]
        bev = sample["bev"].to(device)
        semantic_gt = sample["semantic_gt"]
        prompt = sample["prompt"]
        scene_type = sample["scene_type"]

        with torch.no_grad():
            # Test with and without prompt
            outputs_no_prompt = model(bev.unsqueeze(0), None)
            outputs_with_prompt = model(bev.unsqueeze(0), prompt)

        # Predictions
        pred_no_prompt = outputs_no_prompt["semantic_prob"].argmax(1).squeeze().cpu()
        pred_with_prompt = (
            outputs_with_prompt["semantic_prob"].argmax(1).squeeze().cpu()
        )
        uncertainty_no_prompt = (
            outputs_no_prompt["semantic_uncertainty"].squeeze().cpu()
        )
        uncertainty_with_prompt = (
            outputs_with_prompt["semantic_uncertainty"].squeeze().cpu()
        )

        # Plot
        axes[i, 0].imshow(bev.squeeze().cpu(), cmap="viridis")
        axes[i, 0].set_title(f"Input BEV\n{scene_type}")

        axes[i, 1].imshow(semantic_gt, cmap="tab10")
        axes[i, 1].set_title("Ground Truth")

        axes[i, 2].imshow(pred_no_prompt, cmap="tab10")
        axes[i, 2].set_title("Pred (No Prompt)")

        axes[i, 3].imshow(pred_with_prompt, cmap="tab10")
        axes[i, 3].set_title(f'Pred (Prompt)\n"{prompt}"')

        axes[i, 4].imshow(uncertainty_with_prompt, cmap="hot")
        axes[i, 4].set_title("Uncertainty")

        # Remove axes
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("opiclaw_predictions.png", dpi=150, bbox_inches="tight")
    plt.show()


def main():
    """Main training and evaluation loop"""
    print("🚀 Enhanced OpiClaw Training Demo")
    print("=" * 50)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model
    model = PanopticOpiClaw(in_channels=1, num_classes=4, embed_dim=64)
    model.to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Dataset
    train_dataset = MarineDataset(
        num_samples=500,
        scene_types=["hydrothermal_field", "debris_field", "flat_seafloor"],
    )
    val_dataset = MarineDataset(
        num_samples=100,
        scene_types=["hydrothermal_field", "debris_field", "flat_seafloor"],
    )

    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, collate_fn=marine_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False, collate_fn=marine_collate_fn
    )

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Training loop
    print("\n🎯 Starting training...")
    train_losses = []
    val_accuracies = []
    val_uncertainties = []

    for epoch in range(5):  # Short demo training
        print(f"\nEpoch {epoch + 1}/5")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)

        # Evaluate
        val_acc, val_unc = evaluate_model(model, val_loader, device)
        val_accuracies.append(val_acc)
        val_uncertainties.append(val_unc)

        scheduler.step()

        print(
            f"Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}, Val Uncertainty: {val_unc:.4f}"
        )

    # Plot training curves
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 3, 2)
    plt.plot(val_accuracies)
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.subplot(1, 3, 3)
    plt.plot(val_uncertainties)
    plt.title("Validation Uncertainty")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Uncertainty")

    plt.tight_layout()
    plt.savefig("opiclaw_training_curves.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Visualize predictions
    print("\n📊 Generating prediction visualizations...")
    visualize_predictions(model, val_dataset, device)

    # Language-guided analysis
    print("\n🗣️ Language-Guided Analysis:")
    print("=" * 30)

    # Test prompt sensitivity
    test_sample = val_dataset[0]
    bev = test_sample["bev"].to(device).unsqueeze(0)

    marine_prompts = [
        "find hydrothermal_vent",
        "detect debris wreck",
        "map seafloor sediment",
        "avoid pipeline cable",
    ]

    for prompt in marine_prompts:
        with torch.no_grad():
            outputs = model(bev, prompt)
            uncertainty = outputs["semantic_uncertainty"].mean().item()
            pred_entropy = (
                (-outputs["semantic_prob"] * torch.log(outputs["semantic_prob"] + 1e-8))
                .sum(1)
                .mean()
                .item()
            )

        print(
            f"Prompt: '{prompt:20s}' -> Uncertainty: {uncertainty:.3f}, Entropy: {pred_entropy:.3f}"
        )

    print("\n✅ Enhanced OpiClaw training complete!")
    print("🌊 Ready for deep-sea deployment with ConvViT + LGRS fusion!")


if __name__ == "__main__":
    main()
