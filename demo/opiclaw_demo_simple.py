"""
OpiClaw Enhanced Demo - Lightweight Showcase
Demonstrates ConvViT + LGRS fusion for deep-sea panoptic segmentation
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from opiclaw_upgrade import (
    PanopticOpiClaw, project_to_polar_bev, generate_marine_scene
)

def demonstrate_language_guidance():
    """Demonstrate language-guided refinement capabilities"""
    print("🗣️ Language-Guided Refinement Demonstration")
    print("=" * 50)
    
    # Initialize model
    model = PanopticOpiClaw(in_channels=1, num_classes=4, embed_dim=64)
    model.eval()
    
    # Generate different marine scenes
    scenes = {
        "Hydrothermal Vent Field": generate_marine_scene(1000, "hydrothermal_field"),
        "Debris Field": generate_marine_scene(1000, "debris_field"),
        "Flat Seafloor": generate_marine_scene(1000, "flat_seafloor")
    }
    
    # Marine task prompts
    prompts = [
        "find hydrothermal_vent",
        "detect debris wreck",
        "map seafloor sediment",
        "avoid pipeline cable",
        None  # No prompt baseline
    ]
    
    print(f"Testing {len(scenes)} scene types with {len(prompts)} prompt variations...\n")
    
    results = {}
    
    for scene_name, points in scenes.items():
        print(f"🌊 Scene: {scene_name}")
        bev = project_to_polar_bev(points)
        
        scene_results = {}
        
        for prompt in prompts:
            with torch.no_grad():
                outputs = model(bev, prompt)
            
            # Calculate metrics
            uncertainty = outputs['semantic_uncertainty'].mean().item()
            confidence = outputs['semantic_prob'].max(1)[0].mean().item()
            entropy = (-outputs['semantic_prob'] * torch.log(outputs['semantic_prob'] + 1e-8)).sum(1).mean().item()
            
            prompt_str = prompt if prompt else "No Prompt"
            scene_results[prompt_str] = {
                'uncertainty': uncertainty,
                'confidence': confidence,
                'entropy': entropy
            }
            
            print(f"  📝 '{prompt_str:20s}' -> Uncertainty: {uncertainty:.3f}, Confidence: {confidence:.3f}")
        
        results[scene_name] = scene_results
        print()
    
    return results

def demonstrate_uncertainty_analysis():
    """Demonstrate uncertainty-aware predictions"""
    print("🎯 Uncertainty-Aware Analysis")
    print("=" * 30)
    
    model = PanopticOpiClaw(in_channels=1, num_classes=4, embed_dim=64)
    model.eval()
    
    # Create challenging scenarios
    scenarios = {
        "Clean Hydrothermal Field": generate_marine_scene(1000, "hydrothermal_field"),
        "Noisy Mixed Scene": torch.cat([
            generate_marine_scene(300, "hydrothermal_field"),
            generate_marine_scene(300, "debris_field"),
            generate_marine_scene(400, "flat_seafloor")
        ], dim=0)
    }
    
    for scenario_name, points in scenarios.items():
        print(f"\n🔍 Analyzing: {scenario_name}")
        bev = project_to_polar_bev(points)
        
        with torch.no_grad():
            outputs = model(bev, "find hydrothermal_vent")
        
        # Uncertainty analysis
        uncertainty = outputs['semantic_uncertainty'].squeeze()
        prob = outputs['semantic_prob'].squeeze()
        
        # Statistics
        unc_mean = uncertainty.mean().item()
        unc_std = uncertainty.std().item()
        high_unc_ratio = (uncertainty > unc_mean + unc_std).float().mean().item()
        
        print(f"  Mean Uncertainty: {unc_mean:.3f} ± {unc_std:.3f}")
        print(f"  High Uncertainty Regions: {high_unc_ratio:.1%}")
        print(f"  Prediction Entropy: {(-prob * torch.log(prob + 1e-8)).sum(0).mean():.3f}")

def demonstrate_architecture_components():
    """Demonstrate individual architecture components"""
    print("\n🏗️ Architecture Component Analysis")
    print("=" * 35)
    
    model = PanopticOpiClaw(in_channels=1, num_classes=4, embed_dim=64)
    
    # Model complexity analysis
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # Component breakdown
    components = {
        'Encoder (ConvViT)': sum(p.numel() for p in [*model.enc1.parameters(), 
                                                     *model.vit_block1.parameters(),
                                                     *model.enc2.parameters(),
                                                     *model.vit_block2.parameters()]),
        'LGRS Fusion': sum(p.numel() for p in model.lgrs.parameters()),
        'Decoder': sum(p.numel() for p in [*model.dec1.parameters(), *model.dec2.parameters()]),
        'Semantic Head': sum(p.numel() for p in model.semantic_head.parameters()),
        'Instance Head': sum(p.numel() for p in model.instance_head.parameters())
    }
    
    print("\nComponent Parameter Breakdown:")
    for name, params in components.items():
        percentage = (params / total_params) * 100
        print(f"  {name:15s}: {params:6,} ({percentage:4.1f}%)")

def create_visualization():
    """Create comprehensive visualization"""
    print("\n📊 Creating Comprehensive Visualization")
    print("=" * 40)
    
    model = PanopticOpiClaw(in_channels=1, num_classes=4, embed_dim=64)
    model.eval()
    
    # Generate test scenarios
    scenarios = [
        ("Hydrothermal Vent Field", generate_marine_scene(1000, "hydrothermal_field"), "find hydrothermal_vent"),
        ("Debris Field", generate_marine_scene(1000, "debris_field"), "detect debris wreck"),
        ("Mixed Environment", torch.cat([
            generate_marine_scene(500, "hydrothermal_field"),
            generate_marine_scene(500, "debris_field")
        ], dim=0), "navigate seafloor"),
    ]
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    for i, (scene_name, points, prompt) in enumerate(scenarios):
        bev = project_to_polar_bev(points)
        
        with torch.no_grad():
            outputs_no_prompt = model(bev, None)
            outputs_with_prompt = model(bev, prompt)
        
        # Extract outputs
        sem_pred_no = outputs_no_prompt['semantic_prob'].argmax(1).squeeze()
        sem_pred_with = outputs_with_prompt['semantic_prob'].argmax(1).squeeze()
        uncertainty = outputs_with_prompt['semantic_uncertainty'].squeeze()
        center_heatmap = outputs_with_prompt['instance_center'].squeeze()
        
        # Plot
        axes[i, 0].imshow(bev.squeeze(), cmap='viridis')
        axes[i, 0].set_title(f'Input BEV\n{scene_name}')
        
        axes[i, 1].imshow(sem_pred_no, cmap='tab10', vmin=0, vmax=3)
        axes[i, 1].set_title('Semantic (No Prompt)')
        
        axes[i, 2].imshow(sem_pred_with, cmap='tab10', vmin=0, vmax=3)
        axes[i, 2].set_title(f'Semantic (Prompted)\n"{prompt}"')
        
        axes[i, 3].imshow(uncertainty, cmap='hot')
        axes[i, 3].set_title('Uncertainty Map')
        
        # Remove axes
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.suptitle('OpiClaw Enhanced: ConvViT + LGRS Fusion for Deep-Sea Segmentation', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('opiclaw_enhanced_demo.png', dpi=150, bbox_inches='tight')
    print("✅ Visualization saved as 'opiclaw_enhanced_demo.png'")
    plt.show()

def main():
    """Main demonstration function"""
    print("🌊 Enhanced OpiClaw Demonstration")
    print("Deep-Sea Panoptic Segmentation with ConvViT + LGRS")
    print("=" * 60)
    
    # Core demonstrations
    results = demonstrate_language_guidance()
    demonstrate_uncertainty_analysis()
    demonstrate_architecture_components()
    create_visualization()
    
    # Summary insights
    print("\n🎯 Key Insights:")
    print("=" * 15)
    print("✅ ConvViT blocks provide spatial attention for sonar feature extraction")
    print("✅ LGRS fusion enables task-specific marine vocabulary guidance")
    print("✅ Evidential uncertainty quantifies prediction confidence")
    print("✅ Polar BEV projection handles radial sonar geometry")
    print("✅ Panoptic heads provide both semantic and instance segmentation")
    
    print("\n🚀 Next Steps for Production:")
    print("=" * 25)
    print("🔬 Train on real NOAA bathymetry data")
    print("⚡ Optimize for real-time AUV deployment")
    print("🌐 Integrate multi-modal sensor fusion")
    print("🧠 Add temporal consistency for navigation")
    print("🛡️ Implement open-set unknown object detection")
    
    print("\n🌊 Enhanced OpiClaw ready for the abyss! 🤖")

if __name__ == "__main__":
    main() 