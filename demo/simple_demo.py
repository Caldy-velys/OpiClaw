"""
OpiClaw Simple Demo
Quick demonstration of the enhanced OpiClaw model
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import matplotlib.pyplot as plt
from models import PanopticOpiClaw
from utils import project_to_polar_bev, generate_marine_scene

def main():
    """Simple demonstration of OpiClaw capabilities"""
    print("🌊 OpiClaw Simple Demo")
    print("=" * 30)
    
    # Initialize model
    model = PanopticOpiClaw(in_channels=1, num_classes=4, embed_dim=64)
    model.eval()
    
    print(f"✅ Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Generate test scene
    points = generate_marine_scene(1000, "hydrothermal_field")
    bev = project_to_polar_bev(points)
    
    print("✅ Generated synthetic hydrothermal field")
    
    # Test with different prompts
    prompts = [
        "find hydrothermal_vent",
        "detect debris wreck",
        None  # No prompt
    ]
    
    print("\n🔍 Testing language-guided refinement:")
    
    for prompt in prompts:
        with torch.no_grad():
            outputs = model(bev, prompt)
        
        uncertainty = outputs['semantic_uncertainty'].mean().item()
        confidence = outputs['semantic_prob'].max().item()
        
        prompt_str = prompt if prompt else "No prompt"
        print(f"  📝 '{prompt_str}' -> Uncertainty: {uncertainty:.3f}, Confidence: {confidence:.3f}")
    
    print("\n🎯 Demo completed successfully!")
    print("🌊 OpiClaw is ready for deep-sea deployment!")

if __name__ == "__main__":
    main() 