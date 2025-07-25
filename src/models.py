"""
OpiClaw Model Implementation
Core neural network architectures for deep-sea panoptic segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict

# Evidential Head (Dirichlet-based uncertainty)
class EvidentialHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, 1)  # Output evidence
        self.dropout = nn.Dropout2d(0.1)  # Regularization

    def forward(self, x):
        x = self.dropout(x)
        evidence = F.softplus(self.conv(x))  # Non-negative evidence
        alpha = evidence + 1  # Dirichlet params
        S = alpha.sum(1, keepdim=True)
        prob = alpha / S  # Predicted probs
        u = alpha.shape[1] / S  # Uncertainty (K / sum alpha)
        return prob, u, alpha

# Instance Segmentation Head with improved clustering
class InstanceHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.center_head = nn.Conv2d(in_channels, 1, 1)  # Object centers
        self.offset_head = nn.Conv2d(in_channels, 2, 1)  # Center offsets
        self.embedding_head = nn.Conv2d(in_channels, 8, 1)  # Instance embeddings
        
    def forward(self, x):
        center = torch.sigmoid(self.center_head(x))
        offset = self.offset_head(x)
        embedding = F.normalize(self.embedding_head(x), dim=1)  # L2 normalized
        return center, offset, embedding

# Enhanced ConvViT Block with marine-specific adaptations
class MarineConvViTBlock(nn.Module):
    def __init__(self, in_channels, embed_dim=64, num_heads=4, dropout=0.1):
        super().__init__()
        self.conv_proj = nn.Conv2d(in_channels, embed_dim, 1)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.fc1 = nn.Linear(embed_dim, embed_dim * 4)
        self.fc2 = nn.Linear(embed_dim * 4, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Marine-specific: radial position encoding for sonar
        self.register_buffer('radial_pos_enc', self._create_radial_pos_encoding(128, embed_dim))

    def _create_radial_pos_encoding(self, max_len, embed_dim):
        """Create radial position encoding for polar BEV"""
        pe = torch.zeros(max_len, max_len, embed_dim)
        for r in range(max_len):
            for theta in range(max_len):
                pos_r = r / max_len
                pos_theta = theta / max_len
                for i in range(0, embed_dim, 4):
                    pe[r, theta, i] = np.sin(pos_r * np.pi)
                    pe[r, theta, i+1] = np.cos(pos_r * np.pi)
                    if i+2 < embed_dim:
                        pe[r, theta, i+2] = np.sin(pos_theta * 2 * np.pi)
                    if i+3 < embed_dim:
                        pe[r, theta, i+3] = np.cos(pos_theta * 2 * np.pi)
        return pe

    def forward(self, x):
        B, C, H, W = x.shape
        proj = self.conv_proj(x)  # (B, embed_dim, H, W)
        
        # Add radial position encoding
        if H <= self.radial_pos_enc.shape[0] and W <= self.radial_pos_enc.shape[1]:
            pos_enc = self.radial_pos_enc[:H, :W].permute(2, 0, 1).unsqueeze(0)  # (1, embed_dim, H, W)
            proj = proj + pos_enc
        
        # Reshape for attention: (B, HW, embed_dim)
        proj_flat = proj.flatten(2).permute(0, 2, 1)
        
        # Self-attention with residual
        attn_out, attn_weights = self.attn(proj_flat, proj_flat, proj_flat)
        proj_flat = self.norm1(attn_out + proj_flat)
        
        # FFN with residual
        ffn_out = self.fc2(F.gelu(self.fc1(proj_flat)))
        ffn_out = self.dropout(ffn_out)
        proj_flat = self.norm2(ffn_out + proj_flat)
        
        # Reshape back to spatial
        return proj_flat.permute(0, 2, 1).view(B, -1, H, W)

# Enhanced LGRS with marine vocabulary
class MarineLGRSFusion(nn.Module):
    def __init__(self, feat_dim, embed_dim=64):
        super().__init__()
        # Marine-specific vocabulary
        self.marine_vocab = {
            'hydrothermal_vent': 0, 'manganese_nodule': 1, 'seafloor': 2, 'debris': 3,
            'wreck': 4, 'coral': 5, 'sediment': 6, 'rock': 7, 'cable': 8, 'pipeline': 9,
            'seamount': 10, 'trench': 11, 'canyon': 12, 'ridge': 13, 'find': 14,
            'detect': 15, 'avoid': 16, 'map': 17, 'navigate': 18, 'explore': 19
        }
        
        vocab_size = len(self.marine_vocab)
        self.text_embed = nn.Embedding(vocab_size, embed_dim)
        
        # Project features to match text embedding dimension
        self.feat_proj = nn.Linear(feat_dim, embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, 4, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
        # Project back to original feature dimension
        self.out_proj = nn.Linear(embed_dim, feat_dim)
        
    def tokenize_marine_prompt(self, text_prompt: str) -> torch.Tensor:
        """Convert marine text prompt to token tensor"""
        tokens = []
        words = text_prompt.lower().split()
        for word in words:
            if word in self.marine_vocab:
                tokens.append(self.marine_vocab[word])
        return torch.tensor([tokens]) if tokens else torch.tensor([[0]])  # fallback

    def forward(self, features, prompt):
        # Handle text prompt conversion
        if isinstance(prompt, str):
            prompt = self.tokenize_marine_prompt(prompt)
        
        if prompt.dim() == 1:
            prompt = prompt.unsqueeze(0)
            
        B, C, H, W = features.shape
        
        # Text embedding
        text_emb = self.text_embed(prompt)  # (B, seq_len, embed_dim)
        text_emb = text_emb.mean(1, keepdim=True)  # (B, 1, embed_dim)
        
        # Feature flattening and projection
        feat_flat = features.flatten(2).permute(0, 2, 1)  # (B, HW, C)
        feat_proj = self.feat_proj(feat_flat)  # (B, HW, embed_dim)
        
        # Cross-attention: query=features, key=value=text
        fused, _ = self.cross_attn(feat_proj, text_emb, text_emb)
        fused = self.norm(fused + feat_proj)  # Residual connection
        
        # Project back to original feature dimension
        fused = self.out_proj(fused)  # (B, HW, C)
        
        return fused.permute(0, 2, 1).view(B, C, H, W) + features  # Residual with original

# Main Architecture
class PanopticOpiClaw(nn.Module):
    """Complete panoptic segmentation model with ConvViT and LGRS"""
    
    def __init__(self, in_channels=1, num_classes=3, embed_dim=64):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.vit_block1 = MarineConvViTBlock(32, embed_dim)
        self.enc2 = nn.Conv2d(embed_dim, 128, 3, padding=1)
        self.vit_block2 = MarineConvViTBlock(128, embed_dim * 2)
        
        # Decoder with LGRS - pass the correct feature dimension
        feat_dim = embed_dim * 2 + embed_dim  # Concatenated features dimension
        self.lgrs = MarineLGRSFusion(feat_dim, embed_dim)
        self.dec1 = nn.Conv2d(feat_dim, 64, 3, padding=1)
        self.dec2 = nn.Conv2d(64, 32, 3, padding=1)
        
        # Panoptic heads
        self.semantic_head = EvidentialHead(32, num_classes)
        self.instance_head = InstanceHead(32)
        
    def forward(self, x, prompt: Optional[str] = None):
        # Encoder path
        e1 = F.relu(self.enc1(x))
        e1_vit = self.vit_block1(e1)
        
        e2 = F.max_pool2d(F.relu(self.enc2(e1_vit)), 2)
        e2_vit = self.vit_block2(e2)
        
        # Decoder path with LGRS
        d1 = F.interpolate(e2_vit, scale_factor=2, mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e1_vit], dim=1)
        
        if prompt is not None:
            d1 = self.lgrs(d1, prompt)
            
        d1 = F.relu(self.dec1(d1))
        features = F.relu(self.dec2(d1))
        
        # Panoptic outputs
        sem_prob, sem_u, alpha = self.semantic_head(features)
        center, offset, embedding = self.instance_head(features)
        
        return {
            'semantic_prob': sem_prob,
            'semantic_uncertainty': sem_u,
            'semantic_alpha': alpha,
            'instance_center': center,
            'instance_offset': offset,
            'instance_embedding': embedding
        } 