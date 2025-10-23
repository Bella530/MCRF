"""
Multimodal Fusion Network for Text and Image Data

This module implements a deep learning model for multimodal data fusion,
combining text and image features using attention mechanisms and cross-modal interactions.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class FirstChannelModule(nn.Module):
    """
    First Channel Module for multimodal feature fusion.
    
    This module handles the initial fusion of two modalities (text and image)
    using cross-modal attention and dynamic gating mechanisms.
    """
    
    def __init__(self, input_dim_A: int, input_dim_B: int, embed_dim: int, 
                 num_heads: int, fusion_strategy: str = "dynamic_gate"):
        """
        Initialize FirstChannelModule.
        
        Args:
            input_dim_A: Input feature dimension for modality A (text)
            input_dim_B: Input feature dimension for modality B (image)
            embed_dim: Embedding dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            fusion_strategy: Fusion strategy ['add', 'concat', 'dynamic_gate']
        """
        super().__init__()
        
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
            
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.fusion_strategy = fusion_strategy

        # Modality alignment projection layers
        self.embedding_a = nn.Sequential(
            nn.Linear(input_dim_A, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.8)
        )
        
        self.embedding_b = nn.Sequential(
            nn.Linear(input_dim_B, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.8)
        )

        # Dynamic fusion gate
        if fusion_strategy == 'dynamic_gate':
            self.gate_network = nn.Sequential(
                nn.Linear(2 * embed_dim, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim),
                nn.Dropout(0.4),
            )

        # Cross-modal attention and projection layers
        self.reshape = nn.Linear(input_dim_A, embed_dim)
        self.proj_q_A = nn.Linear(embed_dim, embed_dim)
        self.proj_kv_B = nn.Linear(input_dim_B, 2 * embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.add_norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, input_A: torch.Tensor, input_B: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_A: Modality A input tensor of shape [batch, seq_len, input_dim_A]
            input_B: Modality B input tensor of shape [batch, seq_len, input_dim_B]
            
        Returns:
            Output tensor of shape [batch, seq_len, embed_dim]
        """
        # Step 1: Modality projection
        A_embed = self.embedding_a(input_A)
        B_embed = self.embedding_b(input_B)
        
        # Step 2: Fusion strategy
        if self.fusion_strategy == 'add':
            fused = A_embed + B_embed
        elif self.fusion_strategy == 'concat':
            fused = torch.cat([A_embed, B_embed], dim=-1)
        elif self.fusion_strategy == 'dynamic_gate':
            gate_input = torch.cat([A_embed, B_embed], dim=-1)
            gate = self.gate_network(gate_input)
            fused = gate * A_embed + (1 - gate) * B_embed
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

        # Step 3: Reshape and cross-modal attention
        input_reshaped = self.reshape(input_A)
        reshaped = input_reshaped + fused
        q_reshaped = self.proj_q_A(reshaped)
        
        # Generate keys and values from modality B
        kv_B = self.proj_kv_B(input_B)
        k_B, v_B = torch.chunk(kv_B, 2, dim=-1)
        
        # Step 4: Cross-modal attention
        cross_output, _ = self.cross_attn(
            query=q_reshaped,
            key=k_B,
            value=v_B,
            need_weights=False
        )
        
        # Step 5: Residual connection and normalization
        output = reshaped + self.add_norm(cross_output)
        
        return output


class LocalEnhancement(nn.Module):
    """
    Local Enhancement Module for feature refinement.
    
    Enhances local features using self-attention and adaptive pooling.
    """
    
    def __init__(self, input_dim: int, embed_dim: int, num_heads: int):
        """
        Initialize LocalEnhancement module.
        
        Args:
            input_dim: Input feature dimension
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
        """
        super().__init__()
        
        self.local_enhancement = nn.Linear(input_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.max_pooling = nn.AdaptiveMaxPool1d(256)
        self.add_norm = nn.LayerNorm(embed_dim)
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_tensor: Input tensor of shape [batch, seq_len, input_dim]
            
        Returns:
            Enhanced tensor of shape [batch, seq_len, embed_dim]
        """
        enhanced_features = self.local_enhancement(input_tensor)
        q = self.q_proj(enhanced_features)
        k = self.k_proj(enhanced_features)
        v = self.v_proj(enhanced_features)
        
        # Self-attention mechanism
        attn_output, _ = self.self_attention(q, k, v)
        
        # Residual connection and normalization
        norm_output = self.add_norm(attn_output + enhanced_features)
        
        # Adaptive pooling
        pooling_output = self.max_pooling(norm_output)
        
        return pooling_output


class SecondChannelModule(nn.Module):
    """
    Second Channel Module for advanced multimodal interaction.
    
    Implements multi-head cross-modal attention with feed-forward networks.
    """
    
    def __init__(self, input_dim_A: int, input_dim_B: int, embed_dim: int, 
                 num_heads: int, d_ff: int, dropout: float = 0.2):
        """
        Initialize SecondChannelModule.
        
        Args:
            input_dim_A: Input dimension for modality A
            input_dim_B: Input dimension for modality B
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
            
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.d_ff = d_ff

        # Projection layers
        self.multimodal_proj = nn.Linear(input_dim_A, embed_dim)
        self.proj_q_A = nn.Linear(embed_dim, embed_dim)
        self.proj_kv_B = nn.Linear(input_dim_B, 2 * embed_dim)
        self.add_norm = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, input_A: torch.Tensor, input_B: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_A: Modality A input tensor
            input_B: Modality B input tensor
            
        Returns:
            Output tensor
        """
        batch_size, seq_len, _ = input_A.shape
        
        # Project modality A
        embed_A = self.multimodal_proj(input_A)
        
        # Generate queries, keys, and values
        q_A = self.proj_q_A(embed_A)
        kv_B = self.proj_kv_B(input_B)
        k_B, v_B = torch.chunk(kv_B, 2, dim=-1)
        
        # Multi-head attention
        Q = q_A.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = k_B.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = v_B.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention computation
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        
        # Merge multi-head outputs
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.embed_dim)
        
        # Residual connection and normalization
        add_norm = self.add_norm(embed_A + context)
        
        # Feed-forward network
        ff_output = self.ffn(add_norm)
        
        # Final residual connection
        output = self.add_norm(add_norm + ff_output)
        
        return output


class GlobalFeatureExtractor(nn.Module):
    """
    Global Feature Extractor for capturing long-range dependencies.
    
    Processes features in windows to extract global context information.
    """
    
    def __init__(self, input_features_dim: int, window_size: int, 
                 num_heads: int, embed_dim: int):
        """
        Initialize GlobalFeatureExtractor.
        
        Args:
            input_features_dim: Input feature dimension
            window_size: Size of processing window
            num_heads: Number of attention heads
            embed_dim: Embedding dimension
        """
        super().__init__()
        
        self.window_size = window_size
        self.projection = nn.Linear(input_features_dim, embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, features: torch.Tensor, window_size: int) -> torch.Tensor:
        """
        Forward pass with window-based processing.
        
        Args:
            features: Input features tensor
            window_size: Processing window size
            
        Returns:
            Globally enhanced features
        """
        x = self.projection(features)
        seq_length = x.size(1)
        
        # Ensure valid window size
        window_size = min(window_size, seq_length)
        if window_size <= 0:
            raise ValueError(f"Invalid window size: {window_size}")
        
        # Split into windows
        num_full_blocks = seq_length // window_size
        split_sizes = [window_size] * num_full_blocks
        remainder = seq_length % window_size
        if remainder > 0:
            split_sizes.append(remainder)
            
        # Process each window
        x_split = torch.split(x, split_sizes, dim=1)
        outputs = []
        
        for x_window in x_split:
            if x_window.size(1) == 0:
                continue
                
            # Self-attention within window
            attn_output, _ = self.attention(x_window, x_window, x_window)
            norm_output = self.norm(attn_output)
            outputs.append(norm_output)
            
        return torch.cat(outputs, dim=1) if outputs else x


class FeatureShift(nn.Module):
    """
    Feature Shift Module for temporal/spatial feature augmentation.
    
    Applies cyclic shift operation to features for data augmentation.
    """
    
    def __init__(self, input_dim: int, shift_size: int, embed_dim: int):
        """
        Initialize FeatureShift.
        
        Args:
            input_dim: Input feature dimension
            shift_size: Number of positions to shift
            embed_dim: Output embedding dimension
        """
        super().__init__()
        
        self.shift_size = shift_size // 2
        self.projection = nn.Linear(input_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with cyclic shift.
        
        Args:
            features: Input features tensor
            
        Returns:
            Shifted and projected features
        """
        shifted = torch.roll(features, shifts=-self.shift_size, dims=1)
        projected = self.projection(shifted)
        return self.norm(projected)


class MultiModalFusionNetwork(nn.Module):
    """
    Complete Multi-modal Fusion Network for text and image data.
    
    Integrates multiple processing channels and fusion strategies
    for robust multi-modal representation learning.
    """
    
    def __init__(self, input_dim_A: int, input_dim_B: int, embed_dim: int, 
                 num_heads: int, window_size: int):
        """
        Initialize MultiModalFusionNetwork.
        
        Args:
            input_dim_A: Input dimension for modality A (text)
            input_dim_B: Input dimension for modality B (image)
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            window_size: Window size for global processing
        """
        super().__init__()
        
        self.window_size = window_size
        
        # First channel modules
        self.first_channel_A = FirstChannelModule(input_dim_A, input_dim_B, embed_dim, num_heads)
        self.first_channel_B = FirstChannelModule(input_dim_B, input_dim_A, embed_dim, num_heads)
        self.local_enhancement = LocalEnhancement(embed_dim, embed_dim, num_heads)
        
        # Second channel modules
        self.second_channel_A = SecondChannelModule(embed_dim, input_dim_B, embed_dim, num_heads, embed_dim * 4)
        self.second_channel_B = SecondChannelModule(embed_dim, input_dim_A, embed_dim, num_heads, embed_dim * 4)
        
        # Fusion and global processing
        self.feature_fusion = nn.Linear(embed_dim, embed_dim)
        self.global_extractor = GlobalFeatureExtractor(embed_dim, window_size, num_heads, embed_dim)
        self.feature_shift = FeatureShift(embed_dim, window_size, embed_dim)
        self.final_projection = nn.Linear(2 * embed_dim, embed_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x_A: torch.Tensor, x_B: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for multi-modal fusion.
        
        Args:
            x_A: Modality A input tensor
            x_B: Modality B input tensor
            
        Returns:
            Classification logits
        """
        # First channel processing
        first_output_A = self.first_channel_A(x_A, x_B)
        enhanced_A = self.local_enhancement(first_output_A)
        second_output_A = self.second_channel_A(first_output_A, x_B)  # 修正：使用x_B而不是x_A
        
        # Feature combination and fusion
        combined_A = torch.add(enhanced_A, second_output_A)
        fused_A = self.feature_fusion(combined_A)
        
        # Global processing with shift augmentation
        global_A = self.global_extractor(fused_A, self.window_size)
        shifted_A = self.feature_shift(global_A)
        final_A = self.global_extractor(combined_A + shifted_A, self.window_size)
        
        # Second modality processing
        first_output_B = self.first_channel_B(x_B, x_A)
        enhanced_B = self.local_enhancement(first_output_B)
        second_output_B = self.second_channel_B(first_output_B, x_A)  # 修正：使用x_A而不是x_B
        
        combined_B = torch.add(enhanced_B, second_output_B)
        fused_B = self.feature_fusion(combined_B)
        
        global_B = self.global_extractor(fused_B, self.window_size)
        shifted_B = self.feature_shift(global_B)
        final_B = self.global_extractor(combined_B + shifted_B, self.window_size)
        
        # Multi-modal fusion
        fused_output = torch.add(final_A, final_B)
        final_fused = self.global_extractor(fused_output, self.window_size)
        
        shifted_final = self.feature_shift(final_fused)
        enhanced_final = fused_output + shifted_final
        
        # Classification
        pooled = enhanced_final.mean(dim=1)
        return self.classifier(pooled).squeeze(-1)

