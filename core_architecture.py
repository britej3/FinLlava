# 04_core_trading_architecture.py - Core HRM/ZRIA/FinLLaVA Architecture
# Fully Autonomous Algorithmic Crypto High Leveraged Futures Scalping Bot

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Optional, Tuple
from transformers import (
    LlavaForConditionalGeneration, 
    LlavaProcessor, 
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
import logging
from pathlib import Path
import yaml

BOT_NAME = "Fully Autonomous Algorithmic Crypto High Leveraged Futures Scalping Bot"
VERSION = "v1.0.0"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProbabilisticFractalActivation(nn.Module):
    """P-FAF: Probabilistic Fractal Activation Function for ZRIA blocks"""
    
    def __init__(self, input_dim: int, num_fractals: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.num_fractals = num_fractals
        
        # Dynamic weight generation network
        self.weight_generator = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_fractals),
            nn.Softmax(dim=-1)
        )
        
        # Fractal transformation parameters
        self.fractal_params = nn.Parameter(torch.randn(num_fractals, 3))  # scale, phase, frequency
        
        # Market regime adaptation parameters
        self.regime_weights = nn.Parameter(torch.ones(num_fractals))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Enhanced fractal activation for high-frequency trading signals
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
        Returns:
            Enhanced tensor with market-adapted fractal activations
        """
        batch_size, seq_len, _ = x.shape
        
        # Stabilize input for high-leverage scenarios
        x_stable = torch.sigmoid(x)
        
        # Generate dynamic weights from global market context
        global_context = x.mean(dim=1, keepdim=True)  # (batch, 1, input_dim)
        fractal_weights = self.weight_generator(global_context)  # (batch, 1, num_fractals)
        
        # Apply market regime adaptation
        regime_factor = torch.softmax(self.regime_weights, dim=0)
        fractal_weights = fractal_weights * regime_factor.unsqueeze(0).unsqueeze(0)
        
        # Apply fractal functions optimized for trading patterns
        fractal_responses = []
        for i in range(self.num_fractals):
            scale, phase, freq = self.fractal_params[i]
            
            # Different fractal functions for different market patterns
            if i % 4 == 0:  # Trend detection
                fractal = torch.sin(freq * x_stable + phase) * torch.sigmoid(scale)
            elif i % 4 == 1:  # Mean reversion 
                fractal = torch.cos(freq * x_stable + phase) * torch.sigmoid(scale)
            elif i % 4 == 2:  # Volatility expansion
                fractal = torch.tanh(freq * x_stable + phase) * torch.sigmoid(scale)
            else:  # Support/resistance
                fractal = torch.sigmoid(freq * x_stable + phase) * torch.sigmoid(scale)
                
            fractal_responses.append(fractal)
        
        # Weighted combination with market context
        fractal_stack = torch.stack(fractal_responses, dim=-1)  # (batch, seq, dim, num_fractals)
        weighted_fractals = torch.sum(fractal_stack * fractal_weights.unsqueeze(-2), dim=-1)
        
        # Residual connection with adaptive gating for stability
        gate = torch.sigmoid(self.weight_generator[0](x_stable))
        output = x + weighted_fractals * gate
        
        return output

class FractalAttentionalResonance(nn.Module):
    """FAR: Fractal Attentional Resonance for multi-timeframe market analysis"""
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        # Standard attention components
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model) 
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        # Fractal bias generation for market regime adaptation
        self.context_processor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_heads),
            nn.Tanh()
        )
        
        # Multi-timeframe resonance parameters
        self.timeframe_embeddings = nn.Embedding(5, d_model)  # 5 timeframes
        self.resonance_gate = nn.Parameter(torch.ones(num_heads))
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, timeframe_ids: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Multi-timeframe fractal attention for scalping signals
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            timeframe_ids: Timeframe identifiers (batch, seq_len) 
            mask: Attention mask (batch, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        residual = x
        
        # Add timeframe embeddings if provided
        if timeframe_ids is not None:
            timeframe_emb = self.timeframe_embeddings(timeframe_ids)
            x = x + timeframe_emb
        
        # Generate Q, K, V
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # (batch, heads, seq_len, head_dim)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Generate fractal bias from global market context
        global_context = x.mean(dim=1)  # (batch, d_model)
        fractal_bias = self.context_processor(global_context)  # (batch, num_heads)
        
        # Apply resonance gating for market regime adaptation
        resonance_factor = torch.sigmoid(self.resonance_gate).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        fractal_bias = fractal_bias.unsqueeze(-1).unsqueeze(-1) * resonance_factor  # (batch, heads, 1, 1)
        
        # Add fractal bias to attention scores
        scores = scores + fractal_bias
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # Softmax and apply to values
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        out = torch.matmul(attention_weights, V)
        
        # Transpose back and concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection and residual connection
        out = self.out_linear(out)
        out = self.layer_norm(out + residual)
        
        return out

class HierarchicalReasoningModule(nn.Module):
    """HRM: Multi-timescale trading brain for scalping decisions"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Multi-timescale processing dimensions
        self.timescale_dims = {
            'micro': config['timescales']['micro']['dim'],    # 1m scalping
            'short': config['timescales']['short']['dim'],    # 5m momentum  
            'medium': config['timescales']['medium']['dim'],  # 15m trend
            'long': config['timescales']['long']['dim']       # 1h context
        }
        
        # Individual timescale processors
        self.timescale_processors = nn.ModuleDict()
        for scale, dim in self.timescale_dims.items():
            self.timescale_processors[scale] = nn.Sequential(
                nn.Linear(config['input_dim'], dim),
                nn.LayerNorm(dim),
                ProbabilisticFractalActivation(dim),
                FractalAttentionalResonance(dim, num_heads=8),
                nn.Dropout(0.1),
                nn.Linear(dim, dim),
                nn.GELU()
            )
        
        # Cross-timescale fusion
        total_dim = sum(self.timescale_dims.values())
        self.fusion_projector = nn.Linear(total_dim, config['fusion_dim'])
        self.fusion_attention = FractalAttentionalResonance(config['fusion_dim'], num_heads=12)
        
        # Market regime detection
        self.regime_classifier = nn.Sequential(
            nn.Linear(config['fusion_dim'], 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 4)  # trending_up, trending_down, ranging, volatile
        )
        
        # Decision heads for different action types
        self.position_head = nn.Sequential(
            ProbabilisticFractalActivation(config['fusion_dim']),
            nn.Linear(config['fusion_dim'], 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3)  # long, short, neutral
        )
        
        self.sizing_head = nn.Sequential(
            nn.Linear(config['fusion_dim'], 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # position size multiplier
            nn.Sigmoid()
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(config['fusion_dim'], 32),
            nn.ReLU(), 
            nn.Linear(32, 1),  # confidence score
            nn.Sigmoid()
        )
        
        logger.info(f"✅ HRM initialized for {BOT_NAME}")
        
    def forward(self, market_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Process multi-timeframe market data for scalping decisions
        
        Args:
            market_data: Dict containing different timeframe data
                - 'micro': (batch, seq_len, features) - 1min data
                - 'short': (batch, seq_len, features) - 5min data  
                - 'medium': (batch, seq_len, features) - 15min data
                - 'long': (batch, seq_len, features) - 1h data
        
        Returns:
            Dict with trading decisions and analysis
        """
        # Process each timescale
        timescale_outputs = {}
        processed_features = []
        
        for timescale, processor in self.timescale_processors.items():
            if timescale in market_data:
                # Process timescale-specific data
                processed = processor(market_data[timescale])
                timescale_outputs[timescale] = processed
                
                # Pool over sequence dimension for fusion
                pooled = processed.mean(dim=1)  # (batch, dim)
                processed_features.append(pooled)
        
        if not processed_features:
            raise ValueError("No valid market data provided")
        
        # Concatenate and fuse different timescales
        combined = torch.cat(processed_features, dim=-1)  # (batch, total_dim)
        fused = self.fusion_projector(combined)  # (batch, fusion_dim)
        
        # Add sequence dimension for attention
        fused_seq = fused.unsqueeze(1)  # (batch, 1, fusion_dim)
        
        # Apply cross-timescale attention
        enhanced = self.fusion_attention(fused_seq)  # (batch, 1, fusion_dim)
        enhanced = enhanced.squeeze(1)  # (batch, fusion_dim)
        
        # Market regime detection
        regime_logits = self.regime_classifier(enhanced)
        
        # Generate trading decisions
        position_logits = self.position_head(enhanced)
        position_size = self.sizing_head(enhanced)
        confidence = self.confidence_head(enhanced)
        
        return {
            'position_logits': position_logits,  # (batch, 3) - long/short/neutral
            'position_size': position_size,      # (batch, 1) - size multiplier
            'confidence': confidence,            # (batch, 1) - confidence score
            'regime_logits': regime_logits,      # (batch, 4) - market regime
            'timescale_features': timescale_outputs,
            'fused_representation': enhanced
        }

class FinLLaVATradingModel(nn.Module):
    """Complete FinLLaVA-based trading model with HRM/ZRIA integration"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Load base FinLLaVA model
        logger.info(f"Loading FinLLaVA base model for {BOT_NAME}...")
        
        # Configure quantization for memory efficiency
        if config.get('quantization', {}).get('load_in_4bit', False):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
            )
        else:
            quantization_config = None
            
        # Load FinLLaVA model
        self.llava_model = LlavaForConditionalGeneration.from_pretrained(
            config['model']['base_model'],
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Initialize processor
        self.processor = LlavaProcessor.from_pretrained(config['model']['base_model'])
        
        # HRM configuration
        hrm_config = {
            'input_dim': self.llava_model.config.hidden_size,
            'timescales': config['model'].get('timescales', {
                'micro': {'dim': 256},
                'short': {'dim': 128}, 
                'medium': {'dim': 64},
                'long': {'dim': 32}
            }),
            'fusion_dim': config['model'].get('fusion_dim', 256)
        }
        
        # Initialize HRM
        self.hrm = HierarchicalReasoningModule(hrm_config)
        
        # LoRA configuration for efficient fine-tuning
        if config.get('use_lora', True):
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.llava_model = get_peft_model(self.llava_model, lora_config)
            logger.info("✅ LoRA configuration applied")
        
        # Trading-specific adapter layers
        self.trading_adapter = nn.Sequential(
            nn.Linear(self.llava_model.config.hidden_size, hrm_config['fusion_dim']),
            nn.LayerNorm(hrm_config['fusion_dim']),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        logger.info(f"✅ {BOT_NAME} model initialized successfully")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                pixel_values: Optional[torch.Tensor] = None, 
                market_data: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass combining FinLLaVA multimodal processing with HRM trading logic
        
        Args:
            input_ids: Text token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            pixel_values: Chart images (batch, channels, height, width)
            market_data: Multi-timeframe market data for HRM
        """
        # Process through FinLLaVA
        llava_outputs = self.llava_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            output_hidden_states=True
        )
        
        # Extract hidden states for trading analysis
        hidden_states = llava_outputs.hidden_states[-1]  # Last layer
        
        # Pool hidden states for trading decision
        pooled_states = hidden_states.mean(dim=1)  # (batch, hidden_size)
        
        # Transform through trading adapter
        trading_features = self.trading_adapter(pooled_states)  # (batch, fusion_dim)
        
        # If market data provided, use HRM for enhanced decision making
        if market_data is not None:
            # Add trading features to each timescale
            enhanced_market_data = {}
            for timescale, data in market_data.items():
                # Broadcast trading features to match sequence length
                batch_size, seq_len, feature_dim = data.shape
                features_expanded = trading_features.unsqueeze(1).expand(-1, seq_len, -1)
                
                # Concatenate with market data
                enhanced_data = torch.cat([data, features_expanded], dim=-1)
                enhanced_market_data[timescale] = enhanced_data
            
            # Process through HRM
            hrm_outputs = self.hrm(enhanced_market_data)
            
            return {
                'llava_logits': llava_outputs.logits,
                'llava_hidden_states': hidden_states,
                'trading_features': trading_features,
                **hrm_outputs
            }
        
        else:
            # Simple trading decision without HRM
            simple_decision = nn.Linear(trading_features.size(-1), 3).to(trading_features.device)
            position_logits = simple_decision(trading_features)
            
            return {
                'llava_logits': llava_outputs.logits,
                'llava_hidden_states': hidden_states,
                'trading_features': trading_features,
                'position_logits': position_logits
            }

def load_config(stage: str = "stage1") -> Dict:
    """Load configuration for specified stage"""
    config_path = Path(f"/workspace/autonomous_trading_system/config/{stage}_config.yaml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"✅ Loaded {stage} configuration")
    return config

def create_trading_model(stage: str = "stage1") -> FinLLaVATradingModel:
    """Create and initialize trading model"""
    logger.info(f"🔄 Creating {BOT_NAME} model for {stage}...")
    
    config = load_config(stage)
    model = FinLLaVATradingModel(config)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"✅ Model created successfully!")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Memory efficiency: {(1 - trainable_params/total_params)*100:.1f}% frozen")
    
    return model

if __name__ == "__main__":
    # Test model creation
    logger.info(f"🧪 Testing {BOT_NAME} architecture...")
    
    try:
        model = create_trading_model("stage1")
        logger.info("✅ Architecture test passed!")
        logger.info("📝 Next: Run 05_stage1_training.py")
        
    except Exception as e:
        logger.error(f"❌ Architecture test failed: {e}")
        raise