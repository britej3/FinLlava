# 02_create_project_structure.py - Create Project Directory Structure

import os
import json
import yaml
from pathlib import Path

def create_directory_structure():
    """Create complete project directory structure"""
    
    base_path = Path("/workspace/autonomous_trading_system")
    
    directories = [
        "agents",
        "src", 
        "data/raw/market_data",
        "data/raw/alternative_data",
        "data/raw/educational_content",
        "data/processed/instruction_pairs",
        "data/processed/multimodal_examples", 
        "data/processed/features",
        "data/models",
        "data/checkpoints",
        "data/finllava_data",
        "config",
        "notebooks",
        "backtests",
        "logs",
        "scripts",
        "monitoring",
        "tests",
        "deployment",
        "emergency_protocols"
    ]
    
    # Create directories
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {dir_path}")
        
        # Create __init__.py for Python packages
        if directory in ["agents", "src"]:
            (dir_path / "__init__.py").touch()

def create_config_files():
    """Create configuration files"""
    
    config_path = Path("/workspace/autonomous_trading_system/config")
    
    # Stage 1 Configuration (Development)
    stage1_config = {
        "model": {
            "base_model": "liuhaotian/llava-v1.5-7b",  # Using LLaVA as base
            "vision_model": "openai/clip-vit-large-patch14-336",
            "parameters": "7B",
            "quantization": {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "float16",
                "bnb_4bit_quant_type": "nf4"
            },
            "vram_requirement": "12GB",
            "inference_speed": "200ms"
        },
        "training": {
            "batch_size": 4,
            "gradient_accumulation_steps": 8,
            "learning_rate": 2e-5,
            "num_epochs": 3,
            "warmup_steps": 100,
            "max_seq_length": 2048
        },
        "data": {
            "dataset_size": 15000,
            "validation_split": 0.2,
            "multimodal_ratio": 0.4  # 40% multimodal, 60% text-only
        },
        "hardware": {
            "gpu_memory": "24GB",
            "cpu_cores": 16,
            "ram": "64GB"
        }
    }
    
    # Stage 2 Configuration (Production)
    stage2_config = {
        "model": {
            "base_model": "liuhaotian/llava-v1.5-13b",  # Larger model for production
            "vision_model": "openai/clip-vit-large-patch14-336",
            "parameters": "13B", 
            "quantization": {
                "load_in_8bit": True,
                "use_gradient_checkpointing": True
            },
            "vram_requirement": "40GB",
            "inference_speed": "400ms"
        },
        "training": {
            "batch_size": 2,
            "gradient_accumulation_steps": 16,
            "learning_rate": 1e-5,
            "num_epochs": 5,
            "warmup_steps": 200,
            "max_seq_length": 4096
        },
        "data": {
            "dataset_size": 50000,
            "validation_split": 0.15,
            "multimodal_ratio": 0.6  # More multimodal data for production
        },
        "hardware": {
            "gpu_memory": "80GB",
            "cpu_cores": 32,
            "ram": "128GB"
        }
    }
    
    # HRM Configuration
    hrm_config = {
        "architecture": {
            "input_dim": 512,
            "timescales": {
                "micro": {"dim": 256, "timeframe": "1m"},
                "short": {"dim": 128, "timeframe": "15m"},
                "medium": {"dim": 64, "timeframe": "4h"},
                "long": {"dim": 32, "timeframe": "1d"}
            },
            "fusion_dim": 128,
            "num_actions": 3,  # buy, sell, hold
            "attention_heads": 8,
            "num_layers": 6
        },
        "zria_blocks": {
            "num_fractals": 5,
            "fractal_functions": ["sin", "cos", "tanh"],
            "resonance_layers": 3
        }
    }
    
    # DRL Configuration
    drl_config = {
        "algorithm": "PPO",
        "environment": "TradingEnv-v1",
        "training": {
            "total_timesteps": 1000000,
            "learning_rate": 3e-4,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "entropy_coef": 0.01,
            "value_function_coef": 0.5
        },
        "reward_function": {
            "sharpe_weight": 1.0,
            "drawdown_penalty": -2.0,
            "transaction_cost": -0.001,
            "consistency_bonus": 0.1
        }
    }
    
    # Market Data Configuration  
    market_config = {
        "exchanges": {
            "binance": {
                "api_key": "${BINANCE_API_KEY}",
                "secret_key": "${BINANCE_SECRET_KEY}",
                "sandbox": True,
                "rate_limit": 1200
            },
            "coinbase": {
                "api_key": "${COINBASE_API_KEY}",
                "secret_key": "${COINBASE_SECRET_KEY}",
                "sandbox": True
            }
        },
        "symbols": ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT"],
        "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
        "data_retention": "90d",
        "alternative_data": {
            "twitter": {
                "api_key": "${TWITTER_API_KEY}",
                "bearer_token": "${TWITTER_BEARER_TOKEN}"
            },
            "reddit": {
                "client_id": "${REDDIT_CLIENT_ID}",
                "client_secret": "${REDDIT_CLIENT_SECRET}"
            },
            "news": {
                "alpha_vantage_key": "${ALPHA_VANTAGE_KEY}"
            }
        }
    }
    
    # Safety Configuration
    safety_config = {
        "risk_limits": {
            "max_position_size": 0.3,  # 30% of capital
            "max_daily_loss": 0.02,    # 2% daily loss limit
            "max_drawdown": 0.15,      # 15% max drawdown
            "min_sharpe_ratio": 0.5
        },
        "emergency_protocols": {
            "auto_shutdown": True,
            "rollback_threshold": -0.05,  # 5% loss triggers rollback
            "alert_channels": ["email", "telegram"],
            "backup_strategy": "cash"
        },
        "monitoring": {
            "performance_check_interval": 300,  # 5 minutes
            "anomaly_detection": True,
            "health_check_endpoints": ["/health", "/metrics"]
        }
    }
    
    # Save configurations
    configs = {
        "stage1_config.yaml": stage1_config,
        "stage2_config.yaml": stage2_config, 
        "hrm_config.yaml": hrm_config,
        "drl_config.yaml": drl_config,
        "market_config.yaml": market_config,
        "safety_config.yaml": safety_config
    }
    
    for filename, config in configs.items():
        with open(config_path / filename, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"✅ Created config: {filename}")

def create_environment_file():
    """Create .env template"""
    
    env_content = """# Autonomous Trading System Environment Variables

# Exchange API Keys (Sandbox/Testnet)
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET_KEY=your_binance_secret_key_here

COINBASE_API_KEY=your_coinbase_api_key_here  
COINBASE_SECRET_KEY=your_coinbase_secret_key_here

# Alternative Data APIs
TWITTER_API_KEY=your_twitter_api_key_here
TWITTER_BEARER_TOKEN=your_twitter_bearer_token_here

REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here

ALPHA_VANTAGE_KEY=your_alpha_vantage_key_here

# Hugging Face
HF_TOKEN=your_huggingface_token_here

# Weights & Biases (Optional)
WANDB_API_KEY=your_wandb_key_here

# Database (Optional - for production)
DATABASE_URL=postgresql://user:password@localhost/trading_db

# Monitoring (Optional)
SENTRY_DSN=your_sentry_dsn_here

# Model Storage
MODEL_UPLOAD_PATH=/workspace/autonomous_trading_system/data/models/
CHECKPOINT_PATH=/workspace/autonomous_trading_system/data/checkpoints/
"""
    
    env_path = Path("/workspace/autonomous_trading_system/.env")
    with open(env_path, 'w') as f:
        f.write(env_content)
    print("✅ Created .env template")

def create_requirements_file():
    """Create requirements.txt for reproducibility"""
    
    requirements = """torch==2.1.0
torchvision==0.16.0  
torchaudio==2.1.0
transformers==4.35.0
accelerate==0.24.1
bitsandbytes==0.41.3
huggingface_hub==0.17.3
datasets==2.14.6
peft==0.6.2
flash-attn==2.3.3
llava
deepspeed==0.12.4
einops==0.7.0
timm==0.9.10

ray[rllib]==2.8.0
gymnasium==0.29.1
stable-baselines3==2.2.1
tensorboard==2.15.1

ccxt==4.1.0
backtrader==1.9.78.123
vectorbt==0.25.2
ta-lib==0.4.28
zipline-reloaded==3.0.3
freqtrade==2023.11
yfinance==0.2.22
python-binance==1.0.19
alpha_vantage==2.3.1
websocket-client==1.6.4

pandas==2.1.3
numpy==1.24.3
polars==0.19.19
scikit-learn==1.3.2
xgboost==2.0.1
lightgbm==4.1.0
ta==0.10.2
tsfresh==0.20.1
featuretools==1.28.0
optuna==3.4.0
hyperopt==0.2.7

auto-gptq==0.5.1
optimum==1.14.1
onnx==1.15.0
onnxruntime-gpu==1.16.3

mplfinance==0.12.9b7
plotly==5.17.0
streamlit==1.28.2
dash==2.14.2
prometheus_client==0.19.0
loguru==0.7.2
sentry-sdk==1.38.0
wandb==0.16.0

unstructured==0.11.2
nlpaug==1.1.11
albumentations==1.3.1
dvc[s3]==3.26.0

psutil==5.9.6
schedule==1.2.0
requests==2.31.0
beautifulsoup4==4.12.2
selenium==4.15.2
praw==7.7.1
tweepy==4.14.0

jupyter==1.0.0
jupyterlab==4.0.9
"""
    
    req_path = Path("/workspace/autonomous_trading_system/requirements.txt")
    with open(req_path, 'w') as f:
        f.write(requirements)
    print("✅ Created requirements.txt")

def create_readme():
    """Create comprehensive README"""
    
    readme_content = """# Autonomous Self-Improving Multimodal Trading Agent

## Overview
A fully autonomous, self-learning cryptocurrency futures trading system that uses:
- **FinLLaVA**: Multimodal LLM for processing price charts, news, and market data
- **HRM (Hierarchical Reasoning Module)**: Multi-timescale market analysis
- **ZRIA Blocks**: Probabilistic fractal activation for enhanced signal processing  
- **Deep RL**: Continuous improvement through market interaction
- **MLE-STAR**: Autonomous code generation and self-healing capabilities

## Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    MLE-STAR (AI Engineer)                   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
│  │ Coder Agent │ │Feature Eng. │ │ Backtest & Ablation    │ │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│                   FinLLaVA Trading Brain                    │
│  ┌──────────────┐ ┌──────────────┐ ┌─────────────────────┐  │
│  │     HRM      │ │ ZRIA Blocks  │ │   Multimodal       │  │
│  │ Multi-scale  │ │ Fractal      │ │   Fusion Layer     │  │
│  │ Reasoning    │ │ Activation   │ │                    │  │
│  └──────────────┘ └──────────────┘ └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Deep RL Engine                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
│  │ PPO Agent   │ │ Custom Env  │ │ Intelligent Rewards    │ │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Environment Setup
```bash
chmod +x 01_setup_environment.sh
./01_setup_environment.sh
```

### 2. Project Structure
```bash
python 02_create_project_structure.py
```

### 3. Data Collection
```bash
python 03_data_collection.py
```

### 4. Model Training (Stage 1)
```bash
python 04_stage1_training.py
```

### 5. Model Training (Stage 2) 
```bash
python 05_stage2_production.py
```

### 6. Autonomous Loop
```bash
python 06_autonomous_loop.py
```

## Features

### ✅ Multimodal Analysis
- Price chart pattern recognition
- News sentiment analysis  
- Social media sentiment tracking
- Order book visualization processing

### ✅ Self-Improvement
- Genetic algorithm feature evolution
- Autonomous strategy generation
- Performance-based model selection
- Continuous hyperparameter optimization

### ✅ Risk Management
- Dynamic position sizing
- Multi-layer stop losses
- Drawdown protection
- Emergency shutdown protocols

### ✅ Self-Healing
- Automated error detection
- Strategy rollback capabilities
- Performance anomaly detection
- Autonomous bug fixing

## Configuration Files
- `stage1_config.yaml` - Development model settings
- `stage2_config.yaml` - Production model settings  
- `hrm_config.yaml` - Hierarchical reasoning configuration
- `drl_config.yaml` - Deep RL training parameters
- `market_config.yaml` - Exchange and data source settings
- `safety_config.yaml` - Risk management and safety limits

## Usage

### Stage 1 Development (Fast Prototyping)
```bash
python main_stage1_training.py --config config/stage1_config.yaml
```

### Stage 2 Production (Full Performance)
```bash
python main_stage2_upgrade.py --config config/stage2_config.yaml
```

### Autonomous Mode
```bash
python main_autonomous_loop.py --live-trading --paper-mode
```

## Monitoring & Safety
- Real-time performance dashboards
- Automated risk monitoring
- Emergency shutdown triggers  
- Performance anomaly detection
- Strategy rollback capabilities

## Model Download
After training completion, models are automatically saved to:
- `/workspace/autonomous_trading_system/data/models/finllava_trading_final/`
- Hugging Face Hub integration for easy sharing

## Support
For issues or questions, check the logs in:
- `/workspace/autonomous_trading_system/logs/`

## License
MIT License - Use at your own risk in financial markets.

## Disclaimer
This is experimental software for educational purposes. Trading cryptocurrencies 
involves substantial risk of loss. Always test thoroughly with paper trading first.
"""
    
    readme_path = Path("/workspace/autonomous_trading_system/README.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print("✅ Created README.md")

def create_gitignore():
    """Create .gitignore file"""
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Environment Variables
.env
.venv
env/
venv/

# Models and Data
data/models/
data/checkpoints/
*.bin
*.safetensors
*.pt
*.pth
*.ckpt

# Logs
logs/
*.log
tensorboard/

# Jupyter Notebooks
.ipynb_checkpoints

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Trading
backtests/*.pkl
backtests/*.csv
strategies/

# Temporary files
tmp/
temp/
"""
    
    gitignore_path = Path("/workspace/autonomous_trading_system/.gitignore")
    with open(gitignore_path, 'w') as f:
        f.write(gitignore_content)
    print("✅ Created .gitignore")

if __name__ == "__main__":
    print("🏗️ Creating Autonomous Trading System Project Structure...")
    
    create_directory_structure()
    create_config_files() 
    create_environment_file()
    create_requirements_file()
    create_readme()
    create_gitignore()
    
    print("\n✅ Project structure created successfully!")
    print("📁 Base path: /workspace/autonomous_trading_system")
    print("📝 Next: Run 03_data_collection.py")