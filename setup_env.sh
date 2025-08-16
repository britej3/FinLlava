#!/bin/bash
# 01_setup_environment.sh - RunPod Environment Setup for Autonomous Trading System

set -e
echo "🚀 Setting up Autonomous Trading System Environment..."

# Update system
apt-get update && apt-get upgrade -y

# Install system dependencies
apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    htop \
    screen \
    tmux \
    postgresql-client \
    redis-tools \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    python3-dev

# Install Conda if not present
if ! command -v conda &> /dev/null; then
    echo "Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
fi

# Create conda environment
echo "Creating trading_agent environment..."
conda create -n trading_agent python=3.10 -y
source activate trading_agent

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA..."
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install transformers and multimodal libraries
echo "Installing Transformers and Multimodal libraries..."
pip install transformers==4.35.0 
pip install accelerate 
pip install bitsandbytes
pip install huggingface_hub 
pip install datasets 
pip install peft
pip install flash-attn --no-build-isolation

# Install FinLLaVA specific dependencies
pip install llava
pip install deepspeed
pip install einops
pip install timm

# Install Reinforcement Learning
echo "Installing RL libraries..."
pip install ray[rllib]==2.8.0 
pip install gymnasium==0.29.1
pip install stable-baselines3==2.2.1 
pip install tensorboard

# Install Trading & Market Data libraries
echo "Installing trading libraries..."
pip install ccxt==4.1.0 
pip install backtrader 
pip install vectorbt 
pip install ta-lib
pip install zipline-reloaded 
pip install freqtrade 
pip install yfinance
pip install python-binance 
pip install alpha_vantage 
pip install websocket-client

# Install Data Processing & ML
echo "Installing data processing libraries..."
pip install pandas==2.1.3 
pip install numpy==1.24.3 
pip install polars
pip install scikit-learn==1.3.2 
pip install xgboost==2.0.1 
pip install lightgbm
pip install ta 
pip install tsfresh 
pip install featuretools 
pip install optuna 
pip install hyperopt

# Install Quantization & Memory Optimization
pip install auto-gptq 
pip install optimum
pip install onnx 
pip install onnxruntime-gpu

# Install Visualization & Monitoring
echo "Installing visualization libraries..."
pip install mplfinance 
pip install plotly 
pip install streamlit 
pip install dash
pip install prometheus_client 
pip install loguru 
pip install sentry-sdk 
pip install wandb

# Install Fine-tuning & Data Processing
pip install unstructured
pip install nlpaug 
pip install albumentations
pip install dvc[s3]

# Additional dependencies for autonomous system
pip install psutil
pip install schedule
pip install requests
pip install beautifulsoup4
pip install selenium
pip install praw  # Reddit API
pip install tweepy  # Twitter API

# Install Jupyter for development
pip install jupyter jupyterlab

echo "✅ Environment setup complete!"
echo "📝 Next: Run 02_create_project_structure.sh"