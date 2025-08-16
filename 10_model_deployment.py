import torch
import yaml
from pathlib import Path
import sys
import shutil
from stable_baselines3 import PPO
from transformers import AutoProcessor

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from Scripts.FinLlava.core_architecture import create_trading_model

import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelDeployer:
    """
    Handles packaging and preparing the trained models for deployment.
    """
    def __init__(self):
        self.base_path = Path("/workspace/autonomous_trading_system")
        self.stage1_model_path = self.base_path / "data" / "models" / "stage1_finllava" / "final"
        self.stage2_model_path = self.base_path / "data" / "models" / "stage2_drl_agent" / "ppo_trading_agent.zip"
        self.deployment_dir = self.base_path / "deployment" / "finllava_trading_bot_v1"

        self.deployment_dir.mkdir(parents=True, exist_ok=True)

    def package_models(self):
        """Package all necessary model artifacts into the deployment directory."""
        logger.info(f"🚀 Starting model packaging process for deployment...")

        # 1. Copy Stage 2 DRL Agent
        logger.info(f"📦 Packaging Stage 2 DRL Agent from {self.stage2_model_path}")
        try:
            shutil.copy(self.stage2_model_path, self.deployment_dir / "ppo_trading_agent.zip")
        except FileNotFoundError:
            logger.error("❌ DRL agent model not found. Run Stage 2 training first.")
            return

        # 2. Copy the FinLLaVA processor from Stage 1
        # The DRL agent uses observations from the environment, but for a fully integrated
        # system, the FinLLaVA model and processor would be needed for feature extraction.
        logger.info(f"📦 Packaging FinLLaVA processor from {self.stage1_model_path}")
        try:
            processor_path = self.stage1_model_path
            if not processor_path.exists():
                raise FileNotFoundError

            # We only need the processor files, not the full model weights for deployment
            # if the DRL agent is the primary inference engine.
            shutil.copytree(processor_path, self.deployment_dir / "finllava_processor", dirs_exist_ok=True)

            # Clean up unnecessary large files from the copied processor dir
            for f in (self.deployment_dir / "finllava_processor").glob("*.bin"):
                f.unlink() # Remove large model weight files

        except FileNotFoundError:
            logger.error("❌ FinLLaVA model/processor not found. Run Stage 1 training first.")
            return

        # 3. Copy relevant configuration files
        logger.info("📦 Packaging configuration files...")
        configs_to_copy = ["drl_config.yaml", "safety_config.yaml", "market_config.yaml"]
        (self.deployment_dir / "config").mkdir(exist_ok=True)
        for config_file in configs_to_copy:
            try:
                shutil.copy(self.base_path / "config" / config_file, self.deployment_dir / "config" / config_file)
            except FileNotFoundError:
                logger.warning(f"⚠️ Config file {config_file} not found, skipping.")

        # 4. Create a deployment README
        self.create_deployment_readme()

        logger.info(f"✅ Deployment package created successfully at: {self.deployment_dir}")

    def create_deployment_readme(self):
        """Create a README file for the deployment package."""
        readme_content = f"""
# FinLLaVA Trading Bot - Deployment Package v1.0

This package contains the necessary artifacts to run the autonomous trading bot.

## Contents
- `ppo_trading_agent.zip`: The trained Stable Baselines 3 PPO agent for making trading decisions.
- `finllava_processor/`: The Hugging Face processor for the FinLLaVA model. Used for any text/image preprocessing if needed.
- `config/`: Directory containing configuration files for the DRL agent, safety monitoring, and market data.

## How to Use

1. **Load the DRL Agent:**
   ```python
   from stable_baselines3 import PPO
   agent = PPO.load("path/to/deployment/package/ppo_trading_agent.zip")
   ```

2. **Initialize the Environment:**
   You will need a live trading environment that provides observations in the same format as the training environment.

3. **Run Inference:**
   ```python
   observation = live_env.get_observation()
   action, _ = agent.predict(observation, deterministic=True)
   live_env.execute_trade(action)
   ```

4. **Safety Monitoring:**
   Ensure the `SafetyMonitor` is active and configured with `safety_config.yaml` to manage risk during live trading.

## Important Notes
- This is a simplified deployment package. A full production system would require a more robust setup (e.g., Docker container, REST API for the model).
- Always run in a paper trading mode first to validate performance before committing real capital.
"""
        with open(self.deployment_dir / "README.md", 'w') as f:
            f.write(readme_content)
        logger.info("✅ Deployment README created.")

if __name__ == "__main__":
    deployer = ModelDeployer()
    deployer.package_models()
    logger.info("🎉 FinLLaVA Trading Bot is ready for deployment!")
