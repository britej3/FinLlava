import torch
import yaml
from pathlib import Path
import sys
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from Scripts.FinLlava.core_architecture import create_trading_model
from Scripts.FinLlava.drl_environment import TradingEnv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BOT_NAME = "Fully Autonomous Algorithmic Crypto High Leveraged Futures Scalping Bot"
VERSION = "v1.0.0"

class Stage2Trainer:
    def __init__(self):
        self.base_path = Path("/workspace/autonomous_trading_system")
        self.config_path = self.base_path / "config"
        self.stage2_config = self.load_yaml("stage2_config.yaml")
        self.drl_config = self.load_yaml("drl_config.yaml")
        self.model_output_dir = self.base_path / "data" / "models" / "stage2_drl_agent"
        self.model_output_dir.mkdir(parents=True, exist_ok=True)

    def load_yaml(self, file_name: str) -> dict:
        """Load a YAML configuration file"""
        try:
            with open(self.config_path / file_name, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"✅ Configuration loaded from {file_name}")
            return config
        except Exception as e:
            logger.error(f"❌ Error loading {file_name}: {e}")
            sys.exit(1)

    def load_market_data(self) -> pd.DataFrame:
        """Load market data for DRL training"""
        # For simplicity, we'll use one of the collected market data files.
        # A more robust implementation would combine or chain multiple datasets.
        market_data_path = self.base_path / "data" / "raw" / "market_data" / "BTCUSDT_1h.csv"
        try:
            df = pd.read_csv(market_data_path, index_col='timestamp', parse_dates=True)
            # Fill NaNs from indicators
            df.fillna(method='bfill', inplace=True)
            logger.info(f"✅ Market data loaded from {market_data_path}")
            return df
        except FileNotFoundError:
            logger.error(f"❌ Market data not found at {market_data_path}. Run data collection first.")
            sys.exit(1)

    def train(self):
        """Run the Stage 2 DRL training pipeline"""
        logger.info(f"🚀 Starting Stage 2 DRL Training for {BOT_NAME}...")

        # 1. Load market data
        market_data = self.load_market_data()

        # 2. Create the trading environment
        # The environment needs to be wrapped for stable-baselines3
        env = DummyVecEnv([lambda: TradingEnv(market_data=market_data, config=self.drl_config)])

        # 3. Initialize the DRL agent (PPO)
        # Note: A custom policy would be needed to integrate the FinLLaVA model.
        # For this script, we use a standard Multi-Layer Perceptron (MLP) policy
        # that takes the environment's observation vector directly.

        ppo_params = self.drl_config.get('training', {})

        agent = PPO(
            "MlpPolicy",
            env,
            learning_rate=float(ppo_params.get('learning_rate', 3e-4)),
            n_steps=ppo_params.get('n_steps', 2048),
            batch_size=ppo_params.get('batch_size', 64),
            n_epochs=ppo_params.get('n_epochs', 10),
            gamma=ppo_params.get('gamma', 0.99),
            gae_lambda=ppo_params.get('gae_lambda', 0.95),
            clip_range=ppo_params.get('clip_range', 0.2),
            ent_coef=ppo_params.get('entropy_coef', 0.01),
            vf_coef=ppo_params.get('value_function_coef', 0.5),
            verbose=1,
            tensorboard_log=str(self.base_path / "logs" / "stage2_drl_training"),
        )

        # 4. Train the agent
        logger.info("🏋️‍♂️ Starting DRL agent training...")
        total_timesteps = int(ppo_params.get('total_timesteps', 1_000_000))

        try:
            agent.learn(total_timesteps=total_timesteps, progress_bar=True)
            logger.info("✅ DRL training completed successfully!")
        except Exception as e:
            logger.error(f"❌ DRL training failed: {e}")
            raise

        # 5. Save the trained agent
        final_model_path = self.model_output_dir / "ppo_trading_agent.zip"
        agent.save(final_model_path)

        logger.info(f"💾 DRL agent saved to {final_model_path}")
        logger.info("✅ Stage 2 training finished.")

if __name__ == "__main__":
    trainer = Stage2Trainer()
    trainer.train()
    logger.info("📝 Next: Run 08_autonomous_loop.py")
