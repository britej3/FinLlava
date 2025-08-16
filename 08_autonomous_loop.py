import schedule
import time
import logging
from pathlib import Path
import sys
import pandas as pd
from stable_baselines3 import PPO

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from Scripts.FinLlava.drl_environment import TradingEnv
from Scripts.FinLlava.core_architecture import create_trading_model

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AutonomousAgent:
    """
    The main autonomous agent that runs the trading and learning loop.
    """
    def __init__(self, paper_trading: bool = True):
        self.base_path = Path("/workspace/autonomous_trading_system")
        self.paper_trading = paper_trading

        # Load the trained DRL agent
        self.drl_agent = self.load_drl_agent()

        # In a real scenario, you would have a live data feed and a live trading environment
        # For this example, we will simulate it with the test environment.
        self.market_data = self.load_market_data()
        self.live_env = TradingEnv(market_data=self.market_data, config={})

        logger.info(f"🤖 Autonomous Agent initialized in {'paper' if paper_trading else 'live'} trading mode.")

    def load_drl_agent(self):
        """Load the trained Stage 2 DRL agent"""
        model_path = self.base_path / "data" / "models" / "stage2_drl_agent" / "ppo_trading_agent.zip"
        try:
            agent = PPO.load(model_path)
            logger.info(f"✅ DRL agent loaded from {model_path}")
            return agent
        except FileNotFoundError:
            logger.error(f"❌ DRL agent not found at {model_path}. Run Stage 2 training first.")
            sys.exit(1)

    def load_market_data(self) -> pd.DataFrame:
        """Load market data for simulation"""
        # In a real system, this would be a live data feed.
        market_data_path = self.base_path / "data" / "raw" / "market_data" / "BTCUSDT_1h.csv"
        try:
            df = pd.read_csv(market_data_path, index_col='timestamp', parse_dates=True)
            df.fillna(method='bfill', inplace=True)
            return df
        except FileNotFoundError:
            logger.error(f"❌ Market data not found at {market_data_path}. Run data collection first.")
            sys.exit(1)

    def trading_cycle(self):
        """Execute one cycle of trading logic"""
        logger.info("🔄 Running trading cycle...")

        # Get current observation from the environment
        obs, _ = self.live_env.reset() # In a real scenario, this would be `self.live_env.get_observation()`

        # Predict action using the DRL agent
        action, _states = self.drl_agent.predict(obs, deterministic=True)

        # Execute the action in the environment
        obs, reward, terminated, truncated, info = self.live_env.step(action)

        logger.info(f"📈 Trade executed. Action: {action}, Reward: {reward:.5f}, Equity: ${info['equity']:.2f}")

        if terminated:
            logger.warning("Episode finished. Resetting environment.")
            self.live_env.reset()

    def retraining_cycle(self):
        """
        Execute a cycle of retraining the model.
        This is a simplified placeholder for the MLE-STAR loop.
        """
        logger.info("🔄 Starting retraining cycle...")

        # 1. Collect new data (in a real system, this would be from live trading)
        # For simulation, we can just use the existing data.
        new_data = self.load_market_data()

        # 2. Re-train the DRL agent
        # For simplicity, we just call `learn` again on the existing agent.
        # A more robust implementation would handle data versioning, etc.
        logger.info("🏋️‍♂️ Retraining DRL agent with new data...")
        self.drl_agent.set_env(DummyVecEnv([lambda: TradingEnv(market_data=new_data, config={})]))
        self.drl_agent.learn(total_timesteps=10000, reset_num_timesteps=False)

        # 3. Save the updated agent
        model_path = self.base_path / "data" / "models" / "stage2_drl_agent" / "ppo_trading_agent_retrained.zip"
        self.drl_agent.save(model_path)

        logger.info(f"✅ Retraining complete. Updated agent saved to {model_path}")

    def run(self):
        """Run the main autonomous loop"""
        logger.info("🚀 Starting autonomous trading loop...")

        # Schedule the trading cycle to run every minute
        schedule.every(1).minutes.do(self.trading_cycle)

        # Schedule the retraining cycle to run every day
        schedule.every(1).days.at("01:00").do(self.retraining_cycle)

        while True:
            schedule.run_pending()
            time.sleep(1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Autonomous Trading Agent Loop")
    parser.add_argument("--live", action="store_true", help="Run in live trading mode (use with caution!)")
    args = parser.parse_args()

    agent = AutonomousAgent(paper_trading=not args.live)
    agent.run()
    logger.info("📝 Next: Run 09_safety_monitoring.py")
