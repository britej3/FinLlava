import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingEnv(gym.Env):
    """
    Custom Gymnasium Environment for Deep Reinforcement Learning-based Trading.

    This environment simulates a cryptocurrency futures trading account, allowing
    a DRL agent to learn optimal trading strategies.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self,
                 market_data: pd.DataFrame,
                 config: Dict[str, Any],
                 initial_balance: float = 10000.0):
        super().__init__()

        self.market_data = market_data
        self.config = config
        self.initial_balance = initial_balance

        # Trading parameters
        self.leverage = self.config.get('leverage', 10)
        self.transaction_cost = self.config.get('transaction_cost', 0.0004) # 0.04%

        # State tracking
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.current_step = 0
        self.position = 0  # 1 for long, -1 for short, 0 for neutral
        self.entry_price = 0

        # Action space: 0: Hold, 1: Long, 2: Short
        self.action_space = spaces.Discrete(3)

        # Observation space: Market features (e.g., OHLCV + indicators)
        # The shape should match the number of features in the market_data
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(self.market_data.columns),),
            dtype=np.float32
        )

        logger.info("✅ DRL Trading Environment initialized")

    def _get_observation(self) -> np.ndarray:
        """Get the current market observation"""
        return self.market_data.iloc[self.current_step].values.astype(np.float32)

    def _get_info(self) -> Dict[str, Any]:
        """Get auxiliary information about the current state"""
        return {
            "equity": self.equity,
            "balance": self.balance,
            "position": self.position,
            "entry_price": self.entry_price,
            "current_step": self.current_step
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to the initial state"""
        super().reset(seed=seed)

        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.current_step = 0
        self.position = 0
        self.entry_price = 0

        logger.info(f"🔄 Environment reset. Initial balance: ${self.initial_balance:.2f}")

        return self._get_observation(), self._get_info()

    def _calculate_reward(self, previous_equity: float) -> float:
        """Calculate the reward for the current step"""
        reward_config = self.config.get('reward_function', {})

        # Profit-based reward
        profit_reward = (self.equity - previous_equity) / previous_equity

        # Sharpe ratio (simplified for single step)
        sharpe_reward = profit_reward * reward_config.get('sharpe_weight', 1.0)

        # Penalty for holding a position (encourages quick scalping)
        holding_penalty = -0.0001 if self.position != 0 else 0

        # Penalty for excessive drawdown
        drawdown = (self.equity - self.initial_balance) / self.initial_balance
        drawdown_penalty = drawdown * reward_config.get('drawdown_penalty', -2.0) if drawdown < 0 else 0

        total_reward = sharpe_reward + holding_penalty + drawdown_penalty
        return total_reward

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one time step within the environment.

        Args:
            action (int): The action to be taken (0: Hold, 1: Long, 2: Short)

        Returns:
            Tuple containing:
            - next_observation (np.ndarray)
            - reward (float)
            - terminated (bool): Whether the episode has ended
            - truncated (bool): Whether the episode was truncated
            - info (Dict): Auxiliary diagnostic information
        """
        previous_equity = self.equity

        # Execute trade based on action
        self._execute_trade(action)

        # Update current step
        self.current_step += 1

        # Update equity based on current price
        current_price = self.market_data.iloc[self.current_step]['close']

        if self.position == 1: # Long
            unrealized_pnl = (current_price - self.entry_price) * self.leverage
            self.equity = self.balance + unrealized_pnl
        elif self.position == -1: # Short
            unrealized_pnl = (self.entry_price - current_price) * self.leverage
            self.equity = self.balance + unrealized_pnl
        else: # Neutral
            self.equity = self.balance

        # Calculate reward
        reward = self._calculate_reward(previous_equity)

        # Check for termination conditions
        terminated = self.equity <= 0 or self.current_step >= len(self.market_data) - 1
        truncated = False # Not used here, but part of the standard API

        if terminated:
            logger.warning(f"Episode terminated at step {self.current_step}. Final equity: ${self.equity:.2f}")

        # Get next observation and info
        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _execute_trade(self, action: int):
        """Execute a trade based on the agent's action"""
        current_price = self.market_data.iloc[self.current_step]['close']

        # Action 1: Go Long
        if action == 1:
            if self.position == 0: # Open new long position
                self.position = 1
                self.entry_price = current_price
                self.balance -= self.transaction_cost * self.balance # Apply transaction cost
            elif self.position == -1: # Close short and open long
                # Close short
                pnl = (self.entry_price - current_price) * self.leverage
                self.balance += pnl - (self.transaction_cost * self.balance)
                # Open long
                self.position = 1
                self.entry_price = current_price
                self.balance -= self.transaction_cost * self.balance

        # Action 2: Go Short
        elif action == 2:
            if self.position == 0: # Open new short position
                self.position = -1
                self.entry_price = current_price
                self.balance -= self.transaction_cost * self.balance
            elif self.position == 1: # Close long and open short
                # Close long
                pnl = (current_price - self.entry_price) * self.leverage
                self.balance += pnl - (self.transaction_cost * self.balance)
                # Open short
                self.position = -1
                self.entry_price = current_price
                self.balance -= self.transaction_cost * self.balance

        # Action 0: Hold (or close position if desired by strategy)
        # For simplicity, this implementation doesn't have a separate "close" action.
        # A change in position (e.g., from long to short) implies closing the previous one.

    def render(self, mode='human'):
        """Render the environment (optional)"""
        if mode == 'human':
            print(f"Step: {self.current_step}, Equity: ${self.equity:.2f}, Position: {self.position}")

def test_environment():
    """Function to test the trading environment"""
    logger.info("🧪 Testing DRL Trading Environment...")

    # Create dummy market data
    data = {
        'open': np.random.uniform(100, 200, 1000),
        'high': np.random.uniform(100, 200, 1000),
        'low': np.random.uniform(100, 200, 1000),
        'close': np.random.uniform(100, 200, 1000),
        'volume': np.random.uniform(1000, 5000, 1000)
    }
    market_data = pd.DataFrame(data)

    # Load DRL config
    try:
        with open("config/drl_config.yaml", 'r') as f:
            drl_config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error("❌ drl_config.yaml not found. Cannot run test.")
        return

    # Initialize environment
    env = TradingEnv(market_data=market_data, config=drl_config)
    obs, info = env.reset()

    # Run a few random steps
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated:
            logger.info("Episode finished.")
            break

    logger.info("✅ Environment test completed successfully!")

if __name__ == "__main__":
    test_environment()
    logger.info("📝 Next: Run 07_stage2_production.py")
