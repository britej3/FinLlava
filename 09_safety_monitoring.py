import yaml
from pathlib import Path
import logging
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SafetyMonitor:
    """
    Monitors the trading bot's performance and enforces safety protocols.
    """
    def __init__(self, portfolio_state: Dict[str, Any]):
        self.base_path = Path("/workspace/autonomous_trading_system")
        self.config = self.load_safety_config()
        self.risk_limits = self.config.get('risk_limits', {})
        self.emergency_protocols = self.config.get('emergency_protocols', {})

        # Live portfolio state (should be updated by the main trading loop)
        self.portfolio_state = portfolio_state

        # State tracking for monitoring
        self.daily_pnl = 0
        self.peak_equity = portfolio_state.get('equity', 0)
        self.drawdown = 0

        logger.info("🛡️ Safety Monitor initialized.")

    def load_safety_config(self) -> Dict[str, Any]:
        """Load the safety configuration file"""
        config_path = self.base_path / "config" / "safety_config.yaml"
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info("✅ Safety configuration loaded.")
            return config
        except FileNotFoundError:
            logger.error(f"❌ Safety config not found at {config_path}.")
            return {}

    def update_portfolio_state(self, new_state: Dict[str, Any]):
        """Update the monitor with the latest portfolio state"""
        previous_equity = self.portfolio_state.get('equity', 0)
        self.portfolio_state = new_state

        # Update daily PnL
        pnl_change = self.portfolio_state['equity'] - previous_equity
        self.daily_pnl += pnl_change

        # Update drawdown
        self.peak_equity = max(self.peak_equity, self.portfolio_state['equity'])
        self.drawdown = (self.peak_equity - self.portfolio_state['equity']) / self.peak_equity

        # Run all checks
        self.enforce_all_rules()

    def check_max_position_size(self, proposed_position_size: float) -> bool:
        """Check if a proposed trade exceeds the max position size."""
        max_size = self.risk_limits.get('max_position_size', 0.3)
        current_capital = self.portfolio_state.get('equity', 0)

        if (proposed_position_size / current_capital) > max_size:
            logger.warning(f"⚠️ Position size violation: Proposed {proposed_position_size} > Limit {max_size * 100}%.")
            return False
        return True

    def check_daily_loss_limit(self) -> bool:
        """Check if the daily loss limit has been breached."""
        max_daily_loss = self.risk_limits.get('max_daily_loss', 0.02)

        if self.daily_pnl < 0 and abs(self.daily_pnl / self.peak_equity) > max_daily_loss:
            logger.critical(f"🚨 Daily loss limit breached! Loss: {self.daily_pnl:.2f} > Limit: {max_daily_loss * 100}%")
            self.trigger_emergency_protocol("daily_loss_limit_breached")
            return False
        return True

    def check_max_drawdown(self) -> bool:
        """Check if the maximum drawdown has been breached."""
        max_drawdown = self.risk_limits.get('max_drawdown', 0.15)

        if self.drawdown > max_drawdown:
            logger.critical(f"🚨 Max drawdown breached! Drawdown: {self.drawdown * 100:.2f}% > Limit: {max_drawdown * 100}%")
            self.trigger_emergency_protocol("max_drawdown_breached")
            return False
        return True

    def enforce_all_rules(self):
        """Run all safety checks."""
        if not self.check_daily_loss_limit() or not self.check_max_drawdown():
            # Emergency protocols are triggered within the check functions
            pass

    def trigger_emergency_protocol(self, reason: str):
        """Trigger emergency protocols based on the reason."""
        logger.critical(f"🔥 EMERGENCY PROTOCOL TRIGGERED: {reason} 🔥")

        if self.emergency_protocols.get('auto_shutdown', True):
            logger.critical("... Initiating automated shutdown ...")
            # In a real system, this would involve:
            # 1. Closing all open positions via the exchange API.
            # 2. Canceling all open orders.
            # 3. Sending alerts.
            self.send_alert(f"Emergency shutdown triggered due to: {reason}")

            # Halt further trading by exiting the script.
            # This is a drastic measure for a critical failure.
            logger.critical("🛑 TRADING HALTED. MANUAL INTERVENTION REQUIRED. 🛑")
            sys.exit(1)

    def send_alert(self, message: str):
        """Send an alert to the user (placeholder)."""
        alert_channels = self.emergency_protocols.get('alert_channels', [])

        if "email" in alert_channels:
            logger.info(f"📧 Sending email alert: {message}")
            # Placeholder for email sending logic

        if "telegram" in alert_channels:
            logger.info(f"💬 Sending Telegram alert: {message}")
            # Placeholder for Telegram bot API logic

def test_safety_monitor():
    """Function to test the SafetyMonitor"""
    logger.info("🧪 Testing Safety Monitor...")

    # Initial portfolio state
    initial_state = {'equity': 10000, 'balance': 10000}
    monitor = SafetyMonitor(portfolio_state=initial_state)

    # Test position size check
    is_safe = monitor.check_max_position_size(2000) # 20% of capital
    logger.info(f"Position size check (20%): {'✅ Safe' if is_safe else '❌ Unsafe'}")
    is_safe = monitor.check_max_position_size(4000) # 40% of capital (should fail)
    logger.info(f"Position size check (40%): {'✅ Safe' if is_safe else '❌ Unsafe'}")

    # Simulate a loss
    logger.info("\nSimulating a trading loss...")
    new_state = {'equity': 9800, 'balance': 9800} # $200 loss
    monitor.update_portfolio_state(new_state)
    logger.info(f"Drawdown: {monitor.drawdown * 100:.2f}%, Daily PnL: {monitor.daily_pnl}")

    # Simulate a larger loss to trigger daily loss limit (but not max drawdown)
    logger.info("\nSimulating a larger loss...")
    new_state = {'equity': 9700, 'balance': 9700} # $300 total loss (3%)
    try:
        monitor.update_portfolio_state(new_state)
    except SystemExit:
        logger.info("✅ System exit correctly triggered by daily loss limit.")

    # Reset and test max drawdown
    logger.info("\nTesting max drawdown...")
    monitor = SafetyMonitor(portfolio_state=initial_state)
    new_state = {'equity': 8400, 'balance': 8400} # 16% drawdown
    try:
        monitor.update_portfolio_state(new_state)
    except SystemExit:
        logger.info("✅ System exit correctly triggered by max drawdown.")

    logger.info("✅ Safety Monitor test completed.")

if __name__ == "__main__":
    test_safety_monitor()
    logger.info("📝 Next: Run 10_model_deployment.py")
