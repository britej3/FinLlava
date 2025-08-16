#!/usr/bin/env python3
# run_complete_system.py - Main Orchestrator for Complete System Setup
# Fully Autonomous Algorithmic Crypto High Leveraged Futures Scalping Bot

import os
import sys
import subprocess
import logging
from pathlib import Path
import yaml
import time

BOT_NAME = "Fully Autonomous Algorithmic Crypto High Leveraged Futures Scalping Bot"
VERSION = "v1.0.0"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/workspace/setup.log')
    ]
)
logger = logging.getLogger(__name__)

class SystemOrchestrator:
    """Complete system orchestrator for autonomous trading setup"""
    
    def __init__(self):
        self.workspace = Path("/workspace/autonomous_trading_system")
        self.scripts_completed = []
        self.scripts_failed = []
        
        # Define all scripts in execution order
        self.script_pipeline = [
            {
                "name": "01_setup_environment.sh",
                "description": "Environment setup and dependency installation",
                "estimated_time": "15-20 minutes",
                "critical": True
            },
            {
                "name": "02_create_project_structure.py", 
                "description": "Project structure and configuration creation",
                "estimated_time": "2-3 minutes",
                "critical": True
            },
            {
                "name": "03_data_collection.py",
                "description": "Market data collection and educational dataset generation", 
                "estimated_time": "45-60 minutes",
                "critical": True
            },
            {
                "name": "04_core_trading_architecture.py",
                "description": "Core HRM/ZRIA/FinLLaVA architecture setup",
                "estimated_time": "5-10 minutes", 
                "critical": True
            },
            {
                "name": "05_stage1_training.py",
                "description": "Stage 1 model training (development)",
                "estimated_time": "2-4 hours",
                "critical": True,
                "status": "PENDING_CREATION"
            },
            {
                "name": "06_drl_environment.py",
                "description": "Deep RL environment and training setup",
                "estimated_time": "30-45 minutes",
                "critical": True,
                "status": "PENDING_CREATION"
            },
            {
                "name": "07_stage2_production.py",
                "description": "Stage 2 production model training",
                "estimated_time": "4-8 hours",
                "critical": True,
                "status": "PENDING_CREATION"
            },
            {
                "name": "08_autonomous_loop.py",
                "description": "MLE-STAR autonomous improvement loop",
                "estimated_time": "20-30 minutes setup",
                "critical": True,
                "status": "PENDING_CREATION"
            },
            {
                "name": "09_safety_monitoring.py",
                "description": "Safety monitoring and emergency protocols",
                "estimated_time": "15-20 minutes",
                "critical": True,
                "status": "PENDING_CREATION"
            },
            {
                "name": "10_model_deployment.py",
                "description": "Model packaging and deployment preparation",
                "estimated_time": "10-15 minutes",
                "critical": True,
                "status": "PENDING_CREATION"
            }
        ]
    
    def print_system_overview(self):
        """Print comprehensive system overview"""
        logger.info(f"\n{'='*80}")
        logger.info(f"🚀 {BOT_NAME} {VERSION}")
        logger.info(f"{'='*80}")
        logger.info("📋 Complete Setup Pipeline:")
        logger.info("")
        
        total_time_min = 0
        total_time_max = 0
        
        for i, script in enumerate(self.script_pipeline, 1):
            status = script.get('status', 'READY')
            status_emoji = {
                'READY': '✅',
                'PENDING_CREATION': '⏳', 
                'COMPLETED': '✅',
                'FAILED': '❌'
            }.get(status, '❓')
            
            logger.info(f"{i:2d}. {status_emoji} {script['name']}")
            logger.info(f"    📝 {script['description']}")
            logger.info(f"    ⏱️  {script['estimated_time']}")
            logger.info(f"    🔥 Critical: {script.get('critical', False)}")
            
            if script.get('status') == 'PENDING_CREATION':
                logger.info(f"    ⚠️  Status: Needs to be created")
            logger.info("")
            
            # Extract time estimates
            time_str = script['estimated_time'].lower()
            if 'hour' in time_str:
                if '-' in time_str:
                    min_h, max_h = map(int, time_str.split('-')[0].strip().split()[-1]), map(int, time_str.split('-')[1].strip().split()[0])
                    total_time_min += min_h * 60
                    total_time_max += max_h * 60
                else:
                    h = int(time_str.split()[0])
                    total_time_min += h * 60
                    total_time_max += h * 60
            elif 'minute' in time_str:
                if '-' in time_str:
                    min_m = int(time_str.split('-')[0].strip())
                    max_m = int(time_str.split('-')[1].strip().split()[0])
                    total_time_min += min_m
                    total_time_max += max_m
                else:
                    m = int(time_str.split()[0])
                    total_time_min += m
                    total_time_max += m
        
        logger.info(f"⏱️  Total Estimated Time: {total_time_min//60}h {total_time_min%60}m - {total_time_max//60}h {total_time_max%60}m")
        logger.info(f"{'='*80}\n")
    
    def run_script(self, script_info: dict) -> bool:
        """Execute a single script with error handling"""
        script_name = script_info['name']
        logger.info(f"🔄 Starting: {script_name}")
        logger.info(f"📝 {script_info['description']}")
        logger.info(f"⏱️  Estimated time: {script_info['estimated_time']}")
        
        start_time = time.time()
        
        try:
            script_path = self.workspace / script_name
            
            # Check if script exists
            if not script_path.exists():
                if script_info.get('status') == 'PENDING_CREATION':
                    logger.error(f"❌ Script not yet created: {script_name}")
                    logger.info(f"📝 This script needs to be generated next")
                    return False
                else:
                    logger.error(f"❌ Script not found: {script_path}")
                    return False
            
            # Execute based on file extension
            if script_name.endswith('.sh'):
                # Make executable
                os.chmod(script_path, 0o755)
                result = subprocess.run(['bash', str(script_path)], 
                                      capture_output=True, text=True)
            elif script_name.endswith('.py'):
                result = subprocess.run([sys.executable, str(script_path)], 
                                      capture_output=True, text=True)
            else:
                logger.error(f"❌ Unknown script type: {script_name}")
                return False
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                logger.info(f"✅ Completed: {script_name}")
                logger.info(f"⏱️  Actual time: {execution_time:.1f}s")
                
                # Log output if not too verbose
                if result.stdout and len(result.stdout) < 2000:
                    logger.info(f"Output: {result.stdout[-500:]}")  # Last 500 chars
                
                self.scripts_completed.append(script_name)
                return True
            else:
                logger.error(f"❌ Failed: {script_name}")
                logger.error(f"Error code: {result.returncode}")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                
                self.scripts_failed.append({
                    'name': script_name,
                    'error': result.stderr,
                    'code': result.returncode
                })
                return False
                
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ Exception in {script_name}: {str(e)}")
            self.scripts_failed.append({
                'name': script_name,
                'error': str(e),
                'code': -1
            })
            return False
    
    def run_complete_pipeline(self, start_from: str = None, interactive: bool = True):
        """Run the complete setup pipeline"""
        logger.info(f"🚀 Starting complete {BOT_NAME} setup pipeline...")
        
        start_index = 0
        if start_from:
            for i, script in enumerate(self.script_pipeline):
                if script['name'] == start_from:
                    start_index = i
                    break
        
        # Run scripts in sequence
        for i in range(start_index, len(self.script_pipeline)):
            script_info = self.script_pipeline[i]
            
            if interactive and i > start_index:
                response = input(f"\n⏸️  Continue with {script_info['name']}? (y/n/q): ").lower()
                if response == 'q':
                    logger.info("🛑 Pipeline stopped by user")
                    break
                elif response == 'n':
                    logger.info(f"⏭️  Skipping {script_info['name']}")
                    continue
            
            success = self.run_script(script_info)
            
            if not success and script_info.get('critical', False):
                logger.error(f"❌ Critical script failed: {script_info['name']}")
                
                if interactive:
                    response = input("Continue despite failure? (y/n): ").lower()
                    if response != 'y':
                        logger.info("🛑 Pipeline stopped due to critical failure")
                        break
                else:
                    logger.info("🛑 Pipeline stopped due to critical failure")
                    break
        
        # Print final summary
        self.print_final_summary()
    
    def print_final_summary(self):
        """Print final execution summary"""
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 FINAL EXECUTION SUMMARY - {BOT_NAME}")
        logger.info(f"{'='*80}")
        
        logger.info(f"✅ Scripts Completed ({len(self.scripts_completed)}):")
        for script in self.scripts_completed:
            logger.info(f"   • {script}")
        
        if self.scripts_failed:
            logger.info(f"\n❌ Scripts Failed ({len(self.scripts_failed)}):")
            for failure in self.scripts_failed:
                logger.info(f"   • {failure['name']} (Code: {failure['code']})")
        
        # Next steps
        logger.info(f"\n📋 NEXT STEPS:")
        
        if len(self.scripts_completed) == len(self.script_pipeline):
            logger.info("🎉 COMPLETE! All scripts executed successfully!")
            logger.info("🔥 Your autonomous trading system is ready!")
            logger.info(f"📁 Model location: {self.workspace}/data/models/")
            logger.info("🚀 Start autonomous trading with: python run_autonomous_mode.py")
        else:
            remaining = len(self.script_pipeline) - len(self.scripts_completed)
            logger.info(f"⏳ {remaining} scripts remaining to complete setup")
            
            # Show which scripts still need to be created
            pending_creation = [s['name'] for s in self.script_pipeline 
                              if s.get('status') == 'PENDING_CREATION']
            
            if pending_creation:
                logger.info(f"\n📝 Scripts that need to be created next:")
                for script in pending_creation[:3]:  # Show next 3
                    logger.info(f"   • {script}")
        
        logger.info(f"{'='*80}\n")

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description=f'{BOT_NAME} Setup Orchestrator')
    parser.add_argument('--start-from', help='Start from specific script')
    parser.add_argument('--non-interactive', action='store_true', help='Run without prompts')
    parser.add_argument('--overview-only', action='store_true', help='Show overview only')
    
    args = parser.parse_args()
    
    orchestrator = SystemOrchestrator()
    
    # Always show overview
    orchestrator.print_system_overview()
    
    if args.overview_only:
        logger.info("📋 Overview complete. Use --start-from to begin execution.")
        return
    
    # Confirm before starting
    if not args.non_interactive:
        response = input("🚀 Start the complete setup pipeline? (y/n): ").lower()
        if response != 'y':
            logger.info("🛑 Setup cancelled by user")
            return
    
    # Run the pipeline
    orchestrator.run_complete_pipeline(
        start_from=args.start_from,
        interactive=not args.non_interactive
    )

if __name__ == "__main__":
    main()