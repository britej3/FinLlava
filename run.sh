#!/bin/bash

# run.sh - Main executable to run the entire FinLLaVA trading bot pipeline

echo "🚀 Starting the FinLLaVA Trading Bot Pipeline..."

# Activate conda environment
# This assumes conda is initialized in your shell.
source activate trading_agent

# Check if conda environment activation was successful
if [ $? -ne 0 ]; then
    echo "❌ Failed to activate conda environment 'trading_agent'."
    echo "Please make sure you have run the '01_setup_environment.sh' script successfully."
    exit 1
fi

# Run the main orchestrator
echo "▶️ Starting the Main Orchestrator..."
python main_orchestrator.py "$@"

# Check if the orchestrator ran successfully
if [ $? -eq 0 ]; then
    echo "✅ Pipeline finished successfully."
else
    echo "❌ Pipeline encountered an error."
fi
