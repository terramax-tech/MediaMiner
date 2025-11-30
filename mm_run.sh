################################
# mm_run-v0.1
################################

#!/bin/bash

# Check if virtual environment exists
if [ ! -d "ai_env" ]; then
  echo "[ERROR] Virtual environment not found. Please run 'setup.sh' first."
  exit 1
fi

# Activate virtual environment
source ai_env/bin/activate  # For Linux/Mac
# ai_env\Scripts\activate     # For Windows (if applicable)

# Run the Python script
echo "[INFO] Running media_miner.py..."
python3 media_miner.sh
