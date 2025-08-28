#!/bin/bash
source .venv/bin/activate
python main.py --verbose 2>&1 | tee /tmp/ibkr_run.log &
PID=$!
echo "Running with PID $PID for 10 minutes..."
sleep 600
kill -TERM $PID
wait $PID
echo "Test run completed"