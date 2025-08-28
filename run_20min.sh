#!/bin/bash
source .venv/bin/activate
python main.py --verbose 2>&1 | tee /tmp/priority_queue_test.log &
PID=$!
echo "Running with PID $PID for 20 minutes..."
sleep 1200
kill -TERM $PID
wait $PID
echo "Test completed"