# Graceful Shutdown Fix

## Problem
The program wasn't shutting down properly with Ctrl+C (SIGINT), leaving zombie processes running.

## Solution Implemented

### 1. Process Tracking
- Added `CHILD_PROCESSES` list to track all spawned subprocesses
- Each subprocess is added when created and removed when completed

### 2. Enhanced Signal Handling
```python
def handle_signal(signum, frame):
    global STOP
    STOP = True
    logging.info(f"Received signal {signum}, shutting down gracefully...")
    cleanup_child_processes()
```

### 3. Process Group Management
- All subprocesses are created with `preexec_fn=os.setsid` to create new process groups
- This allows killing entire process trees with `os.killpg()`

### 4. Subprocess Changes
- Changed from `subprocess.run()` to `subprocess.Popen()` for better control
- Added proper timeout handling with process group termination
- Two-stage termination: SIGTERM first, then SIGKILL if needed

### 5. Responsive Sleep
Instead of `time.sleep(60)`, now uses:
```python
while not STOP and (time.time() - sleep_start) < sleep_time:
    time.sleep(0.5)  # Check every 0.5 seconds
```

### 6. Early Exit Checks
Added `if STOP: return None` checks at the beginning of long-running functions.

## Key Changes Made

### main.py
1. **Global tracking**: `CHILD_PROCESSES = []` list
2. **cleanup_child_processes()**: Kills all tracked processes
3. **Enhanced signal handler**: Calls cleanup on SIGINT/SIGTERM
4. **Popen instead of run**: Better process control
5. **Process group creation**: `preexec_fn=os.setsid`
6. **Responsive sleep loop**: Checks STOP flag every 0.5s
7. **Graceful + forced kill**: Try SIGTERM, then SIGKILL

## Testing

Run the test script to verify:
```bash
python test_shutdown.py
```

This will:
1. Start main.py
2. Wait 5 seconds
3. Send SIGINT (Ctrl+C)
4. Verify graceful shutdown

## Manual Testing

1. Start the bot:
```bash
python main.py --verbose
```

2. Press Ctrl+C

3. Should see:
```
[SHUTDOWN] Received signal 2, shutting down gracefully...
HERMES SHUTTING DOWN
All child processes terminated
```

4. Verify no zombie processes:
```bash
ps aux | grep -E "python.*(main|news|llm)" | grep -v grep
```

## Emergency Kill

If still stuck, use the emergency kill script:
```bash
# Kill all Python processes related to the bot
pkill -f "python.*main.py"
pkill -f "python.*news_harvester.py"
pkill -f "python.*llm.py"
```

## Benefits

1. **Clean shutdown**: All processes terminate properly
2. **No zombies**: Child processes are tracked and killed
3. **Responsive**: Checks STOP flag every 0.5 seconds
4. **Two-stage kill**: Graceful first, then forced
5. **Process groups**: Entire trees killed together
6. **Timeout protection**: Processes can't hang indefinitely