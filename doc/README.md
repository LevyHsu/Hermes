# Documentation

This directory contains all documentation for the Hermes trading signal system.

## 📚 Available Documentation

### Core Documentation

- **[CLAUDE.md](CLAUDE.md)** - Project guidance for Claude AI assistant
- **[MAIN_USAGE.md](MAIN_USAGE.md)** - Complete usage guide for the main orchestrator
- **[SMART_SCHEDULING.md](SMART_SCHEDULING.md)** - Smart scheduling optimization details
- **[SHUTDOWN_FIX.md](SHUTDOWN_FIX.md)** - Graceful shutdown implementation notes

### Quick Links

#### System Setup
- [Installation Guide](MAIN_USAGE.md#quick-start)
- [Configuration Options](MAIN_USAGE.md#configuration-argspy)
- [Production Deployment](MAIN_USAGE.md#production-deployment)

#### Features
- [Smart Scheduling](SMART_SCHEDULING.md)
- [Trading Signals](MAIN_USAGE.md#trade-log-format)
- [Log Files](MAIN_USAGE.md#log-files)

#### Troubleshooting
- [Shutdown Issues](SHUTDOWN_FIX.md)
- [Common Problems](MAIN_USAGE.md#troubleshooting)
- [Monitoring](MAIN_USAGE.md#monitoring)

## 📖 Documentation Overview

### CLAUDE.md
Project instructions and guidelines for AI assistants working with this codebase. Includes:
- Project overview
- System architecture
- Data flow
- Key implementation details
- Critical requirements

### MAIN_USAGE.md
Comprehensive guide for using the main orchestrator. Covers:
- Quick start instructions
- Configuration via args.py
- Command line options
- Log file descriptions
- System flow and timing
- Monitoring and troubleshooting
- Production deployment options

### SMART_SCHEDULING.md
Details the adaptive time allocation system:
- How smart scheduling works
- Time allocation table
- Configuration parameters
- Performance impact
- Example scenarios

### SHUTDOWN_FIX.md
Technical documentation of the graceful shutdown implementation:
- Problem description
- Solution implemented
- Key code changes
- Testing procedures
- Emergency kill procedures

## 🔗 Related Resources

- Main README: [../README.md](../README.md)
- Test Documentation: [../test/README.md](../test/README.md)
- Configuration: [../args.py](../args.py)