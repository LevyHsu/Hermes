#!/usr/bin/env python3
"""
Compatibility shim for Python 3.13+ where cgi module was removed.
Provides parse_header function needed by feedparser.
"""

def parse_header(line):
    """
    Parse a Content-Type like header.
    
    Simple version of cgi.parse_header for feedparser compatibility.
    Returns (main_value, params_dict)
    """
    parts = line.split(';')
    main = parts[0].strip()
    params = {}
    
    for part in parts[1:]:
        if '=' in part:
            key, value = part.split('=', 1)
            key = key.strip()
            value = value.strip()
            # Remove quotes if present
            if value and value[0] == value[-1] == '"':
                value = value[1:-1]
            params[key] = value
    
    return main, params