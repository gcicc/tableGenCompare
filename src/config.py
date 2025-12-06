"""
Global Configuration and Session Management

This module manages global configuration state for the Clinical Synthetic
Data Generation Framework, including session timestamps and dataset identifiers.
"""

from datetime import datetime

# Session timestamp (captured at import time)
SESSION_TIMESTAMP = datetime.now().strftime("%Y-%m-%d")

# Global variables to be set when data is loaded
DATASET_IDENTIFIER = None
CURRENT_DATA_FILE = None

def refresh_session_timestamp():
    """Refresh the session timestamp to current date"""
    global SESSION_TIMESTAMP
    SESSION_TIMESTAMP = datetime.now().strftime("%Y-%m-%d")
    print(f"Session timestamp refreshed to: {SESSION_TIMESTAMP}")
    return SESSION_TIMESTAMP

print(f"[CONFIG] Session timestamp: {SESSION_TIMESTAMP}")
