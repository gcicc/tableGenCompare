#!/usr/bin/env python3
"""
Production API server launcher with configuration management.
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path

# Add src to path
sys.path.append('src')

import uvicorn
from api.main import app

def load_config(config_path: str = "api_config.yaml", environment: str = "development"):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Apply environment-specific overrides
        if environment in config:
            env_config = config[environment]
            for section, settings in env_config.items():
                if section in config:
                    config[section].update(settings)
                else:
                    config[section] = settings
        
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}

def setup_logging(config: dict):
    """Setup logging configuration."""
    log_config = config.get('logging', {})
    
    level = getattr(logging, log_config.get('level', 'INFO').upper())
    format_str = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create log directory if specified
    log_file = log_config.get('file')
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file handler with rotation
        from logging.handlers import RotatingFileHandler
        
        max_bytes = log_config.get('max_file_size_mb', 10) * 1024 * 1024
        backup_count = log_config.get('backup_count', 5)
        
        handler = RotatingFileHandler(
            log_file, 
            maxBytes=max_bytes, 
            backupCount=backup_count
        )
        handler.setFormatter(logging.Formatter(format_str))
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        root_logger.addHandler(handler)
        
        # Also log to console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(format_str))
        root_logger.addHandler(console_handler)
    else:
        # Just console logging
        logging.basicConfig(level=level, format=format_str)

def setup_directories(config: dict):
    """Create necessary directories."""
    api_config = config.get('api', {})
    
    directories = [
        api_config.get('data_directory', 'api_data'),
        api_config.get('upload_directory', 'api_data/uploads'),
        api_config.get('output_directory', 'api_data/outputs'),
        api_config.get('model_directory', 'api_data/models'),
        'api_data/logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def main():
    parser = argparse.ArgumentParser(description='Run Synthetic Data Generation API')
    parser.add_argument('--config', default='api_config.yaml', help='Configuration file path')
    parser.add_argument('--env', default='development', choices=['development', 'production'], 
                       help='Environment')
    parser.add_argument('--host', help='Host address (overrides config)')
    parser.add_argument('--port', type=int, help='Port number (overrides config)')
    parser.add_argument('--workers', type=int, help='Number of workers (overrides config)')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    parser.add_argument('--log-level', choices=['debug', 'info', 'warning', 'error'], 
                       help='Log level (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config, args.env)
    api_config = config.get('api', {})
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    # Setup directories
    setup_directories(config)
    
    # Apply command line overrides
    host = args.host or api_config.get('host', '0.0.0.0')
    port = args.port or api_config.get('port', 8000)
    workers = args.workers or api_config.get('workers', 1)
    reload = args.reload or api_config.get('reload', False)
    log_level = args.log_level or api_config.get('log_level', 'info')
    
    # Set environment variables for the app to use
    os.environ['API_CONFIG'] = args.config
    os.environ['API_ENV'] = args.env
    
    logger.info(f"Starting Synthetic Data Generation API")
    logger.info(f"Environment: {args.env}")
    logger.info(f"Host: {host}:{port}")
    logger.info(f"Workers: {workers}")
    logger.info(f"Reload: {reload}")
    
    # Check if we're in development mode
    if args.env == 'development' or reload:
        # Development mode - single process with reload
        uvicorn.run(
            "api.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level=log_level.lower(),
            access_log=True
        )
    else:
        # Production mode - multiple workers
        uvicorn.run(
            "api.main:app",
            host=host,
            port=port,
            workers=workers,
            log_level=log_level.lower(),
            access_log=True
        )

if __name__ == "__main__":
    main()