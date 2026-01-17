"""
Structured JSON logging configuration
Production-ready logging with different levels for dev/prod
"""
import os
import logging
import sys
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger


def setup_logging(app):
    """Configure structured JSON logging"""

    log_level = app.config.get('LOG_LEVEL', 'INFO')
    log_file = app.config.get('LOG_FILE', 'logs/transpara.log')
    log_max_bytes = app.config.get('LOG_MAX_BYTES', 10 * 1024 * 1024)
    log_backup_count = app.config.get('LOG_BACKUP_COUNT', 10)

    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Custom JSON formatter
    class CustomJsonFormatter(jsonlogger.JsonFormatter):
        def add_fields(self, log_record, record, message_dict):
            super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
            log_record['timestamp'] = record.created
            log_record['level'] = record.levelname
            log_record['logger'] = record.name
            log_record['module'] = record.module
            log_record['function'] = record.funcName
            log_record['line'] = record.lineno

            # Add request context if available
            from flask import has_request_context, request
            if has_request_context():
                log_record['request_id'] = getattr(request, 'request_id', None)
                log_record['remote_addr'] = request.remote_addr
                log_record['method'] = request.method
                log_record['path'] = request.path

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler (for development)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))

    if app.config['DEBUG']:
        # Human-readable format for development
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
    else:
        # JSON format for production
        console_formatter = CustomJsonFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s'
        )
        console_handler.setFormatter(console_formatter)

    root_logger.addHandler(console_handler)

    # File handler with rotation
    if not app.config['DEBUG'] or os.getenv('LOG_TO_FILE'):
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=log_max_bytes,
            backupCount=log_backup_count
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_formatter = CustomJsonFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Configure Flask's logger
    app.logger.setLevel(getattr(logging, log_level.upper()))

    # Configure Werkzeug logger (Flask's HTTP logger)
    werkzeug_logger = logging.getLogger('werkzeug')
    werkzeug_logger.setLevel(logging.WARNING if not app.config['DEBUG'] else logging.INFO)

    app.logger.info("Logging configured successfully",
                   extra={'log_level': log_level, 'log_file': log_file})


def get_audit_logger():
    """Get a dedicated logger for audit events"""
    audit_logger = logging.getLogger('transpara.audit')
    return audit_logger


def log_security_event(event_type, message, **kwargs):
    """Log security-related events"""
    logger = logging.getLogger('transpara.security')
    logger.warning(f"SECURITY EVENT - {event_type}: {message}", extra=kwargs)


def log_audit_event(event_type, user_id, action, **kwargs):
    """Log audit trail events for compliance"""
    audit_logger = get_audit_logger()
    audit_data = {
        'event_type': event_type,
        'user_id': user_id,
        'action': action,
        **kwargs
    }
    audit_logger.info(f"AUDIT: {action}", extra=audit_data)
