"""
Production-ready configuration with environment-specific settings
"""
import os
from datetime import timedelta
from typing import List


class Config:
    """Base configuration"""

    # Flask
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = False
    TESTING = False

    # CORS
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:4200').split(',')

    # Database
    SQLALCHEMY_DATABASE_URI = os.getenv(
        'DATABASE_URL',
        'postgresql://transpara:transpara@localhost:5432/transpara_db'
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': int(os.getenv('DB_POOL_SIZE', 20)),
        'max_overflow': int(os.getenv('DB_MAX_OVERFLOW', 10)),
        'pool_timeout': int(os.getenv('DB_POOL_TIMEOUT', 30)),
        'pool_recycle': int(os.getenv('DB_POOL_RECYCLE', 3600)),
        'pool_pre_ping': True,
    }

    # Redis
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    CACHE_TYPE = 'redis'
    CACHE_REDIS_URL = REDIS_URL
    CACHE_DEFAULT_TIMEOUT = int(os.getenv('CACHE_DEFAULT_TIMEOUT', 300))

    # File Upload Security
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 100 * 1024 * 1024))  # 100MB
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', '/tmp/transpara_uploads')
    ALLOWED_EXTENSIONS = {'pdf', 'docx', 'wav', 'mp3', 'mp4', 'avi', 'webm'}

    # File type magic numbers for validation
    ALLOWED_MIME_TYPES = {
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'audio/wav', 'audio/wave', 'audio/x-wav',
        'audio/mpeg', 'audio/mp3',
        'video/mp4',
        'video/x-msvideo',
        'video/webm',
    }

    # External APIs
    GOOGLE_SPEECH_TIMEOUT = int(os.getenv('GOOGLE_SPEECH_TIMEOUT', 30))

    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'logs/transpara.log')
    LOG_MAX_BYTES = int(os.getenv('LOG_MAX_BYTES', 10 * 1024 * 1024))
    LOG_BACKUP_COUNT = int(os.getenv('LOG_BACKUP_COUNT', 10))

    # Feature Flags
    ENABLE_BATCH_PROCESSING = os.getenv('ENABLE_BATCH_PROCESSING', 'True') == 'True'
    ENABLE_WEBHOOKS = os.getenv('ENABLE_WEBHOOKS', 'True') == 'True'
    ENABLE_CACHING = os.getenv('ENABLE_CACHING', 'True') == 'True'
    ENABLE_AUDIT_LOGGING = os.getenv('ENABLE_AUDIT_LOGGING', 'True') == 'True'

    # Data Retention
    DATA_RETENTION_DAYS = int(os.getenv('DATA_RETENTION_DAYS', 90))

    # Email Configuration
    SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
    SMTP_USERNAME = os.getenv('SMTP_USERNAME', '')
    SMTP_PASSWORD = os.getenv('SMTP_PASSWORD', '')
    SMTP_USE_TLS = os.getenv('SMTP_USE_TLS', 'True') == 'True'
    ADMIN_EMAIL = os.getenv('ADMIN_EMAIL', 'admin@transpara.com')

    # Monitoring
    PROMETHEUS_PORT = int(os.getenv('PROMETHEUS_PORT', 9090))


class DevelopmentConfig(Config):
    """Development configuration"""
    CORS_ORIGINS = [
        'http://localhost:4200',
        'http://localhost:3000',
        'http://localhost',
        'http://127.0.0.1',
        'http://localhost:80'
    ]
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///transpara.db')


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False

    # Enforce HTTPS
    PREFERRED_URL_SCHEME = 'https'


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'postgresql://transpara:transpara@localhost:5432/transpara_test'


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config():
    """Get configuration based on environment"""
    env = os.getenv('FLASK_ENV', 'production')
    return config.get(env, config['production'])
