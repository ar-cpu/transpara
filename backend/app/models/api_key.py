"""API Key model for programmatic access"""
import os
import hashlib
from datetime import datetime, timedelta
from app import db


class APIKey(db.Model):
    """API keys for programmatic access to the API"""
    __tablename__ = 'api_keys'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)

    # Key information
    key_hash = db.Column(db.String(64), unique=True, nullable=False, index=True)  # SHA-256 hash
    key_prefix = db.Column(db.String(8), nullable=False)  # First 8 chars for identification
    name = db.Column(db.String(100), nullable=False)  # User-friendly name
    description = db.Column(db.Text)

    # Permissions and limits
    scopes = db.Column(db.JSON, default=list)  # List of allowed scopes: ['analysis', 'admin', etc.]
    rate_limit = db.Column(db.Integer, default=1000)  # Requests per hour

    # Status
    is_active = db.Column(db.Boolean, default=True)
    expires_at = db.Column(db.DateTime)
    last_used_at = db.Column(db.DateTime)
    usage_count = db.Column(db.Integer, default=0)

    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __init__(self, user_id, name, **kwargs):
        self.user_id = user_id
        self.name = name
        for key, value in kwargs.items():
            setattr(self, key, value)

    @staticmethod
    def generate_key():
        """Generate a new API key"""
        # Generate 32-byte random key and encode as hex
        key = os.urandom(32).hex()
        return f"tk_{key}"  # Prefix with 'tk_' for "Transpara Key"

    @staticmethod
    def hash_key(key):
        """Hash an API key"""
        return hashlib.sha256(key.encode()).hexdigest()

    def set_key(self, key):
        """Set the API key (stores hash only)"""
        self.key_hash = self.hash_key(key)
        self.key_prefix = key[:8]

    @staticmethod
    def verify_key(key):
        """Verify an API key and return the associated APIKey object"""
        key_hash = APIKey.hash_key(key)
        api_key = APIKey.query.filter_by(key_hash=key_hash).first()

        if not api_key:
            return None

        # Check if key is active
        if not api_key.is_active:
            return None

        # Check if key has expired
        if api_key.expires_at and api_key.expires_at < datetime.utcnow():
            return None

        # Update usage stats
        api_key.last_used_at = datetime.utcnow()
        api_key.usage_count += 1
        db.session.commit()

        return api_key

    def is_expired(self):
        """Check if the API key has expired"""
        if not self.expires_at:
            return False
        return self.expires_at < datetime.utcnow()

    def has_scope(self, scope):
        """Check if the API key has a specific scope"""
        if not self.scopes:
            return True  # No scopes means all access
        return scope in self.scopes

    def to_dict(self, include_key_hash=False):
        """Serialize API key to dictionary"""
        data = {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'key_prefix': self.key_prefix,
            'scopes': self.scopes,
            'rate_limit': self.rate_limit,
            'is_active': self.is_active,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'last_used_at': self.last_used_at.isoformat() if self.last_used_at else None,
            'usage_count': self.usage_count,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
        if include_key_hash:
            data['key_hash'] = self.key_hash
        return data

    def __repr__(self):
        return f'<APIKey {self.name} ({self.key_prefix}...)>'
