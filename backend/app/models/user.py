"""User model for authentication and authorization"""
import os
from datetime import datetime
from app import db


class User(db.Model):
    """User model with secure password storage"""
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    username = db.Column(db.String(100), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)

    # Profile information
    full_name = db.Column(db.String(255))
    organization = db.Column(db.String(255))
    role = db.Column(db.String(50), default='user')  # user, admin, analyst

    # Status and tracking
    is_active = db.Column(db.Boolean, default=True)
    is_verified = db.Column(db.Boolean, default=False)
    last_login = db.Column(db.DateTime)
    failed_login_attempts = db.Column(db.Integer, default=0)
    account_locked_until = db.Column(db.DateTime)

    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    # Relationships - Disabled for no-auth mode
    # analyses = db.relationship('Analysis', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    # api_keys = db.relationship('APIKey', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    # audit_logs = db.relationship('AuditLog', backref='user', lazy='dynamic')

    def __init__(self, email, username, password, **kwargs):
        self.email = email.lower()
        self.username = username
        self.set_password(password)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def set_password(self, password):
        """Hash and set password (disabled)"""
        self.password_hash = "auth_disabled"

    def check_password(self, password):
        """Verify password (disabled)"""
        return True

    def is_account_locked(self):
        """Check if account is locked due to failed login attempts"""
        if self.account_locked_until and self.account_locked_until > datetime.utcnow():
            return True
        return False

    def record_failed_login(self):
        """Record a failed login attempt"""
        self.failed_login_attempts += 1
        if self.failed_login_attempts >= 5:
            # Lock account for 30 minutes
            from datetime import timedelta
            self.account_locked_until = datetime.utcnow() + timedelta(minutes=30)
        db.session.commit()

    def record_successful_login(self):
        """Record successful login and reset failed attempts"""
        self.last_login = datetime.utcnow()
        self.failed_login_attempts = 0
        self.account_locked_until = None
        db.session.commit()

    def to_dict(self, include_email=False):
        """Serialize user to dictionary"""
        data = {
            'id': self.id,
            'username': self.username,
            'full_name': self.full_name,
            'organization': self.organization,
            'role': self.role,
            'is_active': self.is_active,
            'is_verified': self.is_verified,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }
        if include_email:
            data['email'] = self.email
        return data

    def __repr__(self):
        return f'<User {self.username}>'
