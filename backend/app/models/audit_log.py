"""Audit log model for compliance and security monitoring"""
from datetime import datetime
from app import db


class AuditLog(db.Model):
    """Audit trail for all sensitive operations"""
    __tablename__ = 'audit_logs'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, index=True) # Authentication disabled

    # Event information
    event_type = db.Column(db.String(50), nullable=False, index=True)  # login, logout, analysis, admin_action, etc.
    action = db.Column(db.String(255), nullable=False)
    resource_type = db.Column(db.String(50))  # user, analysis, api_key, etc.
    resource_id = db.Column(db.String(100))

    # Details
    details = db.Column(db.JSON)  # Additional context
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.String(255))

    # Result
    success = db.Column(db.Boolean, default=True)
    error_message = db.Column(db.Text)

    # Timestamp
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)

    def __init__(self, **kwargs):
        super(AuditLog, self).__init__(**kwargs)

    def to_dict(self):
        """Serialize audit log to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'event_type': self.event_type,
            'action': self.action,
            'resource_type': self.resource_type,
            'resource_id': self.resource_id,
            'details': self.details,
            'ip_address': self.ip_address,
            'success': self.success,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

    @staticmethod
    def log_event(event_type, action, user_id=None, resource_type=None,
                  resource_id=None, details=None, success=True, error_message=None):
        """Helper method to create audit log entry"""
        from flask import request

        audit_log = AuditLog(
            user_id=user_id,
            event_type=event_type,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
            ip_address=request.remote_addr if request else None,
            user_agent=request.user_agent.string if request else None,
            success=success,
            error_message=error_message
        )
        db.session.add(audit_log)
        db.session.commit()
        return audit_log

    def __repr__(self):
        return f'<AuditLog {self.event_type} - {self.action}>'
