"""Analysis model for tracking bias detection requests"""
from datetime import datetime
from app import db


class Analysis(db.Model):
    """Analysis request tracking for analytics and auditing"""
    __tablename__ = 'analyses'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=True, index=True) # Authentication disabled

    # Request information
    request_id = db.Column(db.String(32), unique=True, nullable=False, index=True)
    input_type = db.Column(db.String(50), nullable=False)  # text, audio, video, pdf, docx
    input_length = db.Column(db.Integer)  # word count or file size

    # Analysis results
    prediction = db.Column(db.String(20))  # left, center, right
    confidence = db.Column(db.Float)
    probabilities = db.Column(db.JSON)  # Store full probability distribution

    # Metadata
    processing_time_ms = db.Column(db.Integer)  # Processing time in milliseconds
    model_version = db.Column(db.String(50), default='1.0')
    client_ip = db.Column(db.String(45))  # Support IPv6
    user_agent = db.Column(db.String(255))

    # Status
    status = db.Column(db.String(20), default='completed')  # pending, completed, failed
    error_message = db.Column(db.Text)

    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)

    def __init__(self, **kwargs):
        super(Analysis, self).__init__(**kwargs)

    def to_dict(self):
        """Serialize analysis to dictionary"""
        return {
            'id': self.id,
            'request_id': self.request_id,
            'input_type': self.input_type,
            'input_length': self.input_length,
            'prediction': self.prediction,
            'confidence': self.confidence,
            'probabilities': self.probabilities,
            'processing_time_ms': self.processing_time_ms,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

    def __repr__(self):
        return f'<Analysis {self.request_id} - {self.prediction}>'
