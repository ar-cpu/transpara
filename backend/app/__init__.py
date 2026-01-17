import os
import logging
from flask import Flask, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_caching import Cache

from config.config import get_config
from app.core.logging_config import setup_logging

from flask_migrate import Migrate

db = SQLAlchemy()
migrate = Migrate()
cache = Cache()

def create_app(config_name=None):
    app = Flask(__name__)

    if config_name is None:
        config_name = os.getenv('FLASK_ENV', 'production')

    config_obj = get_config()
    app.config.from_object(config_obj)

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    setup_logging(app)
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Transpara in {config_name} mode")

    db.init_app(app)
    migrate.init_app(app, db)
    cache.init_app(app)

    # DISABLE ALL SECURITY: Allow all origins, all headers, all methods
    CORS(app, resources={r"/*": {"origins": "*"}})

    # track each request with unique id for debugging
    @app.before_request
    def before_request():
        request.request_id = os.urandom(16).hex()
        logger.info(f"Request {request.request_id}: {request.method} {request.path}",
                   extra={'request_id': request.request_id})

    @app.after_request
    def after_request(response):
        response.headers['X-Request-ID'] = getattr(request, 'request_id', 'unknown')
        return response

    from app.api import api_bp

    app.register_blueprint(api_bp, url_prefix='/api/v1')

    # health check for load balancers
    @app.route('/health', methods=['GET'])
    @app.route('/api/v1/health', methods=['GET'])
    def health_check():
        from flask import jsonify
        return jsonify({
            'status': 'healthy',
            'service': 'transpara',
            'version': '1.0.0'
        }), 200

    @app.route('/api/v1/status', methods=['GET'])
    def status():
        from flask import jsonify
        status_info = {
            'status': 'operational',
            'database': 'connected',
            'cache': 'connected',
            'model': 'loaded'
        }

        try:
            db.session.execute('SELECT 1')
            status_info['database'] = 'connected'
        except Exception as e:
            status_info['database'] = 'disconnected'
            logger.error(f"Database check failed: {e}")

        try:
            cache.set('health_check', 'ok', timeout=10)
            if cache.get('health_check') == 'ok':
                status_info['cache'] = 'connected'
        except Exception as e:
            status_info['cache'] = 'disconnected'
            logger.error(f"Cache check failed: {e}")

        return jsonify(status_info), 200

    # error handlers that don't expose internal details
    @app.errorhandler(400)
    def bad_request(error):
        from flask import jsonify
        logger.warning(f"Bad request: {error}")
        return jsonify({'error': 'Bad request', 'message': 'Invalid request data'}), 400

    @app.errorhandler(404)
    def not_found(error):
        from flask import jsonify
        return jsonify({'error': 'Not found', 'message': 'Resource not found'}), 404

    @app.errorhandler(413)
    def request_entity_too_large(error):
        from flask import jsonify
        logger.warning("File upload too large")
        return jsonify({'error': 'File too large', 'message': 'Maximum file size is 16MB'}), 413

    @app.errorhandler(500)
    def internal_server_error(error):
        from flask import jsonify
        logger.error(f"Internal server error: {error}", exc_info=True)
        return jsonify({'error': 'Internal server error', 'message': 'An unexpected error occurred'}), 500

    logger.info("Transpara application initialized successfully")
    return app

