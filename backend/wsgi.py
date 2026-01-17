import os
import signal
import sys
import logging
from dotenv import load_dotenv

load_dotenv()

from app import create_app, db

app = create_app()
logger = logging.getLogger(__name__)


def graceful_shutdown(signum, frame):
    logger.info(f"shutting down gracefully...")

    try:
        db.session.remove()
        db.engine.dispose()
        logger.info("database connections closed")
    except Exception as e:
        logger.error(f"error closing database: {e}")

    sys.exit(0)


signal.signal(signal.SIGTERM, graceful_shutdown)
signal.signal(signal.SIGINT, graceful_shutdown)


# initialize database tables
with app.app_context():
    try:
        db.create_all()
        logger.info("database initialized")
    except Exception as e:
        logger.error(f"database error: {e}")


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
