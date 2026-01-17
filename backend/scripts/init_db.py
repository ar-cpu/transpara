#!/usr/bin/env python3
"""
Initialize database tables for Transpara

This script creates all database tables defined in the models.
Run this after the database is created but before the first use.
"""

import os
import sys
from pathlib import Path

# add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import create_app, db

def init_database():
    """Create all database tables"""
    app = create_app()

    with app.app_context():
        print("creating database tables...")

        # create all tables
        db.create_all()

        print("✅ database tables created successfully!")
        print("\ntables created:")
        print("  - users (authentication and user management)")
        print("  - analysis_history (stores analysis results)")
        print("  - api_keys (api key management)")
        print("  - audit_logs (security and audit tracking)")

        print("\nnext step: create admin user")
        print("  docker-compose exec backend python scripts/create_admin.py")

if __name__ == '__main__':
    try:
        init_database()
    except Exception as e:
        print(f"❌ error creating database tables: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
