#!/usr/bin/env python3
"""
Create admin user for Transpara

This script creates the default admin account using credentials from .env file:
- ADMIN_USERNAME=admin
- ADMIN_EMAIL=admin@transpara.local
- ADMIN_PASSWORD=Admin123!
"""

import os
import sys
from pathlib import Path

# add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import create_app, db
from app.models import User

def create_admin_user():
    """Create or update admin user"""
    app = create_app()

    with app.app_context():
        # get admin credentials from environment
        admin_username = os.getenv('ADMIN_USERNAME', 'admin')
        admin_email = os.getenv('ADMIN_EMAIL', 'admin@transpara.local')
        admin_password = os.getenv('ADMIN_PASSWORD', 'Admin123!')

        # check if admin user already exists
        admin = User.query.filter_by(username=admin_username).first()

        if admin:
            print(f"✓ Admin user '{admin_username}' already exists")
            print(f"  Email: {admin.email}")
            print(f"  User ID: {admin.id}")

            # update password in case it changed
            admin.set_password(admin_password)
            admin.role = 'admin'
            admin.is_active = True
            admin.is_verified = True
            db.session.commit()
            print(f"✓ Admin password updated")
        else:
            # create new admin user
            admin = User(
                email=admin_email,
                username=admin_username,
                password=admin_password,
                full_name='System Administrator',
                organization='Transpara',
                role='admin',
                is_active=True,
                is_verified=True
            )

            db.session.add(admin)
            db.session.commit()

            print(f"✓ Admin user created successfully!")
            print(f"  Username: {admin_username}")
            print(f"  Email: {admin_email}")
            print(f"  Password: {admin_password}")
            print(f"  User ID: {admin.id}")

        print("\n✅ Admin account is ready to use!")
        print(f"\nLogin credentials:")
        print(f"  Username: {admin_username}")
        print(f"  Password: {admin_password}")
        print(f"\nOr click 'Quick Admin Login' button in the frontend")

if __name__ == '__main__':
    try:
        create_admin_user()
    except Exception as e:
        print(f"❌ Error creating admin user: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
