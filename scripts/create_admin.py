#!/usr/bin/env python
"""
Create admin user for Transpara
"""
import sys
import os
from getpass import getpass

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app import create_app, db
from backend.app.models.user import User


def create_admin():
    """Create admin user interactively"""
    app = create_app()

    with app.app_context():
        print("="*60)
        print("Create Transpara Admin User")
        print("="*60)

        email = input("Email: ").strip()
        username = input("Username: ").strip()
        full_name = input("Full Name: ").strip()

        # Password with confirmation
        while True:
            password = getpass("Password: ")
            confirm = getpass("Confirm Password: ")

            if password == confirm:
                break
            print("Passwords do not match. Try again.")

        # Check if user exists
        existing = User.query.filter(
            (User.email == email) | (User.username == username)
        ).first()

        if existing:
            print(f"\nError: User with email '{email}' or username '{username}' already exists!")
            sys.exit(1)

        # Create admin user
        admin = User(
            email=email,
            username=username,
            password=password,
            full_name=full_name,
            role='admin',
            is_active=True,
            is_verified=True
        )

        db.session.add(admin)
        db.session.commit()

        print("\n" + "="*60)
        print(f"Admin user '{username}' created successfully!")
        print("="*60)
        print(f"Email: {email}")
        print(f"Role: admin")
        print("\nYou can now login with these credentials.")


if __name__ == '__main__':
    create_admin()
