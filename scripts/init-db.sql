-- Database initialization script for PostgreSQL
-- Creates necessary extensions and sets up initial configuration

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Set timezone
SET timezone = 'UTC';

-- Create indexes for performance (tables will be created by SQLAlchemy/Alembic)
-- These are additional indexes beyond what the models define

-- Note: The actual tables are created by SQLAlchemy migrations
-- This script only sets up extensions and database-level configurations
