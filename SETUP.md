# Transpara - Setup Guide

This guide will help you set up and run the Transpara application using Docker.

## Prerequisites

- **Docker Desktop** installed and running.
- **Git** (optional, for cloning).

## Quick Start

1.  **Clone or Open the Repository**
    Navigate to the project root directory in your terminal.
    ```bash
    cd new_project
    ```

2.  **Start the Application**
    Run the following command to build and start all services (Frontend, Backend, Database, Redis, Nginx):
    ```bash
    docker-compose up -d --build
    ```
    *Note: The first build may take a few minutes as it downloads dependencies and compiles the frontend.*

3.  **Access the Application**
    Once the containers are running, open your web browser and go to:
    
    👉 **http://localhost**

## Architecture

- **Frontend**: Angular application running on port 4200 (proxied via Nginx).
- **Backend**: Flask API running on port 5000.
- **Database**: PostgreSQL (persisted in `postgres_data` volume).
- **Cache**: Redis.
- **Nginx**: Reverse proxy serving the app on port 80.

## Troubleshooting

- **"Internal Server Error" on Analysis**: 
  Ensure the model file exists. The app attempts to train/load it automatically. Check logs:
  ```bash
  docker-compose logs backend
  ```

- **Frontend not loading**:
  The frontend container runs `npm install` on startup. Give it a minute. Check its status:
  ```bash
  docker-compose logs -f frontend
  ```

- **Database Issues**:
  To reset the database, spin down and remove volumes:
  ```bash
  docker-compose down -v
  docker-compose up -d
  ```

## Development

- **Backend Code**: Changes in `backend/` are hot-reloaded.
- **Frontend Code**: Changes in `frontend/src` are hot-reloaded via Angular CLI.
