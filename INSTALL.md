# Installation Guide

## Option 1: Docker (Recommended - Everything Included)

### Prerequisites
- Docker Desktop
- Git

### Install
```bash
# clone or navigate to project
cd transpara

# copy your model
cp bias_detector_model.pkl backend/models/

# start everything
docker-compose up -d
```

**That's it!** Docker includes Python, Node, PostgreSQL, Redis, FFmpeg, and everything else.

---

## Option 2: Manual Installation

### Prerequisites

#### Required Software
1. **Python 3.11+**
   ```bash
   python --version  # should be 3.11 or higher
   ```

2. **Node.js 21+**
   ```bash
   node --version  # should be 21 or higher
   ```

3. **PostgreSQL 15+**
   ```bash
   # mac
   brew install postgresql@15

   # ubuntu
   sudo apt install postgresql-15

   # windows
   # download from postgresql.org
   ```

4. **Redis 7+**
   ```bash
   # mac
   brew install redis

   # ubuntu
   sudo apt install redis-server

   # windows
   # download from redis.io or use WSL
   ```

5. **FFmpeg** (for video processing)
   ```bash
   # mac
   brew install ffmpeg

   # ubuntu
   sudo apt install ffmpeg

   # windows
   choco install ffmpeg
   # or download from ffmpeg.org
   ```

6. **python-magic dependencies**
   ```bash
   # mac
   brew install libmagic

   # ubuntu
   sudo apt install libmagic1

   # windows
   pip install python-magic-bin
   ```

---

### Backend Installation

```bash
cd backend

# create virtual environment
python3 -m venv venv

# activate it
source venv/bin/activate  # mac/linux
# or
venv\Scripts\activate  # windows

# upgrade pip
pip install --upgrade pip

# install all dependencies
pip install -r requirements.txt
```

#### Backend Dependencies (from requirements.txt)

**Core Framework:**
- Flask==3.0.0
- Werkzeug==3.0.1
- gunicorn==21.2.0

**Security:**
- Flask-CORS==4.0.0
- Flask-Talisman==1.1.0
- python-dotenv==1.0.0
- cryptography==41.0.7



**Database:**
- Flask-SQLAlchemy==3.1.1
- psycopg2-binary==2.9.9
- alembic==1.13.1
- SQLAlchemy==2.0.23

**Redis/Caching:**
- redis==5.0.1
- Flask-Caching==2.1.0

**ML Dependencies:**
- scikit-learn==1.3.2
- numpy==1.26.2
- scipy==1.11.4
- joblib==1.3.2

**Document Processing:**
- PyPDF2==3.0.1
- python-docx==1.1.0

**Audio/Video Processing:**
- SpeechRecognition==3.10.1
- pydub==0.25.1

**Validation:**
- marshmallow==3.20.1
- email-validator==2.1.0
- python-magic==0.4.27

**Monitoring & Logging:**
- prometheus-flask-exporter==0.23.0
- sentry-sdk[flask]==1.39.1
- python-json-logger==2.0.7

**API Documentation:**
- flask-swagger-ui==4.11.1
- apispec==6.3.1
- apispec-webframeworks==1.0.0

**Utilities:**
- python-dateutil==2.8.2
- pytz==2023.3
- requests==2.31.0
- celery==5.3.4

---

### Frontend Installation

```bash
cd frontend

# install dependencies
npm install
```

#### Frontend Dependencies (from package.json)

**Core:**
- @angular/animations
- @angular/common
- @angular/compiler
- @angular/core
- @angular/forms
- @angular/platform-browser
- @angular/platform-browser-dynamic
- @angular/router
- rxjs
- tslib
- zone.js

**UI:**
- tailwindcss

---

### Database Setup

```bash
# start postgresql
brew services start postgresql  # mac
sudo systemctl start postgresql  # linux

# create database
createdb transpara_db

# or use psql
psql postgres
CREATE DATABASE transpara_db;
CREATE USER transpara WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE transpara_db TO transpara;
\q

# run migrations
cd backend
source venv/bin/activate
flask db upgrade
```

---

### Redis Setup

```bash
# start redis
brew services start redis  # mac
sudo systemctl start redis  # linux

# test it
redis-cli ping
# should return: PONG
```

---

### Environment Configuration

```bash
cd backend

# copy example env file
cp .env.example .env

# edit .env with your settings
nano .env
```

Minimum required settings:
```bash
SECRET_KEY=your-secret-key-here

DATABASE_URL=postgresql://transpara:password@localhost:5432/transpara_db
REDIS_URL=redis://localhost:6379/0
CORS_ORIGINS=http://localhost:4200
```

---

### Copy Model File

```bash
# copy your trained model
cp bias_detector_model.pkl backend/models/
```

---

### Run the Application

**Backend:**
```bash
cd backend
source venv/bin/activate  # or venv\Scripts\activate on windows
python wsgi.py
```

Backend runs on http://localhost:5000

**Frontend:**
```bash
cd frontend
npm start
```

Frontend runs on http://localhost:4200

---

## Verification

### Check installations:
```bash
# python
python --version  # 3.11+

# node
node --version  # 21+

# npm
npm --version

# postgres
psql --version  # 15+

# redis
redis-cli --version  # 7+

# ffmpeg
ffmpeg -version
```

### Test backend:
```bash
curl http://localhost:5000/health
# should return: {"status":"healthy"}
```

### Test frontend:
Open http://localhost:4200 in browser

---

## Troubleshooting

### "command not found: python3"
```bash
# install python from python.org
# or use package manager
brew install python@3.11  # mac
sudo apt install python3.11  # ubuntu
```

### "pg_config not found"
```bash
# install postgres development headers
sudo apt install libpq-dev  # ubuntu
brew install postgresql  # mac
```

### "redis connection refused"
```bash
# start redis
redis-server
# or
brew services start redis
sudo systemctl start redis
```

### "module not found" errors
```bash
# make sure virtual environment is activated
source venv/bin/activate  # mac/linux
venv\Scripts\activate  # windows

# reinstall dependencies
pip install -r requirements.txt
```

### FFmpeg not found
```bash
# install ffmpeg
brew install ffmpeg  # mac
sudo apt install ffmpeg  # ubuntu
choco install ffmpeg  # windows (with chocolatey)
```

---

## Quick Command Reference

```bash
# backend
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python wsgi.py

# frontend
cd frontend
npm install
npm start

# database
createdb transpara_db
cd backend && flask db upgrade

# redis
redis-server  # or brew services start redis
```

---

## Docker Installation (Easiest)

Don't want to install all this manually? Use Docker:

```bash
docker-compose up -d
```

Docker includes everything:
- ✅ Python 3.11
- ✅ Node.js 21
- ✅ PostgreSQL 15
- ✅ Redis 7
- ✅ FFmpeg
- ✅ All dependencies

No manual setup needed!
