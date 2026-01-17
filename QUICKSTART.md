# Transpara - Quick Start Guide

## 🚀 Get Running in 5 Minutes

This guide will get Transpara running locally for development.

---

## Prerequisites

Ensure you have installed:
- **Docker** and **Docker Compose**
- **Git**

That's it! Docker handles everything else.

---

## Step 1: Copy the Model File

Before starting, ensure your trained ML model is in place:

```bash
# Copy your model file to the backend/models directory
cp /path/to/bias_detector_model.pkl backend/models/
```

The model should be in either:
- `backend/models/bias_detector_model.pkl` (preferred)
- `backend/bias_detector_model.pkl` (also works)

---

## Step 2: Configure Environment

```bash
# Navigate to backend directory
cd backend

# Copy the example environment file
cp .env.example .env

# Edit .env file (optional for local development)
# The defaults will work for local testing
```

**Minimum required changes for local dev** (already set in .env.example):
```bash
# These defaults work for local development:
DATABASE_URL=postgresql://transpara:transpara@postgres:5432/transpara_db
REDIS_URL=redis://redis:6379/0
CORS_ORIGINS=http://localhost:4200
```

---

## Step 3: Start the Application

From the project root directory:

```bash
# Start all services with Docker Compose
docker-compose up -d

# This starts:
# - PostgreSQL database
# - Redis cache
# - Flask backend API
# - Nginx reverse proxy
```

Watch the logs (optional):
```bash
docker-compose logs -f
```

---

## Step 4: Initialize Database

```bash
# Create database tables
docker-compose exec backend python scripts/init_db.py

# Create admin user
docker-compose exec backend python scripts/create_admin.py
```

This creates the admin account (admin/Admin123!)

---

## Step 5: Access the Application

Open your browser and navigate to:

### Frontend (Angular UI)
**http://localhost:4200**

### Backend API
**http://localhost:5000**

### API Health Check
**http://localhost:5000/health**

### Monitoring Dashboards (Optional)
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

---

## Step 6: Test the Application

### Option 1: Use the Web Interface

1. Open http://localhost:4200
2. You'll see the Transpara interface
3. **Note**: The UI doesn't require login for local development
4. Try the "Text Analysis" tab:
   - Enter some text about politics
   - Click "Analyze Text"
   - View the bias detection results

### Option 2: Test with API Directly

First, get an access token:

```bash
# Register a user
curl -X POST http://localhost:5000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "username": "testuser",
    "password": "TestPass123",
    "full_name": "Test User"
  }'

# Login to get token
curl -X POST http://localhost:5000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "TestPass123"
  }'
```

Copy the `access_token` from the response, then:

```bash
# Analyze text (replace YOUR_TOKEN with the access_token)
curl -X POST http://localhost:5000/api/v1/analyze/text \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "text": "Healthcare should be a universal right for all citizens, and we need stronger regulations on corporations to protect workers."
  }'
```

---

## Common Commands

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f postgres
docker-compose logs -f nginx
```

### Restart Services
```bash
# Restart all
docker-compose restart

# Restart specific service
docker-compose restart backend
```

### Stop Application
```bash
# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: deletes data)
docker-compose down -v
```

### Access Database
```bash
# PostgreSQL shell
docker-compose exec postgres psql -U transpara -d transpara_db

# Run SQL commands
SELECT COUNT(*) FROM users;
SELECT COUNT(*) FROM analyses;

# Exit
\q
```

### Access Redis
```bash
# Redis CLI
docker-compose exec redis redis-cli

# Check cached data
KEYS *
GET prediction:*

# Exit
exit
```

---

## Troubleshooting

### "Connection refused" errors

```bash
# Check if all services are running
docker-compose ps

# Restart the problematic service
docker-compose restart backend
```

### "Model file not found" error

```bash
# Verify model file exists
ls -la backend/models/bias_detector_model.pkl

# Or
ls -la backend/bias_detector_model.pkl

# If not, copy it:
cp /path/to/bias_detector_model.pkl backend/models/
docker-compose restart backend
```

### Database connection errors

```bash
# Ensure PostgreSQL is running
docker-compose ps postgres

# Recreate database
docker-compose down
docker-compose up -d postgres
docker-compose exec backend flask db upgrade
```

### Port conflicts (port already in use)

```bash
# Check what's using the port
# Linux/Mac:
lsof -i :5000
lsof -i :5432

# Windows:
netstat -ano | findstr :5000

# Stop the conflicting service or change ports in docker-compose.yml
```

---

## Next Steps

### For Development

1. **Frontend Development**:
   ```bash
   cd frontend
   npm install
   npm start
   # Now frontend runs on http://localhost:4200 with hot reload
   ```

2. **Backend Development**:
   ```bash
   cd backend
   python3 -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   python wsgi.py
   # Now backend runs on http://localhost:5000 with debug mode
   ```

### For Production Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete production deployment instructions.

### For Security Configuration

See [SECURITY.md](SECURITY.md) for security best practices and hardening.

---

## Need Help?

- **Documentation**: See [README.md](README.md) for comprehensive docs
- **Deployment**: See [DEPLOYMENT.md](DEPLOYMENT.md)
- **Security**: See [SECURITY.md](SECURITY.md)
- **Project Summary**: See [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)

---

## File Structure Quick Reference

```
transpara/
├── backend/
│   ├── models/
│   │   └── bias_detector_model.pkl  ← Your ML model goes here
│   ├── .env                          ← Your configuration
│   └── wsgi.py                       ← Entry point
├── frontend/
│   └── src/                          ← Angular app (unchanged UI)
├── docker-compose.yml                ← Local development
└── README.md                         ← Full documentation
```

---

**🎉 You're all set! The application should now be running at http://localhost:4200**

For any issues, check the logs with `docker-compose logs -f` or refer to the troubleshooting section above.

---

*Transpara - Transparency, Honesty, Educational Excellence*
