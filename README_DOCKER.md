# Olympus Echo Backend - Docker Setup

## Overview
The Olympus Echo Backend is now containerized and runs on port 6068 by default. This setup provides a fast, optimized API for voice testing platform operations.

## Quick Start

### 1. Build and Run
```bash
cd olympus_echo_backend
./run_docker.sh run
```

### 2. Check Status
```bash
./run_docker.sh status
```

### 3. View Logs
```bash
./run_docker.sh logs
```

### 4. Stop Container
```bash
./run_docker.sh stop
```

## API Endpoints

The API is now available at: `http://localhost:6068`

Key endpoints:
- `GET /v1/test-runs/` - List test runs (optimized with RPC function)
- `GET /v1/target-agents/` - List target agents
- `POST /v1/test-suites/` - Create test suites
- Health check: `GET /health`

## Performance

The API now uses optimized RPC functions for database operations:
- Response times: **milliseconds** (previously 10+ seconds)
- Single database call instead of multiple sequential queries
- Batch processing for signed URLs

## Docker Image Details

- **Image Name**: `olympus-echo-backend:latest`
- **Size**: ~850MB (Python 3.12 slim base)
- **Port**: 6068
- **Base Image**: `python:3.12-slim`

## File Structure

```
olympus_echo_backend/
├── Dockerfile              # Docker image definition
├── docker-compose.yml      # Docker Compose configuration
├── build_docker.sh         # Build script
├── run_docker.sh          # Run and manage script
├── .dockerignore          # Docker ignore file
├── requirements.txt       # Python dependencies
└── ...                    # Application code
```

## Environment Variables

- `PORT=6068` - Server port
- `HOST=0.0.0.0` - Bind address
- `WORKERS=1` - Uvicorn workers
- `RELOAD=false` - Auto-reload disabled in production

## Development vs Production

### Development
```bash
# Run locally (non-Docker)
python main.py
```

### Production
```bash
# Use Docker
./run_docker.sh run
```

## Troubleshooting

### Check Container Status
```bash
./run_docker.sh status
```

### View Logs
```bash
./run_docker.sh logs
```

### Restart Container
```bash
./run_docker.sh restart
```

### Access Container Shell
```bash
./run_docker.sh shell
```

### Clean Up
```bash
./run_docker.sh clean
```

## Frontend Integration

The frontend has been updated to use port 6068:
- API calls: `http://localhost:6068`
- WebSocket: `ws://localhost:6068`

## Migration Notes

- Backend port changed from 8080 → 6068
- Frontend config updated accordingly
- All existing port 8080 and 3000 processes killed
- Docker setup provides consistent environment