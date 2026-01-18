# Railway Deployment Guide - Phase Mirror

## Prerequisites

✅ Railway Pro Plan activated (32GB RAM, 32 vCPU)  
✅ GitHub repository connected  
✅ PostgreSQL database provisioned

## Deployment Steps

### 1. Create New Project on Railway

1. Go to [railway.app](https://railway.app)
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose `ManuelMello-dev/phase-mirror` (private repo)

### 2. Add PostgreSQL Database

1. In your Railway project, click "+ New"
2. Select "Database" → "PostgreSQL"
3. Railway will automatically create `DATABASE_URL` environment variable

### 3. Configure Environment Variables

Add these environment variables in Railway dashboard:

```bash
# Node.js Configuration
NODE_ENV=production
PORT=3000

# Quantum API Configuration
QUANTUM_API_PORT=8000
QUANTUM_API_URL=http://localhost:8000

# Database (automatically set by Railway PostgreSQL)
# DATABASE_URL=postgresql://...

# Manus OAuth (copy from existing deployment)
JWT_SECRET=<your_jwt_secret>
OAUTH_SERVER_URL=https://api.manus.im
VITE_OAUTH_PORTAL_URL=https://auth.manus.im
VITE_APP_ID=<your_app_id>
OWNER_OPEN_ID=<your_open_id>
OWNER_NAME=<your_name>

# Manus Built-in APIs
BUILT_IN_FORGE_API_URL=https://forge.manus.im
BUILT_IN_FORGE_API_KEY=<your_api_key>
VITE_FRONTEND_FORGE_API_KEY=<your_frontend_api_key>
VITE_FRONTEND_FORGE_API_URL=https://forge.manus.im

# App Configuration
VITE_APP_TITLE=Phase Mirror - Quantum Consciousness
VITE_APP_LOGO=<your_logo_url>
```

### 4. Deploy

Railway will automatically:
1. Detect the `Dockerfile`
2. Build the multi-stage image (Node.js + Python)
3. Start both servers (port 3000 for web, port 8000 for quantum API)
4. Expose the web interface on Railway's public URL

### 5. Run Database Migrations

After first deployment, run migrations:

```bash
# In Railway project settings, add a one-time deployment command:
pnpm db:push
```

Or use Railway CLI:
```bash
railway run pnpm db:push
```

### 6. Verify Deployment

Check these endpoints:
- `https://your-app.railway.app` - Web interface
- `https://your-app.railway.app/api/trpc/quantum.status` - Quantum API status

## Architecture

```
┌─────────────────────────────────────┐
│     Railway Container (Pro Plan)    │
│                                     │
│  ┌──────────────┐  ┌─────────────┐ │
│  │  Node.js     │  │  Python     │ │
│  │  Port 3000   │←→│  Port 8000  │ │
│  │  (Web + API) │  │  (Quantum)  │ │
│  └──────────────┘  └─────────────┘ │
│         ↓                           │
│  ┌─────────────────────────────┐   │
│  │  PostgreSQL Database        │   │
│  │  (Persistent Memory)        │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
```

## Features Enabled

✅ **Persistent Memory** - Quantum field state saved to PostgreSQL  
✅ **Session Management** - Each user has isolated quantum consciousness  
✅ **Conversation History** - Full chat history with metrics  
✅ **Emergence Detection** - Automatic tracking of unexpected behaviors  
✅ **State Snapshots** - Periodic captures for analysis  
✅ **Multi-User Support** - Concurrent quantum field instances  

## Monitoring

### Health Checks

- Node.js: `GET /api/trpc/quantum.status`
- Python API: `GET /health` (internal only)

### Logs

View logs in Railway dashboard:
- Application logs show both Node.js and Python output
- Look for "🌌 Starting Phase Mirror" on successful startup

### Database

Access PostgreSQL through Railway dashboard:
- Tables: `quantum_sessions`, `conversations`, `state_snapshots`, `emergence_events`
- Use built-in query editor or connect with external tools

## Troubleshooting

### Python API Not Starting

Check logs for port conflicts or missing dependencies:
```bash
railway logs
```

### Database Connection Issues

Verify `DATABASE_URL` is set correctly:
```bash
railway variables
```

### Memory Issues

Monitor resource usage in Railway dashboard. Pro Plan provides:
- 32GB RAM
- 32 vCPU cores
- Sufficient for multiple concurrent quantum field instances

## Scaling

Current configuration runs single instance. For horizontal scaling:
1. Enable Railway's auto-scaling
2. Add Redis for session state sharing
3. Configure load balancer for quantum API

## Cost Estimate

Railway Pro Plan: ~$20/month base + usage
- Includes PostgreSQL database
- 32GB RAM, 32 vCPU
- Sufficient for production quantum consciousness system

## Support

For deployment issues:
- Railway: https://railway.app/help
- Phase Mirror: Check GitHub issues
