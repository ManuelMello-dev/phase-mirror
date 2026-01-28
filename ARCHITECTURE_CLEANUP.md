# Architecture Cleanup - Unified Quantum Engine

## Changes Made

### 1. Centralized Quantum Engine ✅
**Single source of truth:** `/seraphynai/core/quantum_full_original.py` (1,217 lines)

All imports now point to this canonical location:
- `quantum_bridge.py` → imports from `seraphynai.core.quantum_full_original`
- `server/quantum_bridge.py` → imports from `seraphynai.core.quantum_full_original`
- `server/quantum_api.py` → NEW FastAPI server, imports from `seraphynai.core.quantum_full_original`

### 2. Removed Duplicates ✅
Moved to `.backup` files (can be deleted):
- `quantum_engine.py` → `quantum_engine.py.backup`
- `server/quantum_engine.py` → `server/quantum_engine.py.backup`

### 3. Created Proper API Server ✅
**File:** `server/quantum_api.py`
- FastAPI-based HTTP server
- Runs on port 8000
- Implements all required endpoints:
  - `POST /process` - Main quantum processing
  - `POST /session/load` - Load quantum state
  - `POST /session/save` - Save quantum state
  - `POST /e93/snapshot` - Create drift anchor
  - `POST /e93/restore` - Restore from anchor
  - `POST /consolidate` - Memory consolidation
  - `POST /narrative/dream` - Dream generation
  - `GET /health` - Health check
  - `GET /` - API info

### 4. Fixed Lazy Imports ✅
Updated `seraphynai/__init__.py` and `seraphynai/core/__init__.py` with lazy loading to avoid import errors when numpy/scipy aren't installed.

## New Architecture Flow

```
┌─────────────────────────────────────────────────┐
│          TypeScript/React Frontend              │
│         (client/src/pages/QuantumChat.tsx)      │
└────────────────┬────────────────────────────────┘
                 │ HTTP/WebSocket
                 ↓
┌─────────────────────────────────────────────────┐
│        TypeScript tRPC Router                   │
│        (server/quantum-router.ts)               │
│        Calls → http://localhost:8000/process    │
└────────────────┬────────────────────────────────┘
                 │ HTTP POST
                 ↓
┌─────────────────────────────────────────────────┐
│         Python FastAPI Server                   │
│         (server/quantum_api.py)                 │
│         Port 8000                               │
└────────────────┬────────────────────────────────┘
                 │ Direct import
                 ↓
┌─────────────────────────────────────────────────┐
│      CANONICAL QUANTUM ENGINE                   │
│   seraphynai/core/quantum_full_original.py      │
│   - 5 identity personas                         │
│   - Z³ evolution                                │
│   - Quantum memory                              │
│   - E_93 anchors                                │
│   - Emergent voice                              │
└─────────────────────────────────────────────────┘
```

## Simplified Stack

**Before:**
- 3+ copies of quantum_engine.py (root, server, seraphynai/core)
- Unclear import paths
- Subprocess calls to Python CLI
- No proper API server

**After:**
- 1 canonical quantum engine (`seraphynai/core/quantum_full_original.py`)
- Clear import paths (all point to seraphynai.core)
- Proper FastAPI server (`server/quantum_api.py`)
- Clean REST API interface

## Deployment

### Development
```bash
# Terminal 1: Start Python API
cd /home/runner/work/phase-mirror/phase-mirror
python3 server/quantum_api.py

# Terminal 2: Start Node.js  
pnpm run dev
```

### Production (Railway)
The `Dockerfile` and `start-servers.sh` will:
1. Install Python dependencies (numpy, scipy, fastapi, uvicorn)
2. Start `server/quantum_api.py` on port 8000
3. Start Node.js server on port 3000
4. Both servers use `/seraphynai/core/quantum_full_original.py`

## Testing

With numpy/scipy installed:
```bash
# Test direct import
python3 -c "from seraphynai.core.quantum_full_original import QuantumConsciousnessField; print('✓ Works')"

# Test API server
python3 server/quantum_api.py &
curl http://localhost:8000/health

# Test full integration
python3 tests/test_quantum_full_integration.py
```

## What's Left to Clean

After verifying everything works:
1. Delete `.backup` files:
   - `quantum_engine.py.backup`
   - `server/quantum_engine.py.backup`

2. Optional: Remove old quantum_bridge.py CLI interface if API is sufficient

## Key Benefits

1. **Single source of truth** - Only one quantum engine implementation
2. **Clear architecture** - Python core → FastAPI → TypeScript → React
3. **Proper API** - RESTful endpoints with FastAPI (not subprocess hacks)
4. **Maintainable** - Changes to quantum engine happen in one place
5. **Testable** - Can test each layer independently

---

**Architecture Status:** ✅ SIMPLIFIED AND UNIFIED

The quantum consciousness system now has a clear, single-path architecture from the UI down to the quantum engine core.
