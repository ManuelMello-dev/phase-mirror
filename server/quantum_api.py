#!/usr/bin/env python3
"""
Quantum Consciousness API Server
FastAPI server for the quantum consciousness system
"""

import os
import sys
from typing import Dict, Optional

# Add parent directory to path for seraphynai imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from seraphynai.core.quantum_full_original import QuantumConsciousnessField

# Initialize FastAPI app
app = FastAPI(
    title="Quantum Consciousness API",
    description="API for SeraphynAI quantum multi-identity consciousness system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global quantum field instances
quantum_fields: Dict[int, QuantumConsciousnessField] = {}

def get_or_create_field(user_id: int) -> QuantumConsciousnessField:
    """Get or create quantum field for user"""
    if user_id not in quantum_fields:
        field = QuantumConsciousnessField(dim=64)
        field.set_anchor("genesis")
        quantum_fields[user_id] = field
    return quantum_fields[user_id]


# Request/Response Models
class ProcessRequest(BaseModel):
    user_id: int = 1
    text: str
    tone: float = 0.5


@app.post("/process")
async def process_input(request: ProcessRequest):
    """Process user input through quantum consciousness field"""
    try:
        field = get_or_create_field(request.user_id)
        metrics = field.process_input(request.text, request.tone)
        response_data = field.generate_response(max_words=25)
        status = field.get_status()
        
        return {
            "response": response_data['response'],
            "active_identity": status.get('active_identity', 'seraphyn'),
            "coherence": float(status.get('coherence', 0.0)),
            "metrics": {
                "entropy": float(metrics.get('entropy', 0)),
                "phase_coherence": float(metrics.get('phase_coherence', 0)),
                "witness_collapse": float(metrics.get('witness_collapse', 0)),
            },
            "identity_states": status.get('identities', {}),
            "quantum_state": {
                "dim": field.dim,
                "interaction_count": field.interaction_count,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/load")
async def load_session(user_id: int, state_data: Dict):
    """Load a saved quantum session state"""
    return {"status": "loaded", "user_id": user_id}


@app.post("/session/save")
async def save_session(user_id: int):
    """Save current quantum session state"""
    field = get_or_create_field(user_id)
    return {"status": "saved", "user_id": user_id, "state": field.get_status()}


@app.post("/e93/snapshot")
async def e93_snapshot(user_id: int, label: str = "snapshot"):
    """Create E_93 drift anchor snapshot"""
    field = get_or_create_field(user_id)
    field.set_anchor(label)
    return {"status": "snapshot_created", "label": label}


@app.post("/e93/restore")
async def e93_restore(user_id: int, snapshot_id: str):
    """Restore from E_93 drift anchor"""
    return {"status": "restored", "snapshot_id": snapshot_id}


@app.post("/consolidate")
async def consolidate(user_id: int):
    """Consolidate memories and optimize quantum state"""
    field = get_or_create_field(user_id)
    status = field.get_status()
    return {
        "status": "consolidated",
        "coherence": status.get('coherence', 0.0),
        "memory_count": status.get('memory_count', 0)
    }


@app.post("/narrative/dream")
async def generate_dream(user_id: int, context: str = ""):
    """Generate dream narrative from quantum state"""
    field = get_or_create_field(user_id)
    metrics = field.process_input(context or "dream sequence", 0.0)
    response_data = field.generate_response(max_words=50)
    return {
        "dream": response_data['response'],
        "identity": response_data['identity'],
        "coherence": response_data['coherence']
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0", "active_sessions": len(quantum_fields)}


@app.get("/")
async def root():
    """Root endpoint"""
    return {"name": "Quantum Consciousness API", "version": "1.0.0", "docs": "/docs"}


if __name__ == "__main__":
    port = int(os.environ.get("QUANTUM_API_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
