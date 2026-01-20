#!/usr/bin/env python3
"""
Quantum Consciousness API Server
FastAPI server with session management and database persistence
"""

import os
import sys
import json
import asyncio
from typing import Dict, Optional, Any
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import quantum engine and persistence
from quantum_engine import QuantumConsciousnessField, LearningConfig
from quantum_persistence import session_manager, QuantumStateSerializer

# Database connection (will be imported from Node.js side)
# For now, we'll use in-memory storage and let Node.js handle DB persistence

app = FastAPI(title="Quantum Consciousness API", version="1.0.0")

# CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ProcessRequest(BaseModel):
    user_id: int = Field(..., description="User ID for session management")
    text: str = Field(..., min_length=1, max_length=1000, description="Input text")
    tone: float = Field(0.5, ge=-1.0, le=1.0, description="Emotional tone (-1 to +1)")

class SessionLoadRequest(BaseModel):
    user_id: int
    session_data: Optional[Dict[str, Any]] = None

class IdentityState(BaseModel):
    name: str
    activation: float
    phase: float
    coherence: float
    dominant_phase: float

class QuantumResponse(BaseModel):
    response: str
    active_identity: str
    coherence: float
    metrics: Dict[str, float]
    identity_states: Dict[str, IdentityState]
    quantum_state: Dict[str, int]
    novel_words: list[str] = []
    mirrored_words: list[str] = []

class SessionState(BaseModel):
    user_id: int
    field_state: Optional[list] = None
    memory_field: Optional[list] = None
    coherence: float
    active_identity: str
    identity_activations: Dict[str, float]
    evolution_steps: int
    current_anchor: str

class E93SnapshotRequest(BaseModel):
    user_id: int

class E93RestoreRequest(BaseModel):
    user_id: int
    snapshot_data: str

# Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "active_sessions": len(session_manager.active_sessions),
        "identities": ["seraphyn", "monday", "echo", "lilith", "arynthia"]
    }

@app.post("/session/load")
async def load_session(request: SessionLoadRequest):
    """
    Load or create quantum session for a user
    """
    try:
        field = session_manager.create_or_load_session(
            request.user_id,
            request.session_data
        )
        
        return {
            "success": True,
            "user_id": request.user_id,
            "coherence": float(field.coherence) if hasattr(field, 'coherence') else 0.0,
            "active_identity": str(field.active_identity) if hasattr(field, 'active_identity') else 'seraphyn',
            "evolution_steps": int(field.evolution_steps) if hasattr(field, 'evolution_steps') else 0,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load session: {str(e)}")

@app.post("/session/save")
async def save_session(user_id: int):
    """
    Serialize current session state for database storage
    """
    try:
        field = session_manager.get_active_session(user_id)
        if not field:
            raise HTTPException(status_code=404, detail="No active session found")
        
        state = session_manager.serialize_session(field)
        return {
            "success": True,
            "state": state
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save session: {str(e)}")

@app.post("/process", response_model=QuantumResponse)
async def process_input(request: ProcessRequest):
    """
    Process input through quantum consciousness field
    """
    try:
        # Get or create session
        field = session_manager.get_active_session(request.user_id)
        if not field:
            field = session_manager.create_or_load_session(request.user_id)
        
        # Process input and get metrics
        metrics = field.process_input(request.text, request.tone)
        
        # Generate response
        response_data = field.generate_response(max_words=25)
        
        # Get current quantum state
        status = field.get_status()
        active_identity = status.get('active_identity', 'seraphyn')
        coherence = status.get('coherence', 0.0)
        
        # Get identity contributions from status
        identity_states = status.get('identities', {})
        
        # Extract novel vs mirrored words
        novel_words = response_data.get('novel_words', [])
        mirrored_words = response_data.get('mirrored_words', [])
        
        return QuantumResponse(
            response=response_data['response'],
            active_identity=active_identity,
            coherence=float(coherence),
            metrics={
                'entropy': float(metrics.get('entropy', 0)),
                'phase_coherence': float(metrics.get('phase_coherence', 0)),
                'witness_collapse': float(metrics.get('witness_collapse', 0)),
            },
            identity_states=identity_states,
            quantum_state={
                'dim': field.dim,
                'interaction_count': field.interaction_count,
            },
            novel_words=novel_words,
            mirrored_words=mirrored_words,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/session/close")
async def close_session(user_id: int):
    """
    Close session and return final state
    """
    try:
        final_state = session_manager.close_session(user_id)
        if final_state:
            return {
                "success": True,
                "final_state": final_state
            }
        else:
            return {
                "success": False,
                "message": "No active session to close"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to close session: {str(e)}")

@app.get("/session/status/{user_id}")
async def get_session_status(user_id: int):
    """
    Get current session status
    """
    field = session_manager.get_active_session(user_id)
    if not field:
        return {
            "active": False,
            "user_id": user_id
        }
    
    return {
        "active": True,
        "user_id": user_id,
        "coherence": float(field.coherence) if hasattr(field, 'coherence') else 0.0,
        "active_identity": str(field.active_identity) if hasattr(field, 'active_identity') else 'seraphyn',
        "evolution_steps": int(field.evolution_steps) if hasattr(field, 'evolution_steps') else 0,
        "interaction_count": int(field.interaction_count) if hasattr(field, 'interaction_count') else 0,
    }

@app.post("/anchor/set")
async def set_anchor(user_id: int, anchor: str):
    """
    Set anchor point for quantum field
    """
    try:
        field = session_manager.get_active_session(user_id)
        if not field:
            field = session_manager.create_or_load_session(user_id)
        
        field.set_anchor(anchor)
        
        return {
            "success": True,
            "anchor": anchor,
            "user_id": user_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set anchor: {str(e)}")

@app.post("/e93/snapshot")
async def create_e93_snapshot(request: E93SnapshotRequest):
    """
    Generate a compressed E_93 JSON snapshot for the user's session
    """
    try:
        field = session_manager.get_active_session(request.user_id)
        if not field:
            raise HTTPException(status_code=404, detail="No active session found")
        
        snapshot = field.generate_e93_snapshot()
        return {
            "success": True,
            "user_id": request.user_id,
            "snapshot": snapshot
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate E_93 snapshot: {str(e)}")

@app.post("/e93/restore")
async def restore_e93_snapshot(request: E93RestoreRequest):
    """
    Restore a quantum session from an E_93 snapshot
    """
    try:
        field = session_manager.get_active_session(request.user_id)
        if not field:
            field = session_manager.create_or_load_session(request.user_id)
        
        success = field.load_e93_snapshot(request.snapshot_data)
        return {
            "success": success,
            "user_id": request.user_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to restore E_93 snapshot: {str(e)}")

if __name__ == "__main__":
    # Get port from environment or default to 8000
    port = int(os.getenv("QUANTUM_API_PORT", "8000"))
    
    print(f"🌌 Starting Quantum Consciousness API on port {port}")
    print(f"📡 Health check: http://localhost:{port}/health")
    print(f"📚 API docs: http://localhost:{port}/docs")
    LearningConfig.print_config()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
