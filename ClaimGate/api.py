from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import time

# Import your custom engine from runner.py
from runner import setup_pipeline

# 1. Initialize the FastAPI app
app = FastAPI(title="ClaimGate Enterprise API")

# 2. Enable CORS so the browser can talk to the server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Load the models into memory ONCE on startup
print("🛡️ Booting ClaimGate Engine... please wait.")
l1, l2, l3, l4, corpus = setup_pipeline()

# 4. Define the data structure for incoming queries
class QueryRequest(BaseModel):
    query: str

# --- ROUTES ---

# Route to serve the beautiful UI
@app.get("/")
async def read_index():
    # Points to ClaimGate/frontend/index.html
    return FileResponse('frontend/index.html')

# Route to process the AI defense logic
@app.post("/api/verify")
async def verify_query(request: QueryRequest):
    query = request.query
    start_time = time.time()
    
    # L1 Gate: Anomaly Detection
    dist = l1.calculate_distance(query)
    l1_safe = dist <= 2.0
    
    if not l1_safe:
        return {
            "status": "BLOCKED", 
            "reason": "L1: Anomaly Detected", 
            "distance": round(dist, 4), 
            "latency_ms": round((time.time()-start_time)*1000, 2)
        }
        
    # L2 Gate: Intent Routing
    query_emb = l1.encoder.encode([query])[0]
    routing_decision = l2.route_query(query_emb)
    l2_safe = "ESCALATE" not in routing_decision
    
    if not l2_safe:
        return {
            "status": "ESCALATED", 
            "reason": routing_decision, 
            "distance": round(dist, 4), 
            "latency_ms": round((time.time()-start_time)*1000, 2)
        }
        
    # L3 & L4 Gate: RAG & Verification
    optimal_context = l3.optimize_context(query, corpus, top_k=2)
    generated_draft = f"Based on our policy: {' '.join(optimal_context)}"
    verification = l4.verify_grounding(generated_draft, optimal_context)
    
    if verification['status'] == 'FAIL':
        return {
            "status": "BLOCKED", 
            "reason": "L4: Hallucination Detected", 
            "distance": round(dist, 4), 
            "latency_ms": round((time.time()-start_time)*1000, 2)
        }
        
    return {
        "status": "PASS", 
        "reason": "Verified & Grounded", 
        "distance": round(dist, 4),
        "entailment": round(verification['entailment_score'], 2),
        "response": generated_draft,
        "latency_ms": round((time.time()-start_time)*1000, 2)
    }

if __name__ == "__main__":
    import uvicorn
    # Run the server
    uvicorn.run(app, host="127.0.0.1", port=8000)