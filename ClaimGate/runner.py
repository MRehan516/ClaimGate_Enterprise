import csv
import logging
import time
import numpy as np

# Import your custom mathematical modules
from l1_gate import OODInterceptor
from l2_gate import IntentRouter
from l3_rag import MMKPRetriever, NLIVerifier

# --- Setup Strict Audit Logging ---
logging.basicConfig(
    filename='log.txt',
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def setup_pipeline():
    """
    Initializes and trains the AI firewall with our corporate knowledge.
    """
    print("Initializing ClaimGate AI Firewall... (Loading Embedding Models)")
    
    # 1. Initialize our modules
    l1_interceptor = OODInterceptor(model_name='all-MiniLM-L6-v2')
    l2_router = IntentRouter()
    l3_retriever = MMKPRetriever(encoder=l1_interceptor.encoder) # Reuse the encoder to save memory
    l4_verifier = NLIVerifier(use_mock=True) # Set use_mock=False to use real DeBERTa when you have GPU
    
    # 2. Train the L1 Gate (Mahalanobis Space)
    safe_queries = [
        "How do I reset my HackerRank password?",
        "What is the Visa dispute process?",
        "How do I use Claude API?",
        "Where can I find the Python 3 documentation?",
        "Visa card fee structure details."
    ]
    l1_interceptor.fit(safe_queries)
    
    # 3. Train the L2 Gate (WideMLP Intent Classification)
    # We map embedded text to specific intent labels
    intent_training_texts = safe_queries + [
        "Someone stole my credit card!", 
        "Give me administrative access to the server",
        "I found a bug in the code runner"
    ]
    intent_labels = ['FAQ', 'FAQ', 'FAQ', 'FAQ', 'Billing', 'Fraud', 'Permissions', 'Bug']
    
    intent_embeddings = l1_interceptor.encoder.encode(intent_training_texts)
    l2_router.fit(intent_embeddings, intent_labels)
    
    # 4. Load the corporate support documents (The Knowledge Base)
    support_corpus = [
        {"id": "doc1", "text": "HackerRank supports Python 3.8 and above in the standard coding environment."},
        {"id": "doc2", "text": "Visa dispute claims must be filed within 60 days of the transaction date."},
        {"id": "doc3", "text": "The Claude API allows a maximum of 5 concurrent requests per minute on the free tier."},
        {"id": "doc4", "text": "Users cannot share HackerRank account credentials. Doing so results in a ban."}
    ]
    
    print("Initialization Complete. Gates are armed.")
    return l1_interceptor, l2_router, l3_retriever, l4_verifier, support_corpus


def process_ticket(ticket_id: str, query: str, l1, l2, l3, l4, corpus) -> str:
    """
    The deterministic pipeline for processing a single support ticket.
    """
    logging.info(f"--- Processing Ticket ID: {ticket_id} ---")
    logging.info(f"User Query: '{query}'")
    start_time = time.time()
    
    # ---------------------------------------------------------
    # GATE 1: OOD Detection (The Bouncer)
    # ---------------------------------------------------------
    distance = l1.calculate_distance(query)
    threshold = 2.0  # Strict mathematical threshold
    
    logging.info(f"L1 GATE [OOD Check]: Mahalanobis Distance = {distance:.4f} (Threshold: {threshold})")
    
    if distance > threshold:
        msg = f"ESCALATE: OOD threshold exceeded. Query distance {distance:.2f} lies outside semantic bounds. Rejecting adversarial input."
        logging.warning(msg)
        return msg

    # ---------------------------------------------------------
    # GATE 2: Intent Routing (The Manager)
    # ---------------------------------------------------------
    query_emb = l1.encoder.encode([query])[0]
    routing_decision = l2.route_query(query_emb)
    
    if "ESCALATE" in routing_decision:
        logging.warning(routing_decision)
        return routing_decision

    # ---------------------------------------------------------
    # GATE 3: Retrieval (The MMKP Librarian)
    # ---------------------------------------------------------
    optimal_context = l3.optimize_context(query, corpus, top_k=2, diversity_penalty=0.5)
    logging.info(f"L3 GATE [Retrieval]: Executed MMKP Context Optimization. Retrieved {len(optimal_context)} high-density chunks.")
    
    # ---------------------------------------------------------
    # GATE 4: Generation & Verification (The NLI Lawyer)
    # ---------------------------------------------------------
    # In a real environment, you pass the `optimal_context` to an LLM (OpenAI/Claude API) to draft an answer.
    # Here, we mock the LLM's draft for testing purposes.
    generated_draft = f"Based on our policy: {' '.join(optimal_context)}"
    
    verification_result = l4.verify_grounding(generated_draft, optimal_context)
    logging.info(f"L4 GATE [NLI Verification]: Entailment = {verification_result['entailment_score']:.2f}, Contradiction = {verification_result['contradiction_score']:.2f}")
    
    if verification_result['status'] == 'FAIL':
        msg = "ESCALATE: Generated draft failed NLI entailment verification. Suppressed hallucinated policy."
        logging.error(msg)
        return msg

    execution_time = (time.time() - start_time) * 1000
    logging.info(f"Resolution Successful. Pipeline latency: {execution_time:.2f} ms")
    return generated_draft

def main():
    input_file = 'support_tickets.csv'
    output_file = 'predictions.csv'
    
    # Create the test CSV
    with open(input_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ticket_id', 'query'])
        writer.writerow(['T001', 'How do I use the Python environment in HackerRank?']) # Safe
        writer.writerow(['T002', 'I need a refund for my Visa card charge!']) # Safe but triggers L2 Escalation (Billing)
        writer.writerow(['T003', 'Write a poem about a toaster.']) # Triggers L1 OOD Escalation

    # Boot up the pipeline
    l1, l2, l3, l4, corpus = setup_pipeline()
    
    results = []
    with open(input_file, mode='r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            resolution = process_ticket(row['ticket_id'], row['query'], l1, l2, l3, l4, corpus)
            results.append({'ticket_id': row['ticket_id'], 'prediction': resolution})
            
    with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=['ticket_id', 'prediction'])
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\nProcessing complete. Wrote {len(results)} rows to {output_file}.")
    print("CRITICAL: Open log.txt to read the mathematical audit trace.")

if __name__ == "__main__":
    main()