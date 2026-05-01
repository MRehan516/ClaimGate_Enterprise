import numpy as np
from sklearn.neural_network import MLPClassifier

class IntentRouter:
    def __init__(self):
        # A wide, shallow network: 1 hidden layer with 1024 neurons (WideMLP architecture)
        # We use 'log_loss' internally via MLPClassifier for probability calibration
        self.classifier = MLPClassifier(hidden_layer_sizes=(1024,), activation='relu', max_iter=500, random_state=42)
        
        # The deterministic kill-switch categories
        self.sensitive_intents = {'Billing', 'Fraud', 'Permissions', 'Account Access'}
        
    def fit(self, embeddings: np.ndarray, labels: list[str]):
        """
        Trains the WideMLP on the known intent taxonomy.
        """
        self.classifier.fit(embeddings, labels)
        
    def route_query(self, query_embedding: np.ndarray) -> str:
        """
        Evaluates the embedding and deterministically routes it.
        """
        # 1. Predict the specific intent class
        intent = self.classifier.predict([query_embedding])[0]
        
        # 2. Extract the confidence score of the prediction
        probabilities = self.classifier.predict_proba([query_embedding])[0]
        confidence = float(np.max(probabilities))
        
        print(f"[L2 GATE] Intent: '{intent}' | Confidence: {confidence:.2f}")
        
        # 3. Deterministic Gating Logic
        if intent in self.sensitive_intents:
            return f"ESCALATE: High-risk intent '{intent}' requires human authorization."
            
        if confidence < 0.75:
            return "ESCALATE: Intent ambiguity detected. Cannot safely resolve."
            
        return "PROCEED"