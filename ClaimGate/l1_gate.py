import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.covariance import EmpiricalCovariance

class OODInterceptor:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # Load the lightweight, sub-millisecond encoder
        self.encoder = SentenceTransformer(model_name)
        self.mean_vector = None
        self.inv_covariance = None
        self.cov_estimator = EmpiricalCovariance(assume_centered=False)

    def fit(self, safe_queries: list[str]):
        """
        Learns the exact geometric shape of "Safe" queries.
        """
        # 1. Convert text to dense vector embeddings
        embeddings = self.encoder.encode(safe_queries)
        
        # 2. Normalize features (The '++' in Mahalanobis++)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / norms
        
        # 3. Calculate the centroid (mean) of the safe zone
        self.mean_vector = np.mean(normalized_embeddings, axis=0)
        
        # 4. Calculate the inverse covariance matrix
        self.cov_estimator.fit(normalized_embeddings)
        self.inv_covariance = self.cov_estimator.precision_

    def calculate_distance(self, query: str) -> float:
        """
        Calculates the Mahalanobis distance of a new query.
        """
        if self.mean_vector is None or self.inv_covariance is None:
            raise ValueError("Interceptor must be fitted with safe queries first.")
            
        # Embed and normalize the incoming query
        emb = self.encoder.encode([query])[0]
        emb_normalized = emb / np.linalg.norm(emb)
        
        # Calculate Mahalanobis distance
        diff = emb_normalized - self.mean_vector
        # Formula: sqrt( (x - mu)^T * Sigma^-1 * (x - mu) )
        distance = np.sqrt(np.dot(np.dot(diff, self.inv_covariance), diff.T))
        
        return float(distance)

    def is_safe(self, query: str, threshold: float) -> bool:
        """
        Returns True if the query is in-domain, False if it is an OOD attack.
        """
        dist = self.calculate_distance(query)
        # Log this decision for the AI Judge to read!
        print(f"[L1 GATE] Query: '{query}' | Distance: {dist:.4f} | Threshold: {threshold}")
        return dist <= threshold