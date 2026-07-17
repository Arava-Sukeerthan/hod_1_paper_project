import math
from typing import List, Dict, Any

def calculate_confidence(retrieved_documents: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Computes a normalized confidence score based on the average CrossEncoder score
    of the retrieved evidence documents.
    
    A sigmoid activation function (1 / (1 + exp(-x))) is applied to normalize the
    unbounded raw CrossEncoder scores (typically ranging from -10 to +10) into a 
    standard [0, 1] probability range.
    
    Args:
        retrieved_documents: A list of candidate evidence chunks containing a 'score' key.
        
    Returns:
        A dictionary in the format {"confidence": float} representing the system confidence.
    """
    if not retrieved_documents:
        return {"confidence": 0.0}
        
    # Extract CrossEncoder scores from documents
    scores = [doc.get("score") for doc in retrieved_documents if doc.get("score") is not None]
    if not scores:
        return {"confidence": 0.0}
        
    # Compute the average score
    avg_score = sum(scores) / len(scores)
    
    # Map raw CrossEncoder logits to a 0-1 range using sigmoid activation
    # Standard sigmoid: 1 / (1 + exp(-x))
    try:
        confidence_val = 1.0 / (1.0 + math.exp(-avg_score))
    except OverflowError:
        # Handle mathematical overflow boundaries
        confidence_val = 0.0 if avg_score < 0 else 1.0
        
    # Round to 2 decimal places for scientific precision/readability
    return {"confidence": round(confidence_val, 2)}
