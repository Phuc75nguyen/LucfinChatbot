import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

try:
    from config.rerank import load_reranker
    print("Import successful.")
    
    model = load_reranker()
    print(f"Model loaded: {model}")
    print(f"Device: {model.device}")
    
    # Simple test
    query = "Test query"
    doc = "Test document"
    score = model.predict([(query, doc)])
    print(f"Test prediction score: {score}")

except Exception as e:
    print(f"Error: {e}")
