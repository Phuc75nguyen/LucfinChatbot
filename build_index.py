from config import load_embed
from index.index_builder import build_index
import os

def main():
    print("Loading embedding model...")
    embed_model = load_embed()
    
    csv_path = os.path.join("data", "foods.csv")
    if not os.path.exists(csv_path):
        print(f"Error: Data file not found at {csv_path}")
        return

    print(f"Building index from {csv_path}...")
    try:
        index = build_index(csv_path, "FoodDB", embed_model)
        print("Index built successfully!")
    except Exception as e:
        print(f"Error building index: {e}")

if __name__ == "__main__":
    main()
