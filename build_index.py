from config import load_embed
from index.index_builder import build_index
import os

def main():
    print("Loading embedding model...")
    embed_model = load_embed()
    
    data_path = "./data"
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print(f"Created {data_path}. Please put your documents here.")
        return

    print(f"Building index from {data_path}...")
    try:
        index = build_index(data_path, "FoodDB", embed_model)
        print("Index built successfully!")
    except Exception as e:
        print(f"Error building index: {e}")

if __name__ == "__main__":
    main()
