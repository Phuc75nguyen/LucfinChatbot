import pandas as pd
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.schema import TextNode # <--- Code mới dùng TextNode
from config.embed import load_embed

def build_index(data_path="data_raw/foods.csv", persist_dir="FoodDB"):
    # 1. Load Model GPU
    print("🔌 Đang khởi động Model trên GPU...")
    embed_model = load_embed()
    Settings.embed_model = embed_model
    
    # 2. Đọc Data
    print("📂 Đang đọc CSV...")
    df = pd.read_csv(data_path)
    
    # 3. Tạo Nodes thủ công (Manual Node Creation)
    nodes = []
    print("⚙️ Đang chuyển đổi dữ liệu sang Nodes...")
    
    for _, row in df.iterrows():
        # Text để search
        text_content = (
            f"Món ăn: {row['dish_name']}\n"
            f"Phân loại: {row['dish_type']}\n"
            f"Mô tả: {row['description']}\n"
            f"Thành phần: {row['ingredients']}\n"
            f"Cách nấu: {row['cooking_method']}"
        )
        
        # Metadata hiển thị
        metadata = {
            "dish_name": str(row['dish_name']),
            "calories": int(row['calories']) if pd.notna(row['calories']) else 0,
            "protein": int(row['protein']) if pd.notna(row['protein']) else 0,
            "fat": int(row['fat']) if pd.notna(row['fat']) else 0,
            "image_link": str(row['image_link']) if pd.notna(row['image_link']) else ""
        }
        
        # Tạo Node
        node = TextNode(text=text_content, metadata=metadata)
        nodes.append(node)
    
    # 4. MANUAL EMBEDDING (BƯỚC QUAN TRỌNG NHẤT)
    # Tự tay nhúng vector, bỏ qua cơ chế chậm chạp mặc định của LlamaIndex
    print(f"🚀 Đang kích hoạt GPU nhúng vector cho {len(nodes)} món ăn...")
    
    # Lấy text ra
    text_chunks = [node.get_content(metadata_mode="embed") for node in nodes]
    
    # Ép Model chạy batching 64
    # Hàm này chạy cực nhanh (giống file debug_gpu.py)
    embeddings = embed_model.get_text_embedding_batch(text_chunks, show_progress=True)
    
    # Gán vector ngược lại vào node
    for node, embedding in zip(nodes, embeddings):
        node.embedding = embedding

    print("⚡ Đang đóng gói vào Index...")
    
    # 5. Tạo Index từ Nodes đã có Vector (cực nhanh vì không cần tính toán nữa)
    index = VectorStoreIndex(nodes)
    
    # 6. Lưu lại
    index.storage_context.persist(persist_dir=persist_dir)
    print(f"✅ Đã XONG! Lưu dữ liệu vào '{persist_dir}'.")
    return index

if __name__ == "__main__":
    build_index()