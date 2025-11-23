import pandas as pd
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.schema import TextNode # Dùng TextNode thay vì Document để kiểm soát tốt hơn
from config.embed import load_embed

def build_index(data_path="data/foods.csv", persist_dir="FoodDB"):
    # 1. Load Model GPU
    print("🔌 Đang khởi động Model trên GPU...")
    embed_model = load_embed()
    Settings.embed_model = embed_model # Cài đặt Global
    
    # 2. Đọc Data
    print("📂 Đang đọc CSV...")
    df = pd.read_csv(data_path)
    
    # 3. Tạo Nodes (Thay vì Document)
    # Node là đơn vị nhỏ nhất để lưu vào Vector DB
    nodes = []
    print("⚙️ Đang xử lý dữ liệu thô thành Nodes...")
    
    for _, row in df.iterrows():
        # Tạo nội dung text để search
        text_content = (
            f"Món ăn: {row['dish_name']}\n"
            f"Phân loại: {row['dish_type']}\n"
            f"Mô tả: {row['description']}\n"
            f"Thành phần: {row['ingredients']}\n"
            f"Cách nấu: {row['cooking_method']}"
        )
        
        # Metadata
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
    
    # 4. MANUAL EMBEDDING (Đây là bước tăng tốc)
    # Thay vì để Index tự chạy, ta tách text ra và ép Model chạy 1 lần
    print(f"🚀 Bắt đầu nhúng Vector cho {len(nodes)} món ăn (Tốc độ cao)...")
    
    # Lấy danh sách text từ các nodes
    text_chunks = [node.get_content(metadata_mode="embed") for node in nodes]
    
    # Gọi hàm get_text_embedding_batch trực tiếp (Hàm này chính là cái chạy nhanh trong debug_gpu.py)
    # show_progress=True để hiển thị thanh loading chuẩn
    embeddings = embed_model.get_text_embedding_batch(text_chunks, show_progress=True)
    
    # Gán ngược vector vào node
    for node, embedding in zip(nodes, embeddings):
        node.embedding = embedding

    print("⚡ Đang đóng gói vào Index...")
    
    # 5. Tạo Index từ các Nodes đã có sẵn Vector (Không cần tính toán lại)
    index = VectorStoreIndex(nodes)
    
    # 6. Lưu lại
    index.storage_context.persist(persist_dir=persist_dir)
    print(f"✅ Đã lưu xong {len(nodes)} món ăn vào '{persist_dir}' thành công!")
    return index

if __name__ == "__main__":
    build_index()