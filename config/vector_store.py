import os
from llama_index.core import StorageContext, load_index_from_storage, Settings
from config.embed import load_embed
from config.llm import load_llm

def get_vector_store():
    # 1. Đường dẫn tới DB vừa build
    PERSIST_DIR = "./FoodDB"
    
    if not os.path.exists(PERSIST_DIR):
        raise ValueError(f"❌ Không tìm thấy thư mục '{PERSIST_DIR}'. Hãy chạy build_index.py trước!")

    print(f"📂 Đang tải Vector Database từ: {PERSIST_DIR}")

    # 2. Cấu hình Global (QUAN TRỌNG: Phải khớp với lúc build)
    Settings.embed_model = load_embed()
    Settings.llm = load_llm()

    # 3. Load Index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    
    print("✅ Đã nạp Index thành công!")
    return index