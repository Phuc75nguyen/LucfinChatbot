import time
import torch
import pandas as pd
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def test_gpu_speed():
    print("="*50)
    print("BẮT ĐẦU CHẨN ĐOÁN GPU & TỐC ĐỘ")
    print("="*50)

    # 1. KIỂM TRA PHẦN CỨNG (HARDWARE CHECK)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ Đã tìm thấy GPU: {gpu_name}")
        print(f"   CUDA Version: {torch.version.cuda}")
        
        # Test thử tạo Tensor trên GPU
        try:
            x = torch.rand(1000, 1000).to("cuda")
            print("✅ Test ghi dữ liệu vào VRAM: THÀNH CÔNG")
        except Exception as e:
            print(f"❌ LỖI VRAM: {e}")
            return
    else:
        print("❌ CẢNH BÁO: Torch không tìm thấy GPU! Code đang chạy bằng CPU.")
        print("   -> Hãy cài lại pytorch bản cuda: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        return

    # 2. KIỂM TRA MODEL & BATCHING
    print("\n⏳ Đang load model 'AITeamVN/Vietnamese_Embedding' vào GPU...")
    try:
        # Ép cứng tham số tại đây để test
        embed_model = HuggingFaceEmbedding(
            model_name="AITeamVN/Vietnamese_Embedding",
            device="cuda",
            embed_batch_size=64 # Test batch 64
        )
        print("✅ Load model thành công.")
    except Exception as e:
        print(f"❌ Lỗi load model: {e}")
        return

    # 3. TEST TỐC ĐỘ THỰC TẾ (BENCHMARK)
    print("\n🏃 Đang test tốc độ embed 100 câu mẫu...")
    sample_texts = ["Hôm nay trời đẹp quá"] * 100 # Tạo 100 câu giả
    
    start_time = time.time()
    embeddings = embed_model.get_text_embedding_batch(sample_texts)
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"✅ Đã embed xong 100 câu trong: {duration:.2f} giây")
    print(f"🚀 Tốc độ trung bình: {100/duration:.2f} câu/giây")

    if duration > 5:
        print("\n⚠️ KẾT LUẬN: QUÁ CHẬM! Có thể GPU vẫn chưa được kích hoạt đúng cách.")
    else:
        print("\n🎉 KẾT LUẬN: GPU CHẠY NGON! Tốc độ này là chuẩn.")
        print("   -> Vấn đề nằm ở file index_builder.py cũ, không phải do máy.")

if __name__ == "__main__":
    test_gpu_speed()