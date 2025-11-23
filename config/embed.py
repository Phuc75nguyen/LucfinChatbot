from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import torch

def load_embed():
    
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"🚀 Đang chạy Embedding trên thiết bị: {device_str.upper()}")

    embed_model = HuggingFaceEmbedding(
        #Phải chỉ định model tiếng Việt
        model_name="AITeamVN/Vietnamese_Embedding", 
        device=device_str,  # Truyền string vào đây
        embed_batch_size=4 # Card T1000 tải tốt mức này
    )
    return embed_model