from sentence_transformers import CrossEncoder
import torch

_reranker_model = None

def load_reranker():
    """
    Loads the Cross-Encoder model as a singleton.
    Forces FP16 via model_kwargs to save VRAM on Quadro T1000.
    """
    global _reranker_model
    if _reranker_model is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Loading Cross-Encoder on device: {device_str.upper()} (FP16 Mode)")
        
        # SỬA LỖI TẠI ĐÂY: Dùng model_kwargs để truyền torch_dtype
        _reranker_model = CrossEncoder(
            'BAAI/bge-reranker-v2-m3', 
            device=device_str,
            # Đây là cách chính xác nhất cho phiên bản mới
            model_kwargs={"torch_dtype": torch.float16} 
        )
        
        # Cấu hình max_length sau khi khởi tạo (An toàn tuyệt đối)
        _reranker_model.max_length = 512 

    return _reranker_model