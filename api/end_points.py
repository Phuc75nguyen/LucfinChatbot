from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import re
from config.vector_store import get_vector_store
from router.prompt import qa_prompt_tmpl  # Lấy cái prompt xịn có format Markdown
from utils.utils import remove_think_tags

# Khởi tạo Router của FastAPI
router = APIRouter()

# Biến global để lưu index (tránh load lại nhiều lần)
QUERY_ENGINE = None

def get_engine():
    global QUERY_ENGINE
    if QUERY_ENGINE is None:
        # Load Index từ FoodDB
        index = get_vector_store()
        
        # Tạo Engine truy vấn trực tiếp
        # similarity_top_k=5: Tăng khả năng tìm kiếm (theo yêu cầu)
        QUERY_ENGINE = index.as_query_engine(
            similarity_top_k=5, 
            text_qa_template=qa_prompt_tmpl # Gắn "nhân cách" Lucfin vào đây
        )
    return QUERY_ENGINE

# --- Models ---
class NutritionRequest(BaseModel):
    question: str

class ChatMessageResponse(BaseModel):
    answer: str
    image: Optional[str] = None
    sourceDocuments: Optional[List[str]] = None

# --- Endpoints ---
@router.post("/ask", response_model=ChatMessageResponse)
async def ask_nutrition(req: NutritionRequest):
    try:
        engine = get_engine()
        
        # Gửi câu hỏi vào hệ thống RAG
        print(f"User asking: {req.question}")
        response = engine.query(req.question)
        
        # Bước 1: Làm sạch thẻ <think>
        clean_answer = remove_think_tags(str(response))
        
        # Bước 2: Tách ảnh thông minh
        image_url = None
        # Regex tìm link ảnh markdown: ![alt](url)
        # Group 1 là url
        match = re.search(r'!\[.*?\]\((.*?)\)', clean_answer)
        
        if match:
            image_url = match.group(1)
            # Xóa chuỗi markdown ảnh khỏi câu trả lời để không bị rác
            # match.group(0) là toàn bộ chuỗi ![...](...)
            clean_answer = clean_answer.replace(match.group(0), "").strip()
            
        # Trích xuất nguồn (nếu có metadata)
        sources = []
        if hasattr(response, 'source_nodes'):
            for node in response.source_nodes:
                # Lấy tên món ăn từ metadata để hiển thị source cho đẹp
                dish_name = node.metadata.get("dish_name", "Unknown Document")
                sources.append(dish_name)

        return ChatMessageResponse(
            answer=clean_answer,
            image=image_url,
            sourceDocuments=list(set(sources)) # Loại bỏ trùng lặp
        )
        
    except Exception as e:
        print(f"Error processing request: {e}")
        # Trả về lỗi 500 hoặc trả về message lỗi trong answer tùy strategy
        # Ở đây mình raise HTTPException để client biết
        raise HTTPException(status_code=500, detail=str(e))