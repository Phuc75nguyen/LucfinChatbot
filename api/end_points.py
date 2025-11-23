from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from config.vector_store import get_vector_store
from router.prompt import qa_prompt_tmpl  # Lấy cái prompt xịn có format Markdown

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
        # similarity_top_k=3: Lấy 3 món giống nhất
        QUERY_ENGINE = index.as_query_engine(
            similarity_top_k=3, 
            text_qa_template=qa_prompt_tmpl # Gắn "nhân cách" Lucfin vào đây
        )
    return QUERY_ENGINE

# --- Models ---
class NutritionRequest(BaseModel):
    question: str

class ChatMessageResponse(BaseModel):
    answer: str
    sourceDocuments: Optional[List[str]] = None

# --- Endpoints ---
@router.post("/ask", response_model=ChatMessageResponse)
async def ask_nutrition(req: NutritionRequest):
    engine = get_engine()
    
    # Gửi câu hỏi vào hệ thống RAG
    print(f"❓ User asking: {req.question}")
    response = engine.query(req.question)
    
    # Trích xuất nguồn (nếu có metadata)
    sources = []
    for node in response.source_nodes:
        # Lấy tên món ăn từ metadata để hiển thị source cho đẹp
        dish_name = node.metadata.get("dish_name", "Unknown Document")
        sources.append(dish_name)

    return ChatMessageResponse(
        answer=str(response),
        sourceDocuments=list(set(sources)) # Loại bỏ trùng lặp
    )