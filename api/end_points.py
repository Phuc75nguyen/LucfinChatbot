import re
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from config.vector_store import get_vector_store
from utils.utils import remove_think_tags
from api.langchain_utils import get_conversational_rag_chain
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
import os
from dotenv import load_dotenv

router = APIRouter()

# --- Global Store ---
CHAT_HISTORIES: Dict[str, List] = {}
ROOT_INDEX = None

def get_root_index():
    global ROOT_INDEX
    if ROOT_INDEX is None:
        ROOT_INDEX = get_vector_store()
    return ROOT_INDEX

def get_chat_history(session_id: str):
    if session_id not in CHAT_HISTORIES:
        CHAT_HISTORIES[session_id] = []
    return CHAT_HISTORIES[session_id]

# --- Models ---
class NutritionRequest(BaseModel):
    question: str
    session_id: str = "default_user"

class ChatMessageResponse(BaseModel):
    answer: str
    image: Optional[str] = None
    sourceDocuments: Optional[List[str]] = None

# --- Helper: Extract Image (Giữ lại để backup nếu metadata không có) ---
def extract_image_from_text(text):
    pattern = r"!\[.*?\]\((http.*?)\)"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return None

# --- Endpoint ---
@router.post("/ask", response_model=ChatMessageResponse)
async def ask_nutrition(req: NutritionRequest):
    try:
        load_dotenv()
        api_key = os.getenv("MY_API_KEY")
        
        # 1. Initialize LLM
        llm = ChatGroq(
            model="qwen/qwen3-32b",
            api_key=api_key,
            temperature=0
        )
        
        # 2. Get Index and Chat History
        index = get_root_index()
        chat_history = get_chat_history(req.session_id)
        
        print(f"🗣️ [{req.session_id}] User asking: {req.question}")
        
        # 3. Create RAG Chain
        rag_chain = get_conversational_rag_chain(llm, index)
        
        # 4. Invoke Chain
        response = rag_chain.invoke({
            "input": req.question,
            "chat_history": chat_history
        })
        
        # 5. Update History
        chat_history.append(HumanMessage(content=req.question))
        chat_history.append(AIMessage(content=response["answer"]))
        
        # Limit history length
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]
            CHAT_HISTORIES[req.session_id] = chat_history
            
        # 6. Process Result (PHẦN QUAN TRỌNG ĐÃ SỬA)
        raw_answer = remove_think_tags(str(response["answer"]))
        
        # --- LOGIC LẤY ẢNH MỚI ---
        image_url = None
        source_docs = response.get("context", [])
        sources = []

        if source_docs:
            # A. Lấy ảnh từ Metadata của tài liệu tìm thấy (Ưu tiên số 1)
            # Lấy tài liệu đầu tiên (độ tương đồng cao nhất)
            first_doc = source_docs[0]
            metadata = first_doc.metadata

            print(f"\n🔥🔥🔥 DEBUG METADATA: {metadata}\n")
            
            # Kiểm tra các key phổ biến mà bạn có thể đã lưu trong DB
            image_url = metadata.get("image_link") or metadata.get("image") or metadata.get("link")

            # B. Lấy nguồn tham khảo
            for doc in source_docs:
                dish_name = doc.metadata.get("dish_name", "Tài liệu gốc")
                sources.append(dish_name)

        # C. Fallback: Nếu metadata không có ảnh, thử tìm trong lời thoại (như code cũ)
        if not image_url:
            image_url = extract_image_from_text(raw_answer)

        # Xóa link ảnh markdown trong text (nếu có) để text sạch đẹp
        final_answer = re.sub(r"!\[.*?\]\((http.*?)\)", "", raw_answer).strip()
        final_answer = re.sub(r'\n\s*\n', '\n\n', final_answer)

        return ChatMessageResponse(
            answer=final_answer,
            image=image_url, # Trả về link ảnh tìm được từ metadata
            sourceDocuments=list(set(sources))
        )
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))