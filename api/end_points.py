import re
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from config.vector_store import get_vector_store
from utils.utils import remove_think_tags
from api.langchain_utils import get_conversational_rag_chain
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
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

# --- Helper: Router
def classify_query(llm, query: str) -> str:
    """
    Phân loại câu hỏi: NUTRITION hay CHITCHAT
    """
    template = """
    Bạn là một công cụ phân loại văn bản.
    Nhiệm vụ: Chỉ trả về đúng 1 từ: "NUTRITION" hoặc "CHITCHAT".
    
    HƯỚNG DẪN:
    - NUTRITION: Câu hỏi về món ăn, cách nấu, calo, thực phẩm, ăn uống, bệnh lý ăn kiêng.
    - CHITCHAT: Câu hỏi về thời tiết, giá vàng, chứng khoán, tin tức, chào hỏi, tên bạn là gì, lập trình, chính trị.

    Câu hỏi: "{question}"
    
    Phân loại (Chỉ trả về 1 từ):
    """
    
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    try:
        # 1. Gọi LLM
        raw_result = chain.invoke({"question": query})
        
        # 2. XÓA SẠCH THẺ <THINK> TRƯỚC KHI KIỂM TRA
        # Đây là bước quan trọng để tránh bắt nhầm từ khóa trong suy nghĩ
        clean_result = remove_think_tags(str(raw_result)).strip().upper()
        
        print(f"🔍 DEBUG ROUTER: Raw='{raw_result[:20]}...' -> Clean='{clean_result}'")
        
        # 3. Kiểm tra logic (Ưu tiên bắt NUTRITION trước cho an toàn)
        if "NUTRITION" in clean_result: 
            return "NUTRITION"
        if "CHITCHAT" in clean_result: 
            return "CHITCHAT"
            
        # Fallback: Nếu không rõ là gì, cứ coi là Nutrition để RAG xử lý tiếp
        return "NUTRITION" 
        
    except Exception as e:
        print(f"⚠️ Router Error: {e}")
        return "NUTRITION"

# --- Helper: Extract Image ---
def extract_image_link(text):
    pattern = r"!\[.*?\]\((http.*?)\)"
    match = re.search(pattern, text)
    if match:
        image_url = match.group(1)
        clean_text = re.sub(pattern, "", text).strip()
        clean_text = re.sub(r'\n\s*\n', '\n\n', clean_text)
        return clean_text, image_url
    return text, None

# --- Endpoint Chính ---
@router.post("/ask", response_model=ChatMessageResponse)
async def ask_nutrition(req: NutritionRequest):
    try:
        load_dotenv()
        api_key = os.getenv("MY_API_KEY")
        
        # 1. Khởi tạo LLM
        llm = ChatGroq(model="qwen/qwen3-32b", api_key=api_key, temperature=0)
        
        # 2. Lấy lịch sử chat
        chat_history = get_chat_history(req.session_id)
        print(f"🗣️ [{req.session_id}] User: {req.question}")

        # 3. PHÂN LOẠI CÂU HỎI (ROUTER STEP)
        intent = classify_query(llm, req.question)
        print(f"🧭 INTENT DETECTED: {intent}")  # <--- Nhìn dòng này trong Terminal để debug

        final_answer = ""
        image_url = None
        sources = []

        # --- TRƯỜNG HỢP 1: HỎI VỀ DINH DƯỠNG (CHẠY RAG) ---
        if intent == "NUTRITION":
            index = get_root_index()
            rag_chain = get_conversational_rag_chain(llm, index)
            
            # Chạy RAG Chain (Tìm kiếm + Trả lời)
            response = rag_chain.invoke({
                "input": req.question,
                "chat_history": chat_history
            })
            
            raw_answer = remove_think_tags(str(response["answer"]))
            
            # Lấy ảnh từ metadata
            source_docs = response.get("context", [])
            if source_docs:
                first_doc = source_docs[0]
                metadata = first_doc.metadata
                # Ưu tiên lấy image_link (key chuẩn trong DB của bạn)
                image_url = metadata.get("image_link") or metadata.get("image") or metadata.get("link")
                
                # Lấy tên các món ăn tham khảo
                for doc in source_docs:
                    sources.append(doc.metadata.get("dish_name", "Tài liệu gốc"))
            
            final_answer, _ = extract_image_link(raw_answer) # Làm sạch text lần nữa

        # --- TRƯỜNG HỢP 2: HỎI XÃ GIAO (KHÔNG RAG) ---
        else:
            # Prompt từ chối khéo léo
            refusal_prompt = [
                ("system", "Bạn là Lucfin, trợ lý ảo chuyên về dinh dưỡng và ẩm thực. "
                           "Phong cách trả lời: Thân thiện, lịch sự, ngắn gọn (dưới 2 câu), lúc nào cũng ghi nhớ tên bạn là Lucfin. "
                           "Nếu người dùng hỏi chủ đề không liên quan (giá vàng, thời tiết, chính trị...), "
                           "hãy xin lỗi khéo léo và gợi ý họ hỏi về món ăn, luôn nhắc tên Lucfin kèm theo lời cảm ơn, xin lỗi"),
                ("human", req.question)
            ]
            # Gọi trực tiếp LLM (Không tốn token vector search)
            ai_msg = llm.invoke(refusal_prompt)
            final_answer = remove_think_tags(str(ai_msg.content))
            
            # Chitchat thì không có ảnh và nguồn
            image_url = None
            sources = []

        # 4. Lưu lịch sử chat
        chat_history.append(HumanMessage(content=req.question))
        chat_history.append(AIMessage(content=final_answer))
        
        # Giữ lại 6 tin nhắn gần nhất để tiết kiệm token
        if len(chat_history) > 6:
            chat_history = chat_history[-6:]
            CHAT_HISTORIES[req.session_id] = chat_history

        # 5. Trả về kết quả
        return ChatMessageResponse(
            answer=final_answer,
            image=image_url,
            sourceDocuments=list(set(sources))
        )
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))