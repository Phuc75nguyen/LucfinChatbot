import re
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from dotenv import load_dotenv

# --- IMPORTS ---
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.vector_store import get_vector_store
from config.rerank import load_reranker 
from api.langchain_utils import get_conversational_rag_chain
from utils.utils import remove_think_tags
from utils.session_manager import update_scan_result, get_scanned_context 

router = APIRouter()

# =========================================================
# 👇 QUẢN LÝ TRẠNG THÁI (STATE MANAGEMENT) TẠI CHỖ
# =========================================================
ROOT_INDEX = None
CHAT_HISTORIES: Dict[str, List] = {}

# Di chuyển biến SESSION_FOCUS về đây để đảm bảo tính nhất quán
# "SCAN": Đang tập trung vào món vừa chụp
# "RAG": Đang chat về món trong database/chủ đề mới
SESSION_FOCUS: Dict[str, str] = {} 

def get_root_index():
    global ROOT_INDEX
    if ROOT_INDEX is None:
        ROOT_INDEX = get_vector_store()
    return ROOT_INDEX

def get_chat_history(session_id: str):
    if session_id not in CHAT_HISTORIES:
        CHAT_HISTORIES[session_id] = []
    return CHAT_HISTORIES[session_id]

CV_TO_VIETNAMESE = {
    "Suon": "Sườn non", "Cha Ca": "Chả cá", "Tofu": "Đậu hũ", "Unknown": ""
}

class NutritionRequest(BaseModel):
    question: str
    session_id: str = "default_user"

class ScanData(BaseModel):
    session_id: str
    detected_classes: List[str] 

class ChatMessageResponse(BaseModel):
    answer: str
    image: Optional[str] = None
    sourceDocuments: Optional[List[str]] = None

# =========================================================
# 👇 ROUTER & HELPER
# =========================================================
def classify_query(llm, query: str) -> str:
    template = """
    Phân loại câu hỏi:
    1. "FOLLOWUP": Hỏi tiếp về món đang nói ("món này", "nó", "vừa ăn", "có béo không", "ngon không").
    2. "NEW_TOPIC": Hỏi về món ăn cụ thể MỚI có tên riêng ("Cơm hến", "Phở", "Bún bò", "Canh chua").
    3. "CHITCHAT": Xã giao, thời tiết, chính trị, bóng đá, không liên quan ăn uống.
    
    Câu hỏi: "{question}"
    Chỉ trả về 1 từ (FOLLOWUP / NEW_TOPIC / CHITCHAT):
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    try:
        res = chain.invoke({"question": query})
        clean = remove_think_tags(str(res)).strip().upper()
        if "CHIT" in clean: return "CHITCHAT"
        if "NEW" in clean: return "NEW_TOPIC"
        return "FOLLOWUP"
    except: return "FOLLOWUP"

def extract_image_link(text):
    pattern = r"!\[.*?\]\((http.*?)\)"
    match = re.search(pattern, text)
    if match: return re.sub(pattern, "", text).strip(), match.group(1)
    return text, None

# --- API SCAN ---
@router.post("/scan")
async def receive_scan_data(data: ScanData):
    mapped = []
    for item in data.detected_classes:
        vn = CV_TO_VIETNAMESE.get(item, item)
        if vn: mapped.append(vn)
    if mapped:
        update_scan_result(data.session_id, mapped)
        
        # 👇 KHI SCAN: BẮT BUỘC CHUYỂN TIÊU ĐIỂM VỀ SCAN
        SESSION_FOCUS[data.session_id] = "SCAN"
        print(f"📸 [Session: {data.session_id}] Focus set to: SCAN")
        
        return {"message": "Đã đồng bộ context.", "mapped_names": mapped}
    return {"message": "Không nhận diện được."}

# --- API ASK ---
@router.post("/ask", response_model=ChatMessageResponse)
async def ask_nutrition(req: NutritionRequest):
    try:
        load_dotenv()
        api_key = os.getenv("MY_API_KEY")
        load_reranker()
        llm = ChatGroq(model="qwen/qwen3-32b", api_key=api_key, temperature=0)
        
        chat_history = get_chat_history(req.session_id)
        scanned_food = get_scanned_context(req.session_id)
        
        # 1. Phân loại ý định
        intent = classify_query(llm, req.question)
        
        # 2. QUẢN LÝ TIÊU ĐIỂM (LOGIC CHẶT CHẼ HƠN)
        if intent == "NEW_TOPIC":
            # Nếu hỏi món mới -> Quên ngay món Scan -> Chuyển sang RAG
            SESSION_FOCUS[req.session_id] = "RAG"
            print(f"🔄 Intent là NEW_TOPIC -> Chuyển Focus sang: RAG")
        
        # Lấy focus hiện tại (Mặc định là RAG nếu chưa có)
        current_focus = SESSION_FOCUS.get(req.session_id, "RAG")
        
        print(f"🗣️ User: {req.question} | Intent: {intent} | Focus: {current_focus}")

        final_answer, image_url, sources = "", None, []

        # ==============================================================================
        # 🔴 LUỒNG A: SCAN FOLLOWUP (Chỉ chạy khi User đang nhìn vào Camera)
        # ==============================================================================
        # Điều kiện: Intent là Followup VÀ Focus đang là SCAN VÀ Có dữ liệu Scan
        if intent == "FOLLOWUP" and current_focus == "SCAN" and scanned_food:
            print("🚀 CASE A: Trả lời về món Scan (General Knowledge).")
            
            system_prompt = (
                f"Bạn là Lucfin. Người dùng đang hỏi về món họ vừa chụp ảnh: {scanned_food}. "
                "Hãy trả lời ngắn gọn (80 chữ), tập trung dinh dưỡng, không cần tra cứu DB."
            )
            ai_msg = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=req.question)])
            final_answer = remove_think_tags(str(ai_msg.content))
            image_url = "USE_LOCAL_IMAGE"
            sources = ["Kiến thức tổng quát Lucfin"]

        # ==============================================================================
        # 🔵 LUỒNG B: RAG FOODDB (Chạy khi New Topic HOẶC Focus đang là RAG)
        # ==============================================================================
        elif intent == "NEW_TOPIC" or (intent == "FOLLOWUP" and current_focus == "RAG"):
            print("books CASE B: Chạy RAG tìm kiếm trong FoodDB.")
            
            index = get_root_index()
            rag_chain = get_conversational_rag_chain(llm, index)
            response = rag_chain.invoke({"input": req.question, "chat_history": chat_history})
            
            raw_answer = remove_think_tags(str(response["answer"]))
            final_answer = raw_answer # Tạm gán
            
            # --- 👇👇👇 LOGIC MỚI: KIỂM TRA TỪ CHỐI (REFUSAL CHECK) 👇👇👇 ---
            # Các từ khóa cho thấy Bot đang từ chối trả lời món hư cấu
            refusal_keywords = ["món ăn hư cấu", "không phải là món ăn thực tế", "không có thực", "xin lỗi"]
            
            is_refused = any(keyword in raw_answer.lower() for keyword in refusal_keywords)
            
            if is_refused:
                print("🚫 Phát hiện câu trả lời từ chối -> Ẩn ảnh và nguồn.")
                image_url = None
                sources = []
            else:
                # Chỉ lấy ảnh nếu KHÔNG bị từ chối
                source_docs = response.get("context", [])
                if source_docs:
                    meta = source_docs[0].metadata
                    image_url = meta.get("image_link") or meta.get("image")
                    sources = [d.metadata.get("dish_name", "Tài liệu") for d in source_docs]
                
                # Check ảnh trong text (nếu có)
                final_answer, extracted_img = extract_image_link(raw_answer)
                if not image_url and extracted_img: image_url = extracted_img

        # ==============================================================================
        # 🟡 LUỒNG C: CHITCHAT (ĐÃ SỬA: CẤM TRẢ LỜI THỜI TIẾT)
        # ==============================================================================
        else:
            print("💬 CASE C: Chitchat (Kích hoạt bộ lọc nội dung).")
            # 👇👇👇 PROMPT CỰC GẮT ĐỂ CẤM HỎI THỜI TIẾT 👇👇👇
            system_instruction = (
                "Bạn là Lucfin, trợ lý chuyên về DINH DƯỠNG và ẨM THỰC. "
                "QUY TẮC TỪ CHỐI (REFUSAL POLICY):"
                "1. Nếu người dùng hỏi về: Thời tiết, Giá vàng, Chứng khoán, Chính trị, Lịch sử, Code, Tin tức..."
                "   -> HÃY TỪ CHỐI LỊCH SỰ. Nói: 'Xin lỗi, tôi là trợ lý dinh dưỡng, tôi không có thông tin về vấn đề này.'"
                "   -> TUYỆT ĐỐI KHÔNG bịa ra thời tiết hay thông tin sai lệch."
                "2. Nếu hỏi 'Bạn là ai', 'Ai tạo ra bạn':"
                "   -> Trả lời: 'Tôi là Lucfin, sản phẩm của đội ngũ NutriAI.'"
            )
            ai_msg = llm.invoke([("system", system_instruction), ("human", req.question)])
            final_answer = remove_think_tags(str(ai_msg.content))

        # 4. Update History
        chat_history.append(HumanMessage(content=req.question))
        chat_history.append(AIMessage(content=final_answer))
        if len(chat_history) > 6: CHAT_HISTORIES[req.session_id] = chat_history[-6:]

        return ChatMessageResponse(answer=final_answer, image=image_url, sourceDocuments=list(set(sources)))

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))