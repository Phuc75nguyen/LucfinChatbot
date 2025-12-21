from typing import List, Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ==============================================================================
# 1. CLASS WRAPPER (CẦU NỐI GIỮA LLAMAINDEX VÀ LANGCHAIN)
# ==============================================================================
class LlamaIndexRetrieverWrapper(BaseRetriever):
    index: Any 

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Hàm này nhận câu hỏi (query), gọi LlamaIndex để tìm kiếm,
        sau đó chuyển đổi kết quả thành định dạng Document của LangChain.
        """
        # Tạo retriever từ index gốc
        retriever = self.index.as_retriever(similarity_top_k=3)
        
        # Thực hiện truy vấn
        nodes = retriever.retrieve(query)
        
        # Chuyển đổi Node (LlamaIndex) -> Document (LangChain)
        docs = []
        for node in nodes:
            # Lấy nội dung text
            content = node.get_text()
            
            # Lấy metadata (tên món, ảnh, nguồn...)
            metadata = node.metadata if node.metadata else {}
            
            # Đóng gói thành Document
            docs.append(Document(page_content=content, metadata=metadata))
            
        return docs

# ==============================================================================
# 2. HÀM TẠO CHAIN RAG (DÙNG PROMPT V14 - CHUẨN RAG)
# ==============================================================================
def get_conversational_rag_chain(llm, index):
    # Bước 1: Khởi tạo Retriever
    retriever = LlamaIndexRetrieverWrapper(index=index)
    
    # Bước 2: Prompt để "Cô lập câu hỏi" (Contextualize)
    # Giúp AI hiểu câu hỏi dựa trên lịch sử (ví dụ: "Nó bao nhiêu calo?" -> "Phở bao nhiêu calo?")
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    
    # Tạo History Aware Retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    # Bước 3: Prompt Trả lời (V14 - RAG CHUẨN MỰC CHO FOODDB)
    qa_system_prompt = (
        "Bạn là Lucfin, chuyên gia dinh dưỡng thực tế. "
        "Dưới đây là tài liệu tham khảo (Context) từ FoodDB:\n"
        "---------------------\n"
        "{context}\n"
        "---------------------\n\n"
        
        "QUY TRÌNH TRẢ LỜI:"
        "1. KIỂM TRA THỰC TẾ (REALITY CHECK - QUAN TRỌNG NHẤT):"
        "   - Trước khi trả lời, hãy tự hỏi: Món này có thật và ăn được không?"
        "   - Nếu món ăn là HƯ CẤU, PHI LÝ hoặc KHÔNG THỂ ĂN ĐƯỢC (Ví dụ: 'Trứng khủng long', 'Thịt rồng', 'Canh nước mắt cá sấu', 'Bê tông xào', 'Gạch nung chấm mắm')..."
        "   -> TỪ CHỐI TRẢ LỜI NGAY. Nói: 'Xin lỗi, món [Tên món] không phải là thực phẩm thực tế, Lucfin không thể phân tích.'"
        "   -> TUYỆT ĐỐI KHÔNG BỊA ra dinh dưỡng cho các món hư cấu này."
        "   -> TUYỆT ĐỐI KHÔNG dùng bảng (Markdown Table) với các ký tự '|' và '---'."
        
        "2. XỬ LÝ DỰA TRÊN CONTEXT:"
        "   - Nếu là món ăn thật (Ví dụ: 'Phở', 'Cơm hến') -> Ưu tiên dùng thông tin trong Context để trả lời."
        
        "3. XỬ LÝ KHI THIẾU DỮ LIỆU (FALLBACK):"
        "   - Nếu là món thật nhưng không có trong Context -> Được phép dùng kiến thức chuyên gia để ước lượng."
        "   - Nếu hỏi CÔNG THỨC mà không có trong Context -> Báo chưa có dữ liệu."
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain