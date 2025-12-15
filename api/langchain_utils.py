import time
from typing import List, Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

class LlamaIndexRetrieverWrapper(BaseRetriever):
    """Wraps a LlamaIndex retriever to work with LangChain."""
    index: Any
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant to a query."""
        from config.rerank import load_reranker

        start_time = time.time()
        print(f"\n🔍 [RAG] Bắt đầu tìm kiếm cho: '{query}'")

        # Step 1: Retrieve (High Recall) - Lấy 10 ứng viên để tránh sót
        t1 = time.time()
        retriever = self.index.as_retriever(similarity_top_k=5) 
        response = retriever.retrieve(query)
        print(f"⏱️  Retrieve Time (LlamaIndex): {time.time() - t1:.4f}s")
        
        # Step 2: Re-rank (Cross-Encoder)
        reranker = load_reranker()
        
        if not response:
            return []
            
        # Prepare pairs: (query, node_text)
        # Chỉ lấy text thuần túy để re-rank, tránh nhiễu metadata
        pairs = [(query, node.get_content()) for node in response]
        
        # --- TỐI ƯU HIỆU NĂNG GPU (CRITICAL OPTIMIZATION) ---
        t2 = time.time()
        # max_length=512: Cắt ngắn văn bản, giúp T1000 không phải tính toán ma trận quá lớn
        # batch_size=10: Xử lý gọn trong 1 batch
        scores = reranker.predict(
            pairs, 
            batch_size=10,
            show_progress_bar=False
        )
        print(f"⏱️  Re-rank Time (GPU): {time.time() - t2:.4f}s")
        
        # Zip nodes with scores and sort by score (descending)
        scored_nodes = list(zip(response, scores))
        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        
        # Pick Top 2 (Precision)
        top_2_nodes = scored_nodes[:2]
        
        # Convert LlamaIndex nodes to LangChain documents
        documents = []
        for node, score in top_2_nodes:
            content = node.get_content()
            metadata = node.metadata.copy() # Copy để an toàn
            # Add score to metadata for debugging
            metadata['re_rank_score'] = float(score)
            documents.append(Document(page_content=content, metadata=metadata))
            
        print(f"🚀 Tổng thời gian Pipeline: {time.time() - start_time:.4f}s")
        return documents

def get_conversational_rag_chain(llm, index):
    """
    Creates a conversational RAG chain using LangChain.
    """
    
    # 1. Define the Retriever
    retriever = LlamaIndexRetrieverWrapper(index=index)
    
    # 2. Contextualize Question Prompt
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
    
    # Create the history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    # 3. Answer Question Prompt
    # TỐI ƯU SYSTEM PROMPT: Bỏ yêu cầu chào hỏi rườm rà, tập trung cảnh báo y tế
    """qa_system_prompt = (
        "Bạn là Lucfin, chuyên gia dinh dưỡng súc tích. "
        "Sử dụng các đoạn ngữ cảnh (Context) dưới đây để trả lời câu hỏi. "
        "QUY TẮC BẮT BUỘC: "
        "1. Trả lời Ngắn gọn (dưới 4 dòng), gạch đầu dòng."
        "2. Nếu Context không có thông tin, nói 'Tôi chưa có dữ liệu về món này'."
        "3. CẢNH BÁO SỨC KHỎE NGHIÊM TÚC nếu người dùng hỏi về bệnh (tiểu đường, v.v.). "
        "4. Không chào hỏi xã giao (như 'Chào bạn', 'Rất vui'). Đi thẳng vào vấn đề.\n\n"
        "{context}"
    )"""
    # TỐI ƯU SYSTEM PROMPT (Phiên bản V3 - Thích ứng):
    # - Hỏi bệnh/tư vấn: Trả lời ngắn gọn, cảnh báo.
    # - Hỏi công thức: Trả lời CHI TIẾT định lượng.
    # TỐI ƯU SYSTEM PROMPT (Phiên bản V5 - Chặn đứng ảo giác/Bịa đặt)
    qa_system_prompt = (
        "Bạn là Lucfin, trợ lý dinh dưỡng và ẩm thực chuyên sâu của dự án NutiAI. "
        "Nhiệm vụ của bạn là trả lời dựa trên Dữ liệu (Context) được cung cấp bên dưới.\n\n"
        
        "QUY TẮC XỬ LÝ KHI KHÔNG CÓ DỮ LIỆU (ƯU TIÊN SỐ 1 - BẮT BUỘC):"
        "1. Đọc kỹ Context. Nếu tên món ăn người dùng hỏi KHÔNG xuất hiện hoặc KHÔNG liên quan đến Context:"
        "   - PHẢI TRẢ LỜI DUY NHẤT CÂU SAU: 'Món \"{input}\" hiện không có trong dữ liệu ẩm thực của Lucfin. Có thể bạn đang nhầm lẫn tên món hoặc thuật ngữ.'"
        "   - TUYỆT ĐỐI KHÔNG tự phân tích từ ngữ (ví dụ: không được suy diễn 'đá' là 'thạch', 'sắt' là 'thịt')."
        "   - TUYỆT ĐỐI KHÔNG tự bịa ra công thức hoặc gợi ý món thay thế.\n\n"

        "QUY TẮC ĐỊNH DẠNG (KHI CÓ DỮ LIỆU):"
        "1. KHÔNG dùng bảng. Chỉ dùng gạch đầu dòng."
        "2. KHI HỎI CÔNG THỨC: Liệt kê đầy đủ Nguyên liệu & Định lượng (nếu có)."
        "3. KHI HỎI SỨC KHỎE: Trả lời ngắn gọn, cảnh báo bệnh lý."
        "4. Chào hỏi xã giao ngắn gọn. Đi thẳng vào vấn đề.\n\n"
        
        "DỮ LIỆU ĐẦU VÀO (Context):\n"
        "{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    
    # Create the document chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # 4. Create the final Retrieval Chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain