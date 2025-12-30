import sys
import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_groq import ChatGroq
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv

# --- SETUP ĐƯỜNG DẪN ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.vector_store import get_vector_store
from config.embed import load_embed
from api.langchain_utils import get_conversational_rag_chain

load_dotenv()
api_key = os.getenv("MY_API_KEY")

# ==============================================================================
# 👇 CLASS WRAPPER ĐÃ FIX LỖI VALIDATION
# ==============================================================================
class LlamaIndexToLangchainWrapper(Embeddings):
    def __init__(self, llama_model):
        # Lưu model thật vào biến khác để dùng tính toán
        self.internal_model = llama_model
        # 👇 QUAN TRỌNG: Gán tên model dạng String để Ragas không báo lỗi Pydantic
        self.model = "AITeamVN/Vietnamese_Embedding"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Gọi model thật để embed
        return [self.internal_model.get_text_embedding(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        # Gọi model thật để embed
        return self.internal_model.get_query_embedding(text)
# ==============================================================================

def run_evaluation():
    print("🚀 Đang khởi động hệ thống Lucfin RAG để chấm thi...")
    
    # 1. Load Model & Wrap lại
    print("   - Loading Embedding Model...")
    llama_embed_model = load_embed()
    ragas_embed_model = LlamaIndexToLangchainWrapper(llama_embed_model) 

    print("   - Loading Vector Store...")
    vector_store = get_vector_store()
    
    # Load LLM
    print("   - Loading LLMs...")
    # Dùng Llama 3.3 70B thay cho Qwen đã bị xóa
    llm_rag = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, temperature=0)
    rag_chain = get_conversational_rag_chain(llm_rag, vector_store)
    
    judge_llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, temperature=0)

    # 2. Đọc Testset & Cắt ngắn
    testset_path = os.path.join("evaluation", "testset_ground_truth.csv")
    if not os.path.exists(testset_path):
        print("❌ Không tìm thấy file testset_ground_truth.csv")
        return

    full_df = pd.read_csv(testset_path)
    
    # 👇 CẮT LẤY 5 CÂU (Đảm bảo cắt ngay từ đầu)
    df = full_df.head(10).copy()
    print(f"📥 Đã tải {len(df)} câu hỏi (Test nhanh 5 câu).")

    # 3. Bot làm bài (Inference)
    answers = []
    contexts = []
    
    print("🤖 Bot đang trả lời...")
    for index, row in df.iterrows():
        question = row['question']
        try:
            # Fake chat history rỗng
            response = rag_chain.invoke({"input": question, "chat_history": []})
            
            ans_text = str(response['answer'])
            # Lấy list nội dung context
            source_docs = [doc.page_content for doc in response['context']]
            
            answers.append(ans_text)
            contexts.append(source_docs)
            print(f"   ✅ Done Q{index+1}")
        except Exception as e:
            print(f"   ❌ Lỗi Q{index+1}: {e}")
            answers.append("Lỗi hệ thống")
            contexts.append(["No context found"])

    # 4. Chuẩn bị dữ liệu chấm
    ragas_data = {
        'question': df['question'].tolist(),
        'answer': answers,
        'contexts': contexts,
        'ground_truth': df['ground_truth'].tolist()
    }
    dataset = Dataset.from_dict(ragas_data)

    # 5. Chấm điểm
    print("\n⚖️  Giám khảo Ragas đang chấm điểm...")
    # Lưu ý: Cảnh báo "1 generations instead of 3" là bình thường với Groq, cứ kệ nó.
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=judge_llm, 
        embeddings=ragas_embed_model # Dùng Wrapper đã fix
    )

    # 6. Xuất kết quả
    print("\n📊 KẾT QUẢ FINAL:")
    print(results)
    
    output_excel = os.path.join("evaluation", "lucfin_final_report.xlsx")
    results.to_pandas().to_excel(output_excel, index=False)
    print(f"✅ Xong! File Excel lưu tại: {output_excel}")

if __name__ == "__main__":
    run_evaluation()