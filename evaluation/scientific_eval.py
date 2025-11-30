import asyncio
import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from dotenv import load_dotenv

# Import directly from API to bypass HTTP overhead
from api.end_points import ask_nutrition, NutritionRequest

# 1. Setup Environment
load_dotenv()
api_key = os.getenv("MY_API_KEY")

# 2. Configure Judge (LLM)
# Using a strong model for evaluation
print("⚖️  Configuring Judge (Llama3-70b)...")
judge_llm = ChatGroq(
    model="mixtral-8x7b-32768", 
    api_key=api_key, 
    temperature=0
)
ragas_judge = LangchainLLMWrapper(judge_llm)

# 3. Configure Embeddings
# Using the same embedding model as the RAG system for consistency
print("🧠 Configuring Embeddings (AITeamVN)...")
# Note: We use Langchain's HuggingFaceEmbeddings wrapper here
hf_embeddings = HuggingFaceEmbeddings(
    model_name="AITeamVN/Vietnamese_Embedding",
    model_kwargs={'device': 'cpu'}, # Use CPU for eval to avoid VRAM conflict if needed, or 'cuda'
    encode_kwargs={'normalize_embeddings': False}
)
ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

# 4. Define Test Data (Ground Truth)
# Format: Question + Ground Truth Answer
test_data_samples = [
    {
        "question": "Phở bò bao nhiêu calo?",
        "ground_truth": "Một tô phở bò trung bình chứa khoảng 300-450 calo, tùy thuộc vào lượng bánh phở, thịt và nước dùng."
    },
    {
        "question": "Cách nấu canh chua cá lóc miền Tây?",
        "ground_truth": "Nấu canh chua cá lóc miền Tây cần cá lóc, bạc hà, đậu bắp, thơm, cà chua, giá đỗ, me chua và rau nêm (ngò gai, rau om). Cá làm sạch, nấu nước me, phi tỏi, cho cá vào nấu chín, vớt ra. Nấu rau củ, nêm gia vị chua ngọt, cho cá lại, thêm rau nêm."
    },
    {
        "question": "Ăn chuối có béo không?",
        "ground_truth": "Chuối không gây béo nếu ăn vừa phải. Một quả chuối trung bình chứa khoảng 105 calo, giàu chất xơ và kali, tốt cho tiêu hóa và tim mạch."
    },
    {
        "question": "Bệnh tiểu đường nên ăn gì?",
        "ground_truth": "Người bệnh tiểu đường nên ăn rau xanh, ngũ cốc nguyên hạt, các loại đậu, cá béo, thịt nạc. Hạn chế tinh bột nhanh, đường, đồ ngọt và chất béo bão hòa."
    },
    {
        "question": "100g ức gà bao nhiêu protein?",
        "ground_truth": "100g ức gà sống chứa khoảng 23g protein. Khi nấu chín, lượng protein có thể cao hơn một chút do mất nước, khoảng 31g protein."
    }
]

async def run_eval():
    print("🚀 Starting Evaluation Loop...")
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    # 5. Data Collection Loop
    for item in test_data_samples:
        q = item["question"]
        gt = item["ground_truth"]
        
        print(f"Testing: {q}")
        
        # Call API function directly
        req = NutritionRequest(question=q, session_id="eval_run")
        response = await ask_nutrition(req)
        
        # Collect data
        questions.append(q)
        answers.append(response.answer)
        # Ragas expects a list of strings for contexts
        # We use the retrieved_contexts field we added to the API
        ctx = response.retrieved_contexts if response.retrieved_contexts else []
        contexts.append(ctx)
        ground_truths.append(gt)
        
        # Optional: Sleep to avoid rate limits if needed
        await asyncio.sleep(1)

    # 6. Prepare Dataset
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data)
    
    print("📊 Running Ragas Evaluation...")
    
    # 7. Run Evaluation
    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
        ],
        llm=ragas_judge,
        embeddings=ragas_embeddings
    )
    
    print("✅ Evaluation Complete!")
    print(results)
    
    # 8. Export Results
    df = results.to_pandas()
    output_file = "evaluation/scientific_report.xlsx"
    df.to_excel(output_file, index=False)
    print(f"💾 Report saved to: {output_file}")

if __name__ == "__main__":
    # Run async loop
    asyncio.run(run_eval())
