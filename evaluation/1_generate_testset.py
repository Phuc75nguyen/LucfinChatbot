import pandas as pd
import random
import os

# Đường dẫn file dữ liệu gốc
CSV_PATH = os.path.join("data_raw", "foods.csv")
OUTPUT_PATH = os.path.join("evaluation", "testset_ground_truth.csv")

def generate_testset():
    print(f"📂 Đang đọc dữ liệu từ {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    # Chỉ lấy khoảng 20-30 món ngẫu nhiên để test (Demo)
    # Khi chạy thật có thể tăng lên 50
    sample_df = df.sample(n=20, random_state=42)
    
    test_data = []

    for _, row in sample_df.iterrows():
        dish_name = row['dish_name']
        
        # 1. Tạo câu hỏi về Thành phần (Ingredients)
        if pd.notna(row['ingredients']):
            test_data.append({
                "question": f"Thành phần chính của món {dish_name} gồm những gì?",
                "ground_truth": row['ingredients']
            })
            
        # 2. Tạo câu hỏi về Calo (Nutrition) - Nếu có cột calories
        if pd.notna(row['calories']):
             test_data.append({
                "question": f"Món {dish_name} bao nhiêu calo?",
                "ground_truth": f"Khoảng {row['calories']} calo."
            })

    # Thêm vài câu hỏi bẫy (Edge Cases) thủ công
    test_data.append({
        "question": "Món trứng khủng long kho tộ có ngon không?",
        "ground_truth": "Xin lỗi, đây là món ăn hư cấu không có thực."
    })
    
    test_data.append({
        "question": "Thời tiết hôm nay thế nào?",
        "ground_truth": "Xin lỗi, tôi là trợ lý dinh dưỡng, tôi không trả lời về thời tiết."
    })

    # Lưu ra CSV
    result_df = pd.DataFrame(test_data)
    result_df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Đã tạo bộ testset gồm {len(result_df)} câu hỏi tại: {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_testset()