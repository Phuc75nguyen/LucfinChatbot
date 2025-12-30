import matplotlib.pyplot as plt
import numpy as np
import os

# --- CẤU HÌNH SỐ LIỆU GIẢ LẬP (FAKE DATA) ---
# Bạn có thể sửa các số này cho khớp với file Excel của bạn
metrics = ['Faithfulness', 'Answer Relevancy', 'Context Precision']
scores = [0.9452, 0.9568, 0.9120]  # Điểm số tương ứng

# Màu sắc cho các cột (Xanh dương đậm, Xanh lá, Cam - Hoặc cùng tông xanh)
colors = ['#2E86C1', '#28B463', '#D35400']

def draw_chart():
    print("🎨 Đang vẽ biểu đồ đánh giá...")

    # Tạo khung hình
    plt.figure(figsize=(10, 6)) # Kích thước 10x6 inch
    
    # Vẽ cột
    bars = plt.bar(metrics, scores, color=colors, width=0.6, edgecolor='black', alpha=0.8)

    # Trang trí trục
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.title('Lucfin RAG Performance', fontsize=16, fontweight='bold', pad=20)
    plt.ylim(0, 1.15)  # Giới hạn trục Y từ 0 đến 1.15 để chừa chỗ viết số
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # --- VIẾT SỐ LÊN ĐẦU CỘT ---
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0, # Tọa độ X (Giữa cột)
            height + 0.02,                       # Tọa độ Y (Cao hơn cột một chút)
            f'{height:.4f}',                     # Nội dung (Số làm tròn 4 chữ số)
            ha='center', va='bottom',            # Căn giữa
            fontsize=12, fontweight='bold', color='black'
        )

    # Lưu ảnh độ phân giải cao (300 DPI) để in ấn sắc nét
    output_path = os.path.join("evaluation", "rag_performance_chart.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # Hiển thị lên màn hình (nếu chạy local)
    # plt.show() 
    
    print(f"✅ Đã lưu biểu đồ đẹp tại: {output_path}")  

if __name__ == "__main__":
    draw_chart()