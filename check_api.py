import os
from dotenv import load_dotenv

print("🔍 ĐANG KIỂM TRA MÔI TRƯỜNG...")

# 1. Thử load file .env
is_loaded = load_dotenv()
if is_loaded:
    print("✅ Đã tìm thấy và load file .env")
else:
    print("❌ KHÔNG tìm thấy file .env (Kiểm tra lại xem bạn có đặt nhầm tên là .env.txt không?)")

# 2. Kiểm tra các biến môi trường thường dùng
keys_to_check = ["MY_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY"]
found_any = False

for key_name in keys_to_check:
    value = os.getenv(key_name)
    if value:
        masked_value = f"{value[:5]}...{value[-4:]}" if len(value) > 10 else "***"
        print(f"✅ Tìm thấy {key_name}: {masked_value}")
        found_any = True
    else:
        print(f"⚪ Không có {key_name}")

if not found_any:
    print("\n⚠️  CẢNH BÁO: Không tìm thấy bất kỳ API Key nào! Hãy mở file .env và kiểm tra lại tên biến.")
else:
    print("\n💡 Gợi ý: Hãy mở file 'config/llm.py' xem code đang gọi tên biến nào (ví dụ: os.getenv('MY_API_KEY')) và sửa file .env cho khớp.")