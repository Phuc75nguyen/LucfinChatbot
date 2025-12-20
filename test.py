import requests
import json

# Cấu hình
BASE_URL = "http://127.0.0.1:8000"
HEADERS = {"Content-Type": "application/json"}
SESSION_ID = "test_device_vip_pro" # ID phiên test

def print_response(step_name, response):
    print(f"\n--- {step_name} ---")
    if response.status_code == 200:
        print("✅ SUCCESS!")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    else:
        print(f"❌ FAILED: {response.status_code}")
        print(response.text)

# 1. Test API Scan (Giả lập Camera gửi Sườn + Tofu lên)
print("🚀 BẮT ĐẦU TEST...")
scan_payload = {
    "session_id": SESSION_ID,
    "detected_classes": ["Suon", "Tofu"] 
}
resp = requests.post(f"{BASE_URL}/scan", json=scan_payload, headers=HEADERS)
print_response("BƯỚC 1: GỬI KẾT QUẢ SCAN", resp)

# 2. Test API Ask (Hỏi trống không để xem có nhận ra Sườn Tofu không)
ask_payload = {
    "question": "Hai món này ăn chung có hợp không?", # Không nhắc tên món
    "session_id": SESSION_ID
}
resp = requests.post(f"{BASE_URL}/ask", json=ask_payload, headers=HEADERS)
print_response("BƯỚC 2: HỎI VỀ MÓN VỪA SCAN (Context Injection)", resp)

# 3. Test Router (Hỏi chuyện phiếm)
chat_payload = {
    "question": "Bạn bao nhiêu tuổi rồi?",
    "session_id": SESSION_ID
}
resp = requests.post(f"{BASE_URL}/ask", json=chat_payload, headers=HEADERS)
print_response("BƯỚC 3: TEST CHITCHAT (Router)", resp)