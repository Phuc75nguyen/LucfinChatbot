import time

# Lưu trữ dữ liệu Scan: { "session_id": { "foods": [...], "timestamp": ... } }
SCAN_SESSIONS = {}

# 👇👇👇 THÊM BIẾN NÀY ĐỂ QUẢN LÝ TIÊU ĐIỂM 👇👇👇
# "SCAN": Đang nói về món vừa chụp
# "RAG": Đang nói về chủ đề khác (FoodDB)
SESSION_FOCUS = {} 

def update_scan_result(session_id, food_names):
    SCAN_SESSIONS[session_id] = {
        "foods": food_names,
        "timestamp": time.time()
    }
    # Khi vừa Scan xong -> Bắt buộc Focus vào SCAN
    SESSION_FOCUS[session_id] = "SCAN"

def get_scanned_context(session_id):
    if session_id in SCAN_SESSIONS:
        data = SCAN_SESSIONS[session_id]
        # Hết hạn sau 10 phút (600s)
        if time.time() - data["timestamp"] < 600: 
            return ", ".join(data["foods"])
    return None

# 👇👇👇 2 HÀM MỚI 👇👇👇
def set_chat_focus(session_id, mode):
    """Set chế độ: 'SCAN' hoặc 'RAG'"""
    SESSION_FOCUS[session_id] = mode

def get_chat_focus(session_id):
    """Lấy chế độ hiện tại"""
    return SESSION_FOCUS.get(session_id, "RAG") # Mặc định là RAG