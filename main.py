import uvicorn
from fastapi import FastAPI
from api.end_points import router as ask_router

# Import hàm load model để nạp trước vào RAM/VRAM
from config.rerank import load_reranker

app = FastAPI(
    title="RAG Lucfin QA",
    description="API truy vấn tài liệu dinh dưỡng, tích hợp Computer Vision & RAG thông minh",
    version="1.0.0"
)

# --- SỰ KIỆN KHỞI ĐỘNG (WARM-UP) ---
# Chạy ngay khi Server bật, giúp request đầu tiên không bị chậm
@app.on_event("startup")
async def startup_event():
    print("🚀 Server đang khởi động: Pre-loading Models...")
    try:
        load_reranker() # Nạp model Cross-Encoder vào GPU ngay lập tức
        print("✅ Model Re-ranker đã sẵn sàng trong VRAM!")
    except Exception as e:
        print(f"⚠️ Cảnh báo: Không thể nạp trước Reranker: {e}")

# --- ĐĂNG KÝ ROUTER ---
# Đưa toàn bộ logic từ api/end_points.py vào App
app.include_router(ask_router, prefix="")

# --- HEALTH CHECK ---
@app.get("/ping")
async def ping():
    return {"message": "pong", "status": "Server is running"}

# --- ENTRY POINT ---
if __name__ == "__main__":
    # Chạy server tại 0.0.0.0 để Android Emulator hoặc thiết bị khác trong LAN gọi được
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)