"""
Cấu hình trung tâm cho toàn bộ project.
Mọi module khác chỉ import từ đây, không tự đọc os.environ ở nơi khác.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load biến môi trường từ file .env
load_dotenv()

# ==== Đường dẫn thư mục (tự tính, không hardcode) ====
BASE_DIR = Path(__file__).resolve().parent.parent  # gốc project
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"

# Tự tạo thư mục nếu chưa có (tránh lỗi khi chạy lần đầu)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

# ==== API Keys ====
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY chưa được thiết lập. "
        "Hãy tạo file .env và điền GOOGLE_API_KEY=your_key"
    )

# ==== Model config ====
LLM_MODEL_NAME = "models/gemini-flash-latest"      # model dùng để trả lời
EMBEDDING_MODEL_NAME = "models/gemini-embedding-001"  # model dùng để tạo embedding

# ==== Chunking config ====
CHUNK_SIZE = 1000        # số ký tự mỗi đoạn
CHUNK_OVERLAP = 150      # số ký tự chồng lặp giữa các đoạn (giữ ngữ cảnh)

# ==== Retrieval config ====
TOP_K_RESULTS = 4        # số đoạn liên quan nhất lấy ra mỗi lần hỏi

# ==== Vectorstore config ====
COLLECTION_NAME = "pdf_documents"