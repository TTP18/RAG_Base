"""
Module tạo embedding cho văn bản, dùng Gemini embedding model.
Embedding = vector số đại diện cho ý nghĩa của đoạn text,
dùng để so sánh mức độ "giống nhau về ngữ nghĩa" giữa các đoạn.

Nâng cấp: cache lại instance embedder bằng lru_cache — trước đây mỗi lần
gọi get_embedder() sẽ khởi tạo lại client từ đầu (tốn thời gian + có thể
mở kết nối HTTP mới mỗi lần), dù config không hề đổi giữa các lần gọi.
Với Streamlit, script chạy lại (rerun) mỗi lần user tương tác, nên việc
cache này ảnh hưởng trực tiếp tới độ trễ cảm nhận được của UI.
"""

from functools import lru_cache

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import GOOGLE_API_KEY, EMBEDDING_MODEL_NAME


@lru_cache(maxsize=1)
def get_embedder() -> GoogleGenerativeAIEmbeddings:
    """
    Khởi tạo embedding model (chỉ 1 lần, các lần gọi sau lấy từ cache).
    Tách thành hàm riêng để nơi khác (vectorstore.py) chỉ cần gọi,
    không cần biết chi tiết cấu hình bên trong.

    Lưu ý: dùng lru_cache(maxsize=1) vì hàm này không có tham số — mọi
    lần gọi đều dùng chung 1 config từ app.config, nên chỉ cần giữ đúng
    1 instance duy nhất trong suốt vòng đời process.

    Returns:
        Instance của embedding model, sẵn sàng dùng (dùng chung, cached)
    """
    embedder = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
    )
    return embedder


def embed_text(text: str) -> list[float]:
    """
    Tạo embedding cho 1 đoạn text đơn lẻ.
    Dùng để test nhanh hoặc embed câu hỏi của user.

    Args:
        text: đoạn text cần embed

    Returns:
        Vector số (list các float)
    """
    embedder = get_embedder()
    vector = embedder.embed_query(text)
    return vector