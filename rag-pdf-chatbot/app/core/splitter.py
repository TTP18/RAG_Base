"""
Module chia nhỏ văn bản thành các đoạn (chunks).
Dùng RecursiveCharacterTextSplitter: ưu tiên cắt theo đoạn văn,
câu, rồi mới tới ký tự, giúp giữ ngữ nghĩa tốt hơn cắt cứng.

Nâng cấp: CHUNK_SIZE/CHUNK_OVERLAP trong config.py là số ĐẾM THEO Ý ĐỊNH
LÀ TOKEN, nhưng trước đây length_function=len lại đếm theo KÝ TỰ. Với
tiếng Việt có dấu (ký tự Unicode tổ hợp, dấu thanh...), tỷ lệ ký tự/token
lệch khá xa so với tiếng Anh — chunk "1000 ký tự" có thể tương ứng với
số token nhiều hơn hoặc ít hơn dự tính khá nhiều, ảnh hưởng tới việc
nhồi context vào prompt (dễ vượt giới hạn ngầm hoặc chunk quá nhỏ).

Giải pháp: đếm độ dài theo token thật bằng tiktoken (cùng họ mã hoá
cl100k_base mà hầu hết mô hình hiện đại dùng — không chính xác 100% với
tokenizer riêng của Gemini, nhưng ước lượng sát hơn nhiều so với len()).
Nếu tiktoken không cài được (offline, thiếu mạng...), tự động rơi về
đếm theo ký tự như cũ để không làm gãy pipeline.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import CHUNK_SIZE, CHUNK_OVERLAP

try:
    import tiktoken

    _ENCODING = tiktoken.get_encoding("cl100k_base")

    def _token_length(text: str) -> int:
        return len(_ENCODING.encode(text))

except ImportError:
    # Fallback: không có tiktoken thì đếm theo ký tự (hành vi cũ)
    _token_length = len


def split_text(text: str) -> list[str]:
    """
    Chia văn bản dài thành các đoạn nhỏ có độ dài giới hạn.

    Độ dài (CHUNK_SIZE, CHUNK_OVERLAP) được tính theo TOKEN (qua tiktoken)
    thay vì ký tự thô, để khớp sát hơn với giới hạn thật của embedding
    model, đặc biệt quan trọng với văn bản tiếng Việt.

    Args:
        text: văn bản gốc (thường lấy từ loader.py)

    Returns:
        Danh sách các đoạn text nhỏ (chunks)
    """
    if not text or not text.strip():
        raise ValueError("Text đầu vào rỗng, không thể chia nhỏ")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # thứ tự ưu tiên cắt
        length_function=_token_length,
    )

    chunks = splitter.split_text(text)

    return chunks