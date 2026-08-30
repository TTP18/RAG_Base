"""
Module đọc file PDF, trích xuất text thô.
Chỉ làm 1 việc: input là đường dẫn file, output là text.
Không xử lý chunking, không liên quan LLM.
"""

from pathlib import Path
from pypdf import PdfReader


def load_pdf(file_path: str | Path) -> str:
    """
    Đọc 1 file PDF và trả về toàn bộ text bên trong.

    Args:
        file_path: đường dẫn tới file PDF

    Returns:
        Chuỗi text trích xuất từ toàn bộ các trang PDF

    Raises:
        FileNotFoundError: nếu file không tồn tại
        ValueError: nếu file không phải PDF hoặc rỗng
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    if file_path.suffix.lower() != ".pdf":
        raise ValueError(f"File không phải PDF: {file_path}")

    reader = PdfReader(file_path)

    text_parts = []
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)

    full_text = "\n".join(text_parts)

    if not full_text.strip():
        raise ValueError(f"Không trích xuất được text nào từ file: {file_path}")

    return full_text