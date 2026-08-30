"""
Service xử lý việc nạp (ingest) tài liệu mới vào hệ thống.
Điều phối: loader -> splitter -> vectorstore.
Đây là lớp mà UI sẽ gọi, không gọi thẳng xuống core/.
"""

import logging
from pathlib import Path

from app.core.loader import load_pdf
from app.core.splitter import split_text
from app.core.vectorstore import add_chunks, delete_source, list_sources
from app.config import RAW_DATA_DIR

logger = logging.getLogger(__name__)


def ingest_pdf(file_path: str | Path, source_name: str | None = None) -> dict:
    """
    Nạp 1 file PDF vào hệ thống: đọc, chia nhỏ, lưu vào vector store.

    Args:
        file_path: đường dẫn tới file PDF cần nạp
        source_name: tên hiển thị của tài liệu (mặc định lấy tên file)

    Returns:
        dict chứa thông tin kết quả: tên file, số chunks đã tạo, trạng thái
    """
    file_path = Path(file_path)
    source_name = source_name or file_path.name

    # Kiểm tra tài liệu đã tồn tại chưa — tránh nạp trùng lặp (nhân đôi
    # chunks trong vector store) nếu user lỡ upload lại cùng 1 file.
    try:
        existing_sources = list_sources()
    except Exception as e:
        logger.warning(f"Không kiểm tra được danh sách tài liệu hiện có: {e}")
        existing_sources = []

    if source_name in existing_sources:
        return {
            "source_name": source_name,
            "num_chunks": 0,
            "num_characters": 0,
            "status": "duplicate",
            "message": (
                f"Tài liệu '{source_name}' đã tồn tại trong hệ thống. "
                "Hãy xóa tài liệu cũ trước nếu muốn nạp lại, hoặc đổi tên file."
            ),
        }

    try:
        # Bước 1: đọc PDF ra text
        text = load_pdf(file_path)

        # Bước 2: chia nhỏ thành chunks
        chunks = split_text(text)

        # Bước 3: lưu vào vector store (gọi Gemini embedding API — có thể
        # lỗi do quota/mạng, cần bắt riêng để không crash toàn bộ luồng)
        add_chunks(chunks, source_name=source_name)
    except Exception as e:
        logger.error(f"Lỗi khi nạp tài liệu '{source_name}': {e}")
        return {
            "source_name": source_name,
            "num_chunks": 0,
            "num_characters": 0,
            "status": "error",
            "message": f"Không thể nạp tài liệu. Chi tiết lỗi: {e}",
        }

    return {
        "source_name": source_name,
        "num_chunks": len(chunks),
        "num_characters": len(text),
        "status": "success",
        "message": None,
    }


def save_uploaded_file(uploaded_file_bytes: bytes, filename: str) -> Path:
    """
    Lưu file PDF user upload (từ Streamlit) xuống thư mục data/raw/.

    Args:
        uploaded_file_bytes: nội dung file dạng bytes (từ st.file_uploader)
        filename: tên file gốc

    Returns:
        Đường dẫn file đã lưu
    """
    save_path = RAW_DATA_DIR / filename
    save_path.write_bytes(uploaded_file_bytes)
    return save_path


def delete_document(source_name: str) -> dict:
    """
    Xóa 1 tài liệu khỏi hệ thống: xóa chunks trong vector store
    và xóa file gốc trong data/raw/ (nếu còn tồn tại).

    Nâng cấp: trước đây nếu xóa file gốc lỗi (VD file đang bị khoá bởi
    chương trình khác, thiếu quyền ghi...) thì exception sẽ bay thẳng lên
    UI, và vector đã bị xóa ở bước trước đó KHÔNG được hoàn tác — dữ liệu
    rơi vào trạng thái không đồng bộ (vector mất, file vẫn còn) mà người
    dùng không biết. Giờ đây bước xóa file được bọc try/except riêng: nếu
    lỗi, hàm không crash mà trả về đủ thông tin để caller (UI) biết chính
    xác phần nào đã xóa thành công, phần nào chưa.

    Args:
        source_name: tên tài liệu cần xóa (khớp với metadata "source")

    Returns:
        dict gồm:
            - num_chunks_deleted: số lượng chunks đã bị xóa khỏi vector store
            - file_deleted: True nếu file gốc đã xóa thành công (hoặc vốn
                             không tồn tại — coi như "không còn gì để xóa")
            - status: "success" | "partial" — "partial" nghĩa là vector đã
                       xóa nhưng file gốc thì chưa (cần người dùng tự xử lý)
            - message: mô tả chi tiết, chỉ có giá trị khi status="partial"
    """
    num_deleted = delete_source(source_name)

    raw_path = RAW_DATA_DIR / source_name
    file_deleted = True
    message = None

    if raw_path.exists():
        try:
            raw_path.unlink()
        except OSError as e:
            # Vector đã xóa xong ở bước trên, không thể hoàn tác — chỉ có
            # thể báo rõ cho người dùng biết để họ tự xóa file thủ công.
            logger.error(
                f"Đã xóa {num_deleted} chunks của '{source_name}' khỏi vector "
                f"store, nhưng KHÔNG xóa được file gốc: {e}"
            )
            file_deleted = False
            message = (
                f"Đã xóa dữ liệu tìm kiếm của '{source_name}', nhưng không "
                f"xóa được file gốc trong data/raw/ (có thể đang bị chương "
                f"trình khác mở, hoặc thiếu quyền ghi). Hãy tự xóa file này "
                f"thủ công. Chi tiết lỗi: {e}"
            )

    return {
        "num_chunks_deleted": num_deleted,
        "file_deleted": file_deleted,
        "status": "success" if file_deleted else "partial",
        "message": message,
    }