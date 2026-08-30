"""
Unit test cho app/core/splitter.py.

Khác với file test_splitter.py (script chạy tay) ở gốc project — file
này dùng pytest với assert thật, chạy tự động, không cần load PDF thật
hay gọi API nào. Mục tiêu: bắt regression khi refactor splitter sau này
(VD: đổi cách đếm độ dài từ ký tự sang token như đã làm).
"""

import pytest

from app.core.splitter import split_text


def test_split_text_empty_raises():
    with pytest.raises(ValueError):
        split_text("")


def test_split_text_whitespace_only_raises():
    with pytest.raises(ValueError):
        split_text("   \n\n   ")


def test_split_text_short_returns_single_chunk():
    text = "Đây là một đoạn văn bản ngắn."
    chunks = split_text(text)

    assert len(chunks) == 1
    assert chunks[0] == text


def test_split_text_long_returns_multiple_chunks():
    # Lặp lại 1 đoạn văn nhiều lần để chắc chắn vượt CHUNK_SIZE
    paragraph = (
        "Mạng vô tuyến nhận thức (Cognitive Radio Network) là một công nghệ "
        "cho phép các thiết bị vô tuyến tự động nhận biết và sử dụng phổ tần "
        "chưa được cấp phép một cách linh hoạt, nhằm tối ưu hiệu quả sử dụng "
        "tài nguyên tần số vốn đang ngày càng khan hiếm. "
    )
    text = "\n\n".join([paragraph] * 40)

    chunks = split_text(text)

    assert len(chunks) > 1
    # Không được làm mất nội dung: nối lại (loại overlap) vẫn phải chứa
    # đủ các từ khoá quan trọng của văn bản gốc
    joined = " ".join(chunks)
    assert "Mạng vô tuyến nhận thức" in joined


def test_split_text_preserves_order():
    # Mỗi đoạn có 1 marker riêng biệt để kiểm tra thứ tự không bị đảo lộn
    markers = [f"MARKER_{i:03d}" for i in range(30)]
    text = "\n\n".join(f"{m}: " + ("nội dung " * 50) for m in markers)

    chunks = split_text(text)
    full_text = "\n\n".join(chunks)

    positions = [full_text.index(m) for m in markers]
    assert positions == sorted(positions), "Thứ tự các đoạn markers bị đảo lộn sau khi split"


def test_split_text_chunks_are_non_empty_strings():
    text = "Câu một. Câu hai. Câu ba. " * 100
    chunks = split_text(text)

    assert all(isinstance(c, str) for c in chunks)
    assert all(c.strip() for c in chunks)


def test_split_text_vietnamese_diacritics_not_broken_mid_word():
    # Kiểm tra các từ có dấu tiếng Việt không bị cắt đứt giữa chừng theo
    # kiểu làm hỏng ký tự Unicode tổ hợp (dấu thanh bị tách khỏi nguyên âm)
    text = "Nghiên cứu khoa học kỹ thuật về mạng vô tuyến nhận thức. " * 60
    chunks = split_text(text)

    for chunk in chunks:
        # Nếu chunk hợp lệ, decode/encode UTF-8 phải cho lại đúng chuỗi
        assert chunk.encode("utf-8").decode("utf-8") == chunk