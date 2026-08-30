"""
Unit test cho app/services/ingest_service.py.

Tập trung vào delete_document() — hàm vừa được sửa để không crash khi
xóa file gốc lỗi (VD: file bị khoá bởi chương trình khác), mà trả về
trạng thái rõ ràng cho UI biết phần nào đã xóa, phần nào chưa.
"""

from unittest.mock import patch, MagicMock

from app.services import ingest_service


def test_delete_document_full_success_when_file_exists_and_removable():
    fake_path = MagicMock()
    fake_path.exists.return_value = True

    fake_raw_dir = MagicMock()
    fake_raw_dir.__truediv__ = MagicMock(return_value=fake_path)

    with patch.object(ingest_service, "delete_source", return_value=5), \
         patch.object(ingest_service, "RAW_DATA_DIR", fake_raw_dir):

        result = ingest_service.delete_document("test.pdf")

    fake_path.unlink.assert_called_once()
    assert result["status"] == "success"
    assert result["file_deleted"] is True
    assert result["num_chunks_deleted"] == 5
    assert result["message"] is None


def test_delete_document_success_when_file_already_absent():
    """
    File gốc không tồn tại sẵn (VD đã bị xóa tay từ trước) — vẫn phải coi
    là thành công, không phải lỗi, vì "không còn gì để xóa" không phải
    tình huống bất thường.
    """
    fake_path = MagicMock()
    fake_path.exists.return_value = False

    fake_raw_dir = MagicMock()
    fake_raw_dir.__truediv__ = MagicMock(return_value=fake_path)

    with patch.object(ingest_service, "delete_source", return_value=3), \
         patch.object(ingest_service, "RAW_DATA_DIR", fake_raw_dir):

        result = ingest_service.delete_document("khong-ton-tai.pdf")

    fake_path.unlink.assert_not_called()
    assert result["status"] == "success"
    assert result["file_deleted"] is True


def test_delete_document_partial_when_file_deletion_fails():
    """
    Vector đã xóa xong (delete_source chạy trước và không lỗi), nhưng xóa
    file gốc thất bại (VD PermissionError/OSError) — phải trả về
    status="partial" kèm message rõ ràng, KHÔNG được raise exception làm
    crash UI.
    """
    fake_path = MagicMock()
    fake_path.exists.return_value = True
    fake_path.unlink.side_effect = OSError("Permission denied")

    fake_raw_dir = MagicMock()
    fake_raw_dir.__truediv__ = MagicMock(return_value=fake_path)

    with patch.object(ingest_service, "delete_source", return_value=7), \
         patch.object(ingest_service, "RAW_DATA_DIR", fake_raw_dir):

        result = ingest_service.delete_document("dang-bi-khoa.pdf")

    assert result["status"] == "partial"
    assert result["file_deleted"] is False
    assert result["num_chunks_deleted"] == 7  # vector vẫn đã xóa thành công
    assert "dang-bi-khoa.pdf" in result["message"]
    assert "Permission denied" in result["message"]