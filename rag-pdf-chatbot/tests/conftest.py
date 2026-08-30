"""
Cấu hình chung cho toàn bộ test suite.

Quan trọng: app/config.py raise ValueError ngay khi import nếu thiếu
GOOGLE_API_KEY. Vì unit test không cần gọi API thật (mọi lời gọi ra
Gemini đều được mock), ta set 1 API key giả TRƯỚC khi bất kỳ module nào
trong app/ được import, để import không bị crash.

sys.path cũng được thêm project root vào, để `import app...` hoạt động
dù pytest được chạy từ đâu.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("GOOGLE_API_KEY", "test-key-for-unit-tests")