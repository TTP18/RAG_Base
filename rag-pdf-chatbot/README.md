Project 1: RAG chatbot hỏi đáp PDF (làm đầu tiên)

Ý tưởng: Upload 1 file PDF (giáo trình, CV, tài liệu công ty) → chatbot trả lời câu hỏi dựa trên nội dung đó.

Stack gợi ý (dễ tiếp cận):

Python + LangChain hoặc LlamaIndex
OpenAI API hoặc Claude API để embedding + generate
ChromaDB (vector store, chạy local, không cần setup phức tạp)
Streamlit làm giao diện (code ít, lên demo nhanh)

Tại sao nên làm: Đây là project RAG "chuẩn" nhất, nhà tuyển dụng nhìn phát hiểu ngay bạn nắm được embedding, vector search, prompt engineering là gì.

Ghi vào CV kiểu: "Xây dựng RAG chatbot hỏi đáp tài liệu PDF sử dụng LangChain + ChromaDB, đạt độ chính xác X% trên bộ test tự tạo"

----------------------------------------------------------------------------------------------------

# 📄 RAG PDF Chatbot

Một hệ thống RAG (Retrieval-Augmented Generation) hoàn chỉnh cho phép upload tài liệu PDF và đặt câu hỏi — chatbot sẽ trả lời dựa trên nội dung tài liệu, kèm trích dẫn nguồn tham khảo.

## Tính năng chính

- Upload và xử lý tài liệu PDF (đọc, chia nhỏ, tạo embedding tự động)
- Hỏi đáp bằng ngôn ngữ tự nhiên dựa trên nội dung tài liệu
- Hiển thị nguồn tham khảo cho mỗi câu trả lời (giúp người dùng kiểm chứng)
- Chống hallucination: hệ thống từ chối trả lời khi không tìm thấy thông tin liên quan, thay vì bịa đặt
- Điều chỉnh số lượng đoạn văn bản dùng để trả lời (top-k) qua giao diện
- Lọc phạm vi tìm kiếm theo tài liệu cụ thể (multi-document filtering)
- Query rewriting: sinh nhiều biến thể câu hỏi để tăng khả năng tìm đúng ngữ cảnh, kèm **rerank theo độ liên quan thật** trước khi đưa vào prompt (không chỉ nối kết quả theo thứ tự biến thể)
- Chunking theo **số token thực tế** (qua tiktoken) thay vì đếm ký tự thô — quan trọng với tiếng Việt vì tỷ lệ ký tự/token lệch khá xa so với tiếng Anh
- Chống nạp trùng tài liệu: cảnh báo nếu tài liệu đã tồn tại trong hệ thống thay vì tạo dữ liệu nhân đôi
- Xử lý lỗi mềm: khi Gemini API lỗi (quá tải, mất mạng...) hệ thống báo thông báo thân thiện thay vì crash; khi thiếu API key, giao diện hướng dẫn cách khắc phục thay vì hiện traceback thô
- Cache embedding model & kết nối vector store (`lru_cache`) để tránh khởi tạo lại kết nối mỗi lần gọi, giúp giao diện phản hồi nhanh hơn giữa các lần tương tác
- Bộ **unit test tự động** (pytest) cho toàn bộ logic lõi — chunking, rerank/dedupe, parsing, xử lý lỗi — chạy nhanh, không tốn quota API

## Kiến trúc hệ thống

```
PDF Upload
   │
   ▼
Loader (trích xuất text) ──► Splitter (chia nhỏ theo token)
                                      │
                                      ▼
                         Embedder (tạo vector embedding, cached)
                                      │
                                      ▼
                    ChromaDB (lưu trữ vector, persistent, cached)

Câu hỏi người dùng
   │
   ▼
Semantic Search (tìm đoạn liên quan nhất, có rerank khi dùng query rewriting)
   │
   ▼
Prompt Template (ghép context + câu hỏi gốc)
   │
   ▼
LLM (sinh câu trả lời dựa trên context)
   │
   ▼
Trả lời + Nguồn tham khảo
```

### Cấu trúc thư mục

```
rag-pdf-chatbot/
├── app/
│   ├── config.py              # Cấu hình trung tâm (model, chunk size, đường dẫn...)
│   ├── core/                  # Logic lõi, độc lập với UI
│   │   ├── loader.py          # Đọc file PDF → text
│   │   ├── splitter.py        # Chia nhỏ văn bản theo token (tiktoken)
│   │   ├── embedder.py        # Tạo embedding vector (cached)
│   │   ├── vectorstore.py     # Lưu trữ & tìm kiếm ngữ nghĩa (ChromaDB, cached)
│   │   └── llm.py             # Gọi LLM sinh câu trả lời + query rewriting
│   ├── chains/
│   │   └── qa_chain.py        # Pipeline RAG hoàn chỉnh (retrieve → rerank → generate)
│   ├── services/               # Lớp điều phối giữa core và UI
│   │   ├── ingest_service.py  # Xử lý nạp tài liệu mới
│   │   └── chat_service.py    # Xử lý câu hỏi từ người dùng
│   └── utils/                  # (dự phòng) hàm tiện ích dùng chung, hiện chưa cần dùng đến
├── ui/
│   └── streamlit_app.py       # Giao diện web
├── notebooks/
│   └── experiment.ipynb       # Sổ tay thử nghiệm nhanh (chunking, prompt, embedding...)
├── data/
│   ├── raw/                   # File PDF gốc đã upload
│   └── vectorstore/           # ChromaDB (persistent)
├── tests/                     # Unit test tự động (pytest)
│   ├── conftest.py            # Cấu hình chung: fake API key, sys.path
│   ├── test_splitter.py       # Chunking: rỗng/ngắn/dài, giữ thứ tự, không vỡ Unicode
│   ├── test_qa_chain.py       # Rerank theo score, dedupe, giới hạn top_k, prompt dùng câu hỏi gốc
│   ├── test_llm.py            # Chuẩn hoá response LLM, parse/dedupe query rewriting
│   ├── test_vectorstore.py    # Build filter theo nguồn, metadata chunk, dedupe danh sách nguồn
│   └── test_services.py       # Input rỗng, không rò rỉ lỗi kỹ thuật ra UI, truncate text
├── pytest.ini                 # Cấu hình pytest (testpaths, pattern file test)
├── requirements.txt
└── README.md
```

## Công nghệ sử dụng

| Thành phần | Công nghệ |
|---|---|
| Ngôn ngữ | Python 3.12 |
| LLM | Google Gemini (`gemini-flash-latest`) |
| Embedding | Google Gemini Embedding (`gemini-embedding-001`) |
| Vector Database | ChromaDB (local, persistent) |
| Framework RAG | LangChain |
| Giao diện | Streamlit |
| Xử lý PDF | pypdf |
| Đếm token khi chunking | tiktoken |
| Kiểm thử | pytest + unittest.mock |

## Cài đặt

### 1. Clone repository

```bash
git clone <repo-url>
cd rag-pdf-chatbot
```

### 2. Tạo virtual environment (khuyến nghị)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 4. Cấu hình API key

Lấy Google API Key miễn phí tại [Google AI Studio](https://aistudio.google.com/app/apikey).

Copy `.env.example` thành `.env` và điền key:

```
GOOGLE_API_KEY=your_actual_api_key_here
```

### 5. Chạy ứng dụng

```bash
streamlit run ui/streamlit_app.py
```

Ứng dụng sẽ mở tại `http://localhost:8501`.

## Cách sử dụng

1. Mở giao diện web, dùng sidebar bên trái để upload file PDF
2. Bấm "Nạp tài liệu vào hệ thống" — chờ xử lý (đọc, chia nhỏ, tạo embedding)
3. Gõ câu hỏi vào ô chat bên dưới
4. Xem câu trả lời kèm nguồn tham khảo (bấm vào phần mở rộng để xem chi tiết)

## Kiểm thử

Bộ unit test dùng `pytest`, mock toàn bộ lời gọi ra Gemini API và ChromaDB thật — chạy nhanh, không tốn quota, không cần file PDF hay `.env` thật (dùng API key giả trong `conftest.py`).

```bash
pytest -v
```

Test tập trung vào phần **logic mình tự viết**, không test lại các thư viện bên ngoài (LangChain, Chroma, Gemini SDK). Trọng tâm là:

- **`test_qa_chain.py`**: đảm bảo kết quả từ nhiều biến thể câu hỏi (query rewriting) được **rerank theo điểm liên quan thật**, không ưu tiên câu hỏi đầu tiên theo thứ tự — đây là phần logic tinh tế nhất trong pipeline.
- **`test_splitter.py`**: chunking không làm vỡ ký tự Unicode tiếng Việt, giữ đúng thứ tự nội dung.
- **`test_llm.py`**: parsing output của LLM (khi sinh biến thể câu hỏi) không bị vỡ khi LLM trả về format không như mong đợi, và không làm crash pipeline khi LLM lỗi.
- **`test_vectorstore.py`** / **`test_services.py`**: xử lý lỗi mềm, không để lỗi kỹ thuật thô rò rỉ ra giao diện người dùng.

## Đánh giá chất lượng hệ thống

Hệ thống được kiểm thử với bộ câu hỏi đa dạng trên một tài liệu nghiên cứu khoa học kỹ thuật (báo cáo NCKH sinh viên, chủ đề mạng vô tuyến nhận thức và kỹ thuật NOMA), nhằm đánh giá cả khả năng trả lời đúng lẫn khả năng từ chối hợp lý.

| Loại câu hỏi | Ví dụ | Kết quả |
|---|---|---|
| Tra cứu trực tiếp | "Mã số đề tài là gì?" | ✅ Trả lời chính xác |
| Trích xuất công thức phức tạp | "Phương trình tính SINR tại U1, U2?" | ✅ Trích xuất đúng công thức toán học |
| Câu hỏi "bẫy" — thông tin không tồn tại | "Đề tài nộp vào hội nghị/tạp chí nào?" | ✅ Từ chối trả lời, không bịa đặt |
| Câu hỏi "bẫy" — thông tin không tồn tại | "Kinh phí thực hiện đề tài là bao nhiêu?" | ✅ Từ chối trả lời, không bịa đặt |
| Suy luận từ dữ kiện có thật | "Có bao nhiêu sinh viên tham gia đề tài?" | ✅ Suy luận hợp lý (1 sinh viên) từ thông tin "chủ nhiệm đề tài" |

**Nhận xét:** Điểm mạnh nhất của hệ thống là khả năng **chống hallucination** — khi thông tin không có trong tài liệu, hệ thống trả lời trung thực thay vì bịa đặt, nhờ việc ràng buộc rõ trong prompt template rằng câu trả lời phải bám sát ngữ cảnh được cung cấp.

## Hạn chế hiện tại & hướng cải thiện

- **Retrieval đôi khi bỏ sót thông tin rải rác:** với câu hỏi mà thông tin nằm rải rác nhiều nơi trong tài liệu (thay vì tập trung 1 đoạn), similarity search cơ bản đôi khi không tìm đủ ngữ cảnh cần thiết — dù đã có query rewriting hỗ trợ, một số trường hợp vẫn cần thêm.
  → Hướng cải thiện: hybrid search (kết hợp keyword search + semantic search), hoặc tăng `top_k` khi cần.

- **Query rewriting tốn thêm chi phí gọi API:** mỗi câu hỏi bật rewriting sẽ tốn thêm 1 lần gọi LLM để sinh biến thể, cộng thêm nhiều lượt search hơn — tăng độ trễ và tiêu tốn quota, nên hiện để tùy chọn bật/tắt qua giao diện chứ không bật mặc định.

- **Chưa có bộ đánh giá định lượng (RAG evaluation):** hiện tại đánh giá thủ công qua test case, chưa có metric như faithfulness, relevance đo tự động.
  → Hướng cải thiện: tích hợp framework như RAGAS để benchmark hệ thống một cách khách quan.

- **Chưa có CI tự động:** test suite hiện chạy thủ công qua `pytest`, chưa tích hợp vào pipeline CI (VD: GitHub Actions) để tự chạy khi push code.
  → Hướng cải thiện: thêm workflow CI đơn giản chạy `pytest` mỗi lần push/pull request.

## Tác giả

Made with ❤️ as a portfolio project — RAG fundamentals: chunking, embedding, semantic search, prompt engineering, hallucination prevention, testing.