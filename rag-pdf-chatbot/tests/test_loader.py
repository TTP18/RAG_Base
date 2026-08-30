from app.core.loader import load_pdf

text = load_pdf("data/raw/SVC2025-002-IN.pdf")
print("Số ký tự trích xuất:", len(text))
print("--- 500 ký tự đầu ---")
print(text[:500])