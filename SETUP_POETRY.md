# 🚀 GYM AGENT RAG - Hướng dẫn Poetry

## 📦 Cài đặt dependencies với Poetry

### 1. Cài đặt dependencies
```bash
# Cài đặt tất cả dependencies (bao gồm RAG)
poetry install

# Hoặc cập nhật nếu đã có env
poetry update
```

### 2. Activate virtual environment
```bash
poetry shell
```

### 3. Chạy RAG Gym Agent
```bash
# Trong poetry shell
python src/gym_agent_test/main_RAG.py

# Hoặc không cần activate shell
poetry run python src/gym_agent_test/main_RAG.py
```

### 4. Test RAG system
```bash
# Test RAG functionality
poetry run python src/gym_agent_test/test_rag.py
```

## 🔧 Dependencies đã thêm cho RAG

Các package mới được thêm vào `pyproject.toml`:

```toml
# RAG dependencies
"chromadb>=0.4.22",           # Vector database
"numpy>=1.24.3",              # Numerical operations  
"sentence-transformers>=2.2.2", # Text embeddings
"faiss-cpu>=1.7.4"            # Optional: faster similarity search
```

## 🎯 So sánh các lệnh

| Task | Command |
|------|---------|
| Install deps | `poetry install` |
| Run original | `poetry run python src/gym_agent_test/main.py` |  
| Run RAG version | `poetry run python src/gym_agent_test/main_RAG.py` |
| Test RAG | `poetry run python src/gym_agent_test/test_rag.py` |
| Add new dep | `poetry add package_name` |

## 📋 Environment setup

1. **Tạo file `.env`:**
```bash
GOOGLE_API_KEY=your_gemini_api_key_here
```

2. **Check dependencies:**
```bash
poetry show --tree
```

3. **Update specific package:**
```bash
poetry update langchain
```

## 🚨 Troubleshooting

### ChromaDB installation issues:
```bash
# If ChromaDB fails on Windows
poetry add chromadb --build
```

### Memory issues:
```bash  
# Use CPU version of faiss
poetry remove faiss-gpu
poetry add faiss-cpu
```

### Poetry not found:
```bash
# Install Poetry first
curl -sSL https://install.python-poetry.org | python3 -
```

## 🎉 Ready to go!

Sau khi `poetry install`, bạn có thể chạy:
- `main.py` - Phiên bản gốc 
- `main_RAG.py` - Phiên bản với RAG cho dinh dưỡng VN + bài tập theo nhóm cơ/chấn thương