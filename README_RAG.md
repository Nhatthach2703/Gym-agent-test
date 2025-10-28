# GYM AGENT RAG - Hướng dẫn sử dụng

## 🚀 Tính năng mới của RAG Gym Agent

### 📋 Cài đặt (Poetry)
```bash
# Cài dependencies bằng Poetry
poetry install

# Nếu log báo thiếu thư viện, cài bổ sung (khớp với main_RAG.py)
poetry add \
  langchain langchain-core langchain-community \
  langchain-google-genai google-generativeai \
  chromadb python-dotenv rich
```

### 🔧 Thiết lập
1. Tạo file `.env` với `GOOGLE_API_KEY=your_api_key`
2. Chạy bằng Poetry:
```bash
poetry run python src/gym_agent_test/main_RAG.py
```

## 🎯 Tính năng RAG mới

### 1. 🍜 RAG Dinh dưỡng món ăn Việt Nam
- **Database:** 20+ món ăn phổ biến với thông tin chi tiết
- **Thông tin:** Calories, protein, carbs, fat, lợi ích sức khỏe
- **Ví dụ câu hỏi:**
  - "Phở bò có bao nhiêu calories?"
  - "Món nào phù hợp để tăng cơ?"
  - "Ăn gì trước khi tập gym?"
  - "Cơm tấm có tốt cho giảm cân không?"

### 2. 💪 RAG Bài tập theo nhóm cơ và chấn thương
- **Database:** 30+ bài tập theo 7 nhóm cơ chính
- **Bao gồm:** Ngực, lưng, vai, tay, chân, bụng + bài tập phục hồi
- **Ví dụ câu hỏi:**
  - "Bài tập ngực tốt nhất là gì?"
  - "Tôi bị đau vai, nên tập gì?"
  - "Chấn thương gối thì làm sao?"
  - "Bài tập nào tăng sức mạnh chân?"

### 3. 🧠 Conversation Memory + Profile
- Nhớ thông tin cá nhân: chiều cao, cân nặng, mục tiêu
- **MỚI:** Theo dõi chấn thương và đưa ra tư vấn phù hợp
- Lưu lịch sử 30 cuộc hội thoại gần nhất
- Gợi ý câu hỏi dựa trên profile

## 📊 So sánh với phiên bản gốc

| Tính năng | Bản gốc | Bản RAG |
|-----------|---------|---------|
| Tính BMI | ✅ | ✅ |
| Tư vấn gym cơ bản | ✅ | ✅ |
| Dinh dưỡng món VN | ❌ | ✅ RAG 20+ món |
| Bài tập theo nhóm cơ | ❌ | ✅ RAG 30+ bài tập |
| Tư vấn chấn thương | ❌ | ✅ RAG + Profile |
| Độ chính xác | Tốt | Rất tốt (RAG) |
| Conversation Memory | ✅ | ✅ Nâng cao |

## 🎮 Demo câu hỏi mẫu

### Về dinh dưỡng:
```
"Tôi cao 1.75m, nặng 70kg, muốn tăng cơ. Nên ăn món gì?"
"Phở bò có phù hợp với người tập gym không?"
"So sánh calories giữa bún bò Huế và phở?"
```

### Về bài tập:
```
"Tôi bị đau vai, nên tập bài tập nào?"
"Bài tập ngực nào hiệu quả nhất?"
"Chấn thương gối thì có thể tập chân được không?"
```

### Tích hợp:
```
"Tôi cao 1.70m, nặng 80kg, muốn giảm cân và bị đau lưng. Tư vấn cho tôi?"
```

## 🔧 Cấu trúc RAG

### Vector Stores:
- **nutrition_vectorstore:** Chứa thông tin 20+ món ăn VN
- **exercise_vectorstore:** Chứa 30+ bài tập + phục hồi chấn thương

### RAG Tools:
1. `nutrition_advisor_rag(query)` - Tư vấn dinh dưỡng
2. `exercise_advisor_rag(query)` - Tư vấn bài tập
3. `calc_bmi(height_weight)` - Tính BMI
4. `gym_advice_tool(question)` - Backup tư vấn tổng quát

### Model Config (theo main_RAG.py)
```python
# LLM
ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# Embeddings
GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
```

### Profile Tracking:
- Height, Weight, BMI
- Goals (tăng cơ, giảm cân, etc.)
- **NEW:** Injuries (vai, tay, lưng, gối, etc.)

## 🚀 Upgrade path

Từ bản gốc sang bản RAG:
1. `poetry install`
2. Tạo `.env` với `GOOGLE_API_KEY`
3. Chạy: `poetry run python src/gym_agent_test/main_RAG.py`
4. Enjoy RAG features! 🎉

## 🛠️ Xử lý lỗi onnxruntime khi `poetry install`

Nếu gặp lỗi dạng:

```
Installing onnxruntime (1.23.2): Failed
Unable to find installation candidates for onnxruntime
```

Thử lần lượt các cách sau (dừng khi thành công):

1) Regenerate lockfile (sạch cache)
```bash
poetry lock --no-cache --regenerate
poetry install
```

2) Cập nhật riêng gói onnxruntime
```bash
poetry update --no-cache onnxruntime
```

3) Pin về phiên bản tương thích (khuyên dùng CPU):
```bash
# Windows/macOS (x86_64, ARM) thường ổn định ở 1.18.0
poetry add onnxruntime==1.18.0
poetry install
```

4) Windows dùng GPU (DirectML) thì dùng bản phù hợp:
```bash
poetry add onnxruntime-directml==1.17.1
```

Ghi chú:
- Dự án này dùng Google Embeddings (models/text-embedding-004), nên onnxruntime chỉ là phụ thuộc gián tiếp (từ chromadb). Pin phiên bản ổn định thường giải quyết được.
- Nếu vẫn lỗi, chạy lệnh với `-v` để xem log chi tiết và kiểm tra wheel tương thích với OS/CPU/Python.