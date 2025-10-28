# GYM AGENT RAG - Hướng dẫn sử dụng

## 🚀 Tính năng mới của RAG Gym Agent

### 📋 Cài đặt
```bash
pip install -r requirements_rag.txt
```

### 🔧 Thiết lập
1. Tạo file `.env` với `GOOGLE_API_KEY=your_api_key`
2. Chạy: `python src/gym_agent_test/main_RAG.py`

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
- Lưu lịch sử 10 cuộc hội thoại gần nhất
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

### Profile Tracking:
- Height, Weight, BMI
- Goals (tăng cơ, giảm cân, etc.)
- **NEW:** Injuries (vai, tay, lưng, gối, etc.)

## 🚀 Upgrade path

Từ bản gốc sang bản RAG:
1. Cài thêm dependencies RAG
2. Thay `main.py` → `main_RAG.py`
3. Enjoy RAG features! 🎉