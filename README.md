# 🤖 Gym Agent - AI Personal Trainer

An intelligent AI Agent using **Langchain Framework** and **Gemini API** for gym consultation, BMI calculation, and personalized workout planning.

![Gym Agent Demo](https://img.shields.io/badge/AI-Agent-blue) ![Langchain](https://img.shields.io/badge/Langchain-Framework-green) ![Gemini](https://img.shields.io/badge/Gemini-API-orange) ![Python](https://img.shields.io/badge/Python-3.10%2B-blue)

## 🎯 Key Features

### 🛠️ **AI Agent with Tools:**

- **`calc_bmi`** - BMI calculation tool with health status classification
- **`gym_advice_tool`** - Gym workout advice, nutrition and training plan consultation tool
- **Intelligent Tool Selection** - Agent automatically selects appropriate tools
- **Conversation Memory** - Stores conversation context and history

### 🧠 **Agent Architecture:**

```
Langchain Agent (create_tool_calling_agent)
├── ChatPromptTemplate + MessagesPlaceholder
├── Tool Selection Logic  
├── BMI Calculator Tool
├── Gym Advice Tool
└── Gemini AI Fallback
```

## 🚀 Setup and Installation

### **Step 1: Clone Repository**

```bash
git clone https://github.com/Nhatthach2703/Gym-agent-test.git
cd Gym-agent-test
```

### **Step 2: Install Poetry**

```bash
# Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -

# macOS/Linux
curl -sSL https://install.python-poetry.org | python3 -
```

### **Step 3: Install Dependencies (Poetry)**

```bash
# Install packages
poetry install

# Check installed dependencies
poetry show

# If logs show missing libs for RAG, add explicitly (matches main_RAG.py)
poetry add
  langchain langchain-core langchain-community \
  langchain-google-genai google-generativeai \
  chromadb python-dotenv rich

# If install fails for onnxruntime (platform wheel not found):
# Try step-by-step and stop when success
# 1) Regenerate lockfile
poetry lock --no-cache --regenerate && poetry install
# 2) Update specific package
poetry update --no-cache onnxruntime
# 3) Pin a stable version
poetry add onnxruntime==1.18.0
# 4) (Windows GPU) DirectML variant
poetry add onnxruntime-directml==1.17.1
```

### **Step 4: Setup Gemini API Key**

#### 4.1. Get API Key from Google AI Studio

1. Visit: <https://makersuite.google.com/app/apikey>
2. Sign in with your Google Account
3. Create a new API Key
4. Copy the API Key

#### 4.2. Create `.env` file

```bash
# Create .env file in root directory
touch .env

# Or manually create .env file with content:
```

#### 4.3. Add API Key to `.env`

```env
# File: .env
GOOGLE_API_KEY=your_gemini_api_key_here

# Example:
# GOOGLE_API_KEY=bvhvbfhvbfvbhvbfhdvbdfhvbdhfvbsh
```

### **Step 5: Configure Model (Optional)**

In `src/gym_agent_test/main.py`, you can change the model:

```python
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # Can change to: gemini-1.5-flash, gemini-pro
    api_key=api_key,
    temperature=0.7  # Adjust creativity (0.0-1.0)
)
```

## 🎮 Usage

### **Run the Agent (Poetry):**

```bash
# Agent cơ bản
poetry run python src/gym_agent_test/main.py

# Agent RAG
poetry run python src/gym_agent_test/main_RAG.py
```

### **RAG Execution Flow (theo main_RAG.py)**
1) Kết nối LLM Gemini (`gemini-2.5-flash`)
2) Khởi tạo embeddings `models/text-embedding-004`
3) Tạo `Document` và 2 vector stores (`nutrition`, `exercises`)
4) Tạo Agent với tools: `calc_bmi`, `nutrition_advisor_rag`, `exercise_advisor_rag`, `gym_advice_tool`
5) Vòng lặp chat với memory (tối đa 30 hội thoại)

### **Test Commands:**

#### **1. Calculate BMI:**

```
👤 You: Calculate my BMI 1.75,70
🤖 Agent: Will call tool calc_bmi("1.75,70")
💬 PT Nam: BMI = 22.86 → Normal
```

#### **2. Workout Advice:**

```
👤 You: Suggest arm muscle building exercises
🤖 Agent: Will call tool gym_advice_tool("arm muscle building") 
💬 PT Nam: 💪 Arm exercises:
- Push-ups, pull-ups, dips
- Bicep curls, tricep extensions
- Overhead press
```

#### **3. Smart Q&A:**

```
👤 You: How to get 6-pack abs?
🤖 Agent: Will use Gemini AI reasoning
💬 PT Nam: 💪 To get 6-pack abs, you need to combine...
```

#### **4. Exit:**

```
👤 You: exit
🙋‍♂️ Thank you for using! See you again!
```

## 📁 Project Structure

```
Gym-agent-test/
├── src/gym_agent_test/
│   ├── __init__.py
│   ├── main.py                    # 🏆 Main Agent (Recommended)
│   ├── main_simple.py            # 🎯 Simple Version
│   ├── main_with_tools.py        # 🛠️ Custom Agent Class
│   └── demo_langchain_prompts.py # 🧪 Demo & Test
├── .env                          # ⚠️ API Keys (create manually)
├── .gitignore
├── pyproject.toml               # Dependencies config  
├── poetry.lock                  # Lock file
└── README.md                    # This guide
```

## ⚙️ Dependencies Configuration

### **pyproject.toml:**

```toml
[project]
name = "gym-agent-test"
version = "0.1.0"
description = "Gym Agent với Langchain và Gemini API"
requires-python = ">=3.10,<4.0"

dependencies = [
    "langchain>=0.3.0",
    "langchain-core>=0.3.0", 
    "python-dotenv>=1.0.0",
    "langchain-google-genai>=2.0.0",
    "langchain-community>=0.3.0",
    "langchainhub>=0.1.0",
    "langchain-experimental>=0.3.0"
]
```

## 🔧 Troubleshooting

### **❌ Error "ModuleNotFoundError":**

```bash
# Reinstall dependencies
poetry install

# Or update
poetry update
```

### **❌ Error "404 models/gemini-xxx not found":**

Change model in `main.py`:

```python
# From:
model="gemini-2.5-flash"  

# To:
model="gemini-1.5-flash"  # or "gemini-pro"
```

### **❌ Error "API Key invalid":**

1. Check `.env` file has correct format
2. Verify API Key at: <https://makersuite.google.com/app/apikey>
3. Ensure no extra spaces in API Key

### **❌ Error "langchain_core.prompts not found":**

```bash
# Install additional langchain-core
poetry add langchain-core
```

## 📊 File Comparison

| **File** | **Description** | **Recommended for** |
|:---------|:---------------|:--------------------|
| `main.py` | **Langchain Agent** - Professional, automatic tool calling | ✅ **Production** |
| `main_simple.py` | **Functional** - Simple, easy to understand | ✅ **Learning** |  
| `main_with_tools.py` | **Custom Agent Class** - OOP, detailed debugging | ✅ **Development** |

## 🎯 Demo & Examples

### **Automated Testing:**

```bash
poetry run python src/gym_agent_test/demo_langchain_prompts.py
```

### **Use Cases:**

- 💪 **Personal Trainer AI** - Personalized workout consultation
- 📊 **Health Calculator** - BMI calculation and health monitoring  
- 🥗 **Nutrition Advisor** - Sports nutrition consultation
- 📱 **Fitness Chatbot** - 24/7 fitness support

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`  
4. Push branch: `git push origin feature/amazing-feature`
5. Create Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📞 Contact

**Author:** Truong Nhat Thach  
**Email:** <nhatthach2703@gmail.com>  
**Project Link:** <https://github.com/Nhatthach2703/Gym-agent-test.git>

## 🎉 Acknowledgments

- [Langchain](https://langchain.com/) - AI Agent Framework
- [Google Gemini](https://ai.google.dev/) - Generative AI Model  
- [Poetry](https://python-poetry.org/) - Dependency Management
- [Python](https://python.org/) - Programming Language

---

**⭐ If this project is helpful, please give it a star! ⭐**

```bash
# Quick start command:
git clone https://github.com/Nhatthach2703/Gym-agent-test.git
cd Gym-agent-test
poetry install
# Tạo file .env với GOOGLE_API_KEY
poetry run python src/gym_agent_test/main.py
```
