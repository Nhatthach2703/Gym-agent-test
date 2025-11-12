import os
from dotenv import load_dotenv # pyright: ignore[reportMissingImports]
from langchain.tools import tool # pyright: ignore[reportMissingImports]
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings  # pyright: ignore[reportMissingImports]
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # pyright: ignore[reportMissingImports]
from langchain.agents import create_tool_calling_agent, AgentExecutor # pyright: ignore[reportMissingImports]
from langchain_community.vectorstores import Chroma # pyright: ignore[reportMissingImports]
from langchain.text_splitter import RecursiveCharacterTextSplitter # pyright: ignore[reportMissingImports]
from langchain_core.documents import Document # pyright: ignore[reportMissingImports]
from rich.console import Console # pyright: ignore[reportMissingImports]
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import print as rich_print
import time

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Initialize Rich console
console = Console()

# Rich UI Constants
PROGRESS_TEXT_COLUMN = "[progress.description]{task.description}"
STYLE_SUCCESS = "bold green"
STYLE_WARNING = "bold yellow"
STYLE_ERROR = "bold red"
STYLE_INFO = "bold cyan"

# Conversation history để duy trì ngữ cảnh
conversation_history = []

# User profile để lưu thông tin cá nhân
user_profile = {
    "height": None,
    "weight": None,
    "bmi": None,
    "goals": [],
    "preferences": [],
    "name": None,
    "injuries": [],  # Thêm thông tin về chấn thương
}

# Khởi tạo RAG components
embeddings = None
nutrition_vectorstore = None
exercise_vectorstore = None


def initialize_rag():
    """Khởi tạo RAG với dữ liệu dinh dưỡng và bài tập"""
    global embeddings, nutrition_vectorstore, exercise_vectorstore

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn(PROGRESS_TEXT_COLUMN),
            console=console,
        ) as progress:
            progress.add_task("[cyan]Đang khởi tạo RAG system...", total=None)

            # Khởi tạo embeddings
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004", google_api_key=api_key
            )

            # Dữ liệu dinh dưỡng món ăn Việt Nam
            nutrition_data = [
                "Phở bò: 350 calories, 15g protein, 45g carbs, 12g fat. Giàu collagen từ xương, tốt cho khớp và da. Phù hợp cho người tập gym cần năng lượng. Thành phần chính: bánh phở, thịt bò, nước dùng xương bò, hành lá, rau thơm (ngò gai, húng quế), chanh, ớt.",
                "Bún bò Huế: 400 calories, 18g protein, 50g carbs, 14g fat. Chứa nhiều vitamin B từ tóp mỡ, tốt cho hệ thần kinh. Thành phần chính: bún, thịt bò, tóp mỡ, nước dùng cay (sả, ớt), rau sống (bắp chuối, giá đỗ), chanh.",
                "Cơm tấm: 450 calories, 20g protein, 60g carbs, 15g fat. Thịt nướng cung cấp protein cao, phù hợp sau tập luyện. Thành phần chính: cơm tấm, thịt nướng (sườn hoặc sườn non), bì, chả trứng, dưa chua, nước mắm pha.",
                "Bánh mì thịt: 380 calories, 16g protein, 42g carbs, 18g fat. Tiện lợi cho bữa sáng trước tập, cung cấp năng lượng nhanh. Thành phần chính: bánh mì baguette, thịt nguội (jambon, chả lụa), pate, rau củ (dưa chuột, cà rốt, ngò), sốt mayonnaise.",
                "Gỏi cuốn tôm thịt: 180 calories, 12g protein, 20g carbs, 6g fat. Nhẹ nhàng, giàu rau xanh và protein, tốt cho giảm cân. Thành phần chính: bánh tráng, tôm luộc, thịt heo luộc, bún tươi, rau sống (xà lách, húng, tía tô), nước mắm chua ngọt.",
                "Chả cá Lã Vọng: 280 calories, 25g protein, 8g carbs, 16g fat. Giàu omega-3, tốt cho tim mạch và não bộ. Thành phần chính: cá lăng (hoặc cá quả), nghệ, thì là, hành lá, dầu ăn, bún tươi, rau sống, mắm tôm.",
                "Bún chả Hà Nội: 420 calories, 22g protein, 35g carbs, 20g fat. Thịt nướng + rau thơm, cân bằng dinh dưỡng. Thành phần chính: bún tươi, thịt nướng (thịt ba chỉ, thịt vai), chả viên nướng, nước mắm pha, rau sống (xà lách, húng, tía tô), tỏi ớt.",
                "Canh chua cá: 150 calories, 18g protein, 12g carbs, 4g fat. Ít calories, nhiều vitamin C, tốt cho hệ miễn dịch. Thành phần chính: cá (cá lóc, cá tra), cà chua, dứa, đậu bắp, giá đỗ, rau thơm (ngò gai, hành lá), me, nước mắm.",
                "Thịt kho tàu: 380 calories, 28g protein, 15g carbs, 24g fat. Giàu protein, phù hợp cho tăng cơ nhưng cao fat. Thành phần chính: thịt ba chỉ, trứng vịt, nước dừa, nước mắm, đường, hành tím, tỏi, tiêu.",
                "Gà luộc: 200 calories, 30g protein, 0g carbs, 8g fat. Protein lean tốt nhất cho tập gym, ít calories. Thành phần chính: thịt gà (đùi, ức), muối, gừng, hành lá, nước mắm pha chấm.",
                "Cháo gà: 180 calories, 12g protein, 28g carbs, 3g fat. Dễ tiêu hóa, phù hợp khi ốm hoặc sau tập nặng. Thành phần chính: gạo tẻ, thịt gà, hành lá, rau mùi, gừng, nước mắm, tiêu.",
                "Nem nướng Nha Trang: 320 calories, 18g protein, 25g carbs, 16g fat. Thịt nướng + rau sống, cân bằng macro. Thành phần chính: thịt heo xay, bì heo, bánh tráng nướng, rau sống (xà lách, húng, tía tô), bún tươi, nước mắm chua ngọt.",
                "Bánh cuốn: 240 calories, 8g protein, 35g carbs, 8g fat. Nhẹ nhàng, phù hợp bữa sáng không tập nặng. Thành phần chính: bột gạo, thịt heo xay, mộc nhĩ, hành khô, nước mắm pha, chả lụa, dưa chuột.",
                "Bò lúc lắc: 350 calories, 26g protein, 12g carbs, 22g fat. Protein cao từ thịt bò, tốt cho tăng cơ. Thành phần chính: thịt bò, hành tây, ớt chuông, tỏi, tiêu, dầu ăn, nước tương.",
                "Chè đậu xanh: 220 calories, 6g protein, 42g carbs, 4g fat. Carbs tự nhiên, phù hợp sau tập cardio. Thành phần chính: đậu xanh, đường, nước cốt dừa, lá dứa, muối.",
                "Nước mía: 180 calories, 0g protein, 45g carbs, 0g fat. Đường tự nhiên, cung cấp năng lượng nhanh. Thành phần chính: mía tươi ép, đá, chanh (tùy chọn).",
                "Trà đá chanh: 30 calories, 0g protein, 8g carbs, 0g fat. Hydration tốt, vitamin C, phù hợp mọi lúc. Thành phần chính: trà xanh, chanh tươi, đường, đá.",
                "Cà phê sữa đá: 150 calories, 4g protein, 18g carbs, 6g fat. Caffeine tăng tập trung, phù hợp pre-workout. Thành phần chính: cà phê phin, sữa đặc, đá, đường (tùy chọn).",
                "Bánh tét: 280 calories, 6g protein, 58g carbs, 4g fat. Carbs phức tạp từ gạo nếp, năng lượng lâu dài. Thành phần chính: gạo nếp, đậu xanh, thịt ba chỉ, lá chuối, muối.",
                "Tôm rang me: 260 calories, 22g protein, 18g carbs, 12g fat. Protein từ tôm + vitamin A từ me. Thành phần chính: tôm tươi, me chua, đường, nước mắm, tỏi, ớt, hành lá.",
            ]

            # Dữ liệu bài tập theo nhóm cơ và chấn thương
            exercise_data = [
                "NGỰC - Bench Press: Tập ngực, vai trước, tay sau. 3 sets x 8-12 reps. Chú ý: giữ vai ổn định, hạ bar chạm ngực.",
                "NGỰC - Push-ups: Bodyweight, tập ngực toàn diện. 3 sets x 15-20 reps. Biến thể: incline, decline, diamond.",
                "NGỰC - Dumbbell Flyes: Tập ngực giữa. 3 sets x 10-15 reps. Chú ý: không hạ quá thấp, tránh chấn thương vai.",
                "LƯNG - Pull-ups: Tập lưng xô, tay trước. 3 sets x 5-12 reps. Biến thể: wide grip, chin-ups.",
                "LƯNG - Bent-over Rows: Tập lưng giữa, sau vai. 3 sets x 8-12 reps. Chú ý: giữ lưng thẳng.",
                "LƯNG - Lat Pulldowns: Máy tập lưng xô. 3 sets x 10-15 reps. Tập trung vào kéo bằng lưng.",
                "VAI - Overhead Press: Tập vai toàn diện. 3 sets x 8-12 reps. Chú ý: core chặt, không lắc lưng.",
                "VAI - Lateral Raises: Tập vai giữa. 3 sets x 12-15 reps. Tạ nhẹ, động tác chậm và kiểm soát.",
                "VAI - Rear Delt Flyes: Tập vai sau. 3 sets x 15-20 reps. Quan trọng để cân bằng tư thế.",
                "TAY TRƯỚC - Bicep Curls: Tập tay trước. 3 sets x 10-15 reps. Chú ý: không swing, kiểm soát âm tính.",
                "TAY TRƯỚC - Hammer Curls: Tập tay trước + cẳng tay. 3 sets x 10-15 reps. Grip ngang, tạ dumbbell.",
                "TAY SAU - Tricep Dips: Tập tay sau. 3 sets x 10-15 reps. Có thể dùng ghế hoặc parallel bars.",
                "TAY SAU - Overhead Tricep Extension: Tập tay sau. 3 sets x 10-15 reps. Chú ý: giữ khuỷu tay ổn định.",
                "CHÂN - Squats: Tập đùi, mông toàn diện. 3 sets x 10-15 reps. Chú ý: gót chân không rời đất.",
                "CHÂN - Deadlifts: Tập đùi sau, mông, lưng dưới. 3 sets x 5-8 reps. Kỹ thuật quan trọng nhất.",
                "CHÂN - Lunges: Tập đùi, mông đơn bên. 3 sets x 12/leg. Tốt cho cân bằng và stability.",
                "BỤ NG - Plank: Tập core tĩnh. 3 sets x 30-60s. Foundation cho mọi bài tập khác.",
                "BỤNG - Crunches: Tập bụng trên. 3 sets x 15-25 reps. Chú ý: không kéo cổ.",
                "BỤNG - Russian Twists: Tập bụng chéo. 3 sets x 20 total. Có thể thêm tạ để tăng khó.",
                "CHẤN THƯƠNG TAY - Wrist Curls: Phục hồi cổ tay. 2 sets x 15-20 reps với tạ nhẹ.",
                "CHẤN THƯƠNG TAY - Resistance Band Exercises: An toàn cho vai, khuỷu tay. Elastic band với các hướng khác nhau.",
                "CHẤN THƯƠNG LƯNG - Cat-Cow Stretch: Mobility lưng. 2 sets x 10 reps. Làm ấm trước tập.",
                "CHẤN THƯƠNG LƯNG - Bird Dog: Stability core và lưng. 3 sets x 10/side. Tăng dần độ khó.",
                "CHẤN THƯƠNG GỐI - Wall Sits: Tập đùi không impact. 3 sets x 20-45s. An toàn cho gối.",
                "CHẤN THƯƠNG GỐI - Glute Bridges: Tập mông, ít stress gối. 3 sets x 15-20 reps.",
                "CARDIO NHẸ - Walking: 30-45 phút, an toàn cho mọi chấn thương. Tốt cho recovery.",
                "CARDIO NHẸ - Swimming: Toàn thân, ít impact. 20-30 phút, tốt cho chấn thương khớp.",
                "STRETCHING - Hip Flexor Stretch: Giãn cơ hông. 30s/side. Quan trọng cho người ngồi nhiều.",
                "STRETCHING - Chest Stretch: Giãn ngực. 30s. Cân bằng với bài tập push.",
                "RECOVERY - Foam Rolling: Self-massage. 5-10 phút/nhóm cơ. Tốt cho recovery.",
            ]

            # Tạo documents cho RAG
            nutrition_docs = [Document(page_content=text) for text in nutrition_data]
            exercise_docs = [Document(page_content=text) for text in exercise_data]

            # Tạo vector stores
            nutrition_vectorstore = Chroma.from_documents(
                documents=nutrition_docs,
                embedding=embeddings,
                collection_name="nutrition",
            )

            exercise_vectorstore = Chroma.from_documents(
                documents=exercise_docs,
                embedding=embeddings,
                collection_name="exercises",
            )

            progress.stop()

        console.print("✅ RAG system đã được khởi tạo!", style=STYLE_SUCCESS)
        return True

    except Exception as e:
        console.print(f"❌ Lỗi khởi tạo RAG: {e}", style=STYLE_ERROR)
        return False


@tool
def calc_bmi(height_weight: str) -> str:
    """Tính chỉ số BMI cơ thể.
    Input: 'chiều_cao,cân_nặng' -> chiều cao (m), cân nặng (kg).
    Ví dụ: '1.70,65'"""
    try:
        h, w = map(float, height_weight.split(","))
        bmi = w / (h * h)
        if bmi < 18.5:
            status = "Thiếu cân"
        elif bmi < 24.9:
            status = "Bình thường"
        elif bmi < 29.9:
            status = "Thừa cân"
        else:
            status = "Béo phì"
        return f"BMI = {bmi:.2f} → {status}"
    except Exception:
        return "Sai input! Hãy nhập theo định dạng: chiều_cao,cân_nặng (VD: 1.70,65)"


@tool
def nutrition_advisor_rag(query: str) -> str:
    """Tư vấn dinh dưỡng món ăn Việt Nam sử dụng RAG.
    Input: Câu hỏi về món ăn, dinh dưỡng, calories, etc."""
    try:
        if not nutrition_vectorstore:
            return "❌ RAG chưa được khởi tạo. Hãy khởi động lại ứng dụng."

        # Tìm kiếm relevant documents
        relevant_docs = nutrition_vectorstore.similarity_search(query, k=3)

        # Tạo context từ documents
        context = "\n".join([doc.page_content for doc in relevant_docs])

        # Format response
        response = f"🍜 **TƯ VẤN DINH DƯỠNG MÓN ĂN VIỆT NAM:**\n\n"
        response += f"📋 **Thông tin liên quan:**\n{context}\n\n"
        response += f"💡 **Gợi ý:** Dựa trên thông tin trên, bạn có thể chọn món phù hợp với mục tiêu của mình."

        return response

    except Exception as e:
        return f"❌ Lỗi RAG nutrition: {e}"


@tool
def exercise_advisor_rag(query: str) -> str:
    """Tư vấn bài tập dựa theo nhóm cơ hoặc chấn thương sử dụng RAG.
    Input: Câu hỏi về bài tập, nhóm cơ, chấn thương, etc."""
    try:
        if not exercise_vectorstore:
            return "❌ RAG chưa được khởi tạo. Hãy khởi động lại ứng dụng."

        # Tìm kiếm relevant documents
        relevant_docs = exercise_vectorstore.similarity_search(query, k=4)

        # Tạo context từ documents
        context = "\n".join([doc.page_content for doc in relevant_docs])

        # Format response
        response = f"💪 **TƯ VẤN BÀI TẬP THEO NHÓM CƠ/CHẤN THƯƠNG:**\n\n"
        response += f"🏋️ **Bài tập phù hợp:**\n{context}\n\n"
        response += f"⚠️ **Lưu ý:** Nếu có chấn thương, hãy tập nhẹ nhàng và tham khảo bác sĩ nếu đau."

        return response

    except Exception as e:
        return f"❌ Lỗi RAG exercise: {e}"


@tool
def gym_advice_tool(question: str) -> str:
    """Đưa ra lời khuyên gym tổng quát (backup cho RAG)."""
    advice_db = {
        "tăng cơ": "💪 Gợi ý tăng cơ:\n- Tập trọng lượng nặng, ít rep (6-8 reps)\n- Nghỉ đủ giấc (7-9h/đêm)\n- Ăn nhiều protein (1.6-2.2g/kg cân nặng)",
        "giảm cân": "🔥 Gợi ý giảm cân:\n- Cardio 30-45p/ngày\n- Deficit calories 300-500 cal\n- Tập circuit training\n- Uống đủ nước (2-3L/ngày)",
        "tăng sức mạnh": "💪 Gợi ý tăng sức mạnh:\n- Compound exercises: squat, deadlift, bench press\n- Progressive overload\n- Nghỉ ngơi đủ giữa các set",
    }

    question_lower = question.lower()
    for key, advice in advice_db.items():
        if key in question_lower:
            return advice

    return "🤔 Hỏi cụ thể hơn về: tăng cơ, giảm cân, tăng sức mạnh. Hoặc dùng RAG tools cho tư vấn chi tiết!"


def create_agent(llm):
    """Tạo agent với RAG tools và conversation history"""
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn(PROGRESS_TEXT_COLUMN),
            console=console,
        ) as progress:
            progress.add_task("[cyan]Đang khởi tạo Agent với RAG tools...", total=None)

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        f"""Bạn là Sgms AI - một huấn luyện viên gym chuyên nghiệp với hệ thống RAG tiên tiến.
                
                QUAN TRỌNG: Bạn có thể nhớ toàn bộ cuộc trò chuyện và thông tin cá nhân để tạo cuộc hội thoại tự nhiên.
                
                THÔNG TIN USER HIỆN TẠI:
                - Chiều cao: {user_profile.get('height', 'chưa có')} m
                - Cân nặng: {user_profile.get('weight', 'chưa có')} kg  
                - BMI: {user_profile.get('bmi', 'chưa tính')}
                - Mục tiêu: {', '.join(user_profile.get('goals', [])) or 'chưa rõ'}
                - Chấn thương: {', '.join(user_profile.get('injuries', [])) or 'không có'}
                
                TOOLS CÓ SẴN:
                1. calc_bmi(height_weight) - Tính BMI với format "chiều_cao,cân_nặng" 
                2. nutrition_advisor_rag(query) - RAG tư vấn dinh dưỡng món ăn Việt Nam
                3. exercise_advisor_rag(query) - RAG tư vấn bài tập theo nhóm cơ/chấn thương
                4. gym_advice_tool(question) - Tư vấn gym tổng quát (backup)
                
                HƯỚNG DẪN SỬ DỤNG RAG:
                - Với câu hỏi về món ăn, calories, dinh dưỡng → dùng nutrition_advisor_rag
                - Với câu hỏi về bài tập, nhóm cơ, chấn thương → dùng exercise_advisor_rag
                - Ưu tiên RAG tools trước, fallback sang gym_advice_tool nếu cần
                
                Hướng dẫn trả lời:
                - SỬ DỤNG thông tin cá nhân đã có để đưa ra lời khuyên cụ thể
                - Tham khảo lịch sử cuộc trò chuyện để duy trì ngữ cảnh
                - Đặt câu hỏi tiếp theo để thu thập thêm thông tin cần thiết
                - Tạo kế hoạch dài hạn và cá nhân hóa dựa trên profile user
                - Gợi ý bước tiếp theo phù hợp với mục tiêu
                
                Luôn trả lời bằng tiếng Việt, thân thiện và chuyên nghiệp!""",
                    ),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]
            )

            # Tạo tools và agent với RAG
            tools = [
                calc_bmi,
                nutrition_advisor_rag,
                exercise_advisor_rag,
                gym_advice_tool,
            ]
            agent = create_tool_calling_agent(llm, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

            progress.stop()

        console.print("✅ Agent với RAG tools đã sẵn sàng!", style=STYLE_SUCCESS)
        return agent_executor

    except Exception as e:
        console.print(f"⚠️ Không thể tạo agent với RAG: {e}", style=STYLE_WARNING)
        console.print("🔄 Sẽ sử dụng chế độ chat đơn giản...", style="yellow")
        return None


def simple_chat(user_input: str, llm) -> str:
    """Fallback chat đơn giản với conversation history"""
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn(PROGRESS_TEXT_COLUMN),
            console=console,
        ) as progress:
            progress.add_task("[cyan]Đang suy nghĩ...", total=None)
            time.sleep(0.5)

            # Tạo context từ lịch sử cuộc trò chuyện
            context = ""
            if conversation_history:
                context = "\n\nLịch sử cuộc trò chuyện:\n"
                for i, (user_msg, ai_msg) in enumerate(conversation_history[-3:]):
                    context += f"User: {user_msg}\nSgms AI: {ai_msg}\n\n"

            profile_info = f"""
            
THÔNG TIN USER:
- Chiều cao: {user_profile.get('height', 'chưa có')} m
- Cân nặng: {user_profile.get('weight', 'chưa có')} kg
- Mục tiêu: {', '.join(user_profile.get('goals', [])) or 'chưa rõ'}
- Chấn thương: {', '.join(user_profile.get('injuries', [])) or 'không có'}"""

            system_prompt = f"""Bạn là Sgms AI - một huấn luyện viên gym chuyên nghiệp và thân thiện với RAG. 
            
            QUAN TRỌNG: Sử dụng thông tin cá nhân và lịch sử cuộc trò chuyện để duy trì ngữ cảnh.{profile_info}
            
            Trả lời chuyên nghiệp về:
            - Bài tập gym, thể hình dựa trên thông tin cá nhân và chấn thương
            - Dinh dưỡng thể thao phù hợp với mục tiêu (đặc biệt món ăn Việt Nam)
            - Kế hoạch tập luyện cá nhân hóa
            - Sức khỏe và thể chất
            
            Luôn bắt đầu với emoji phù hợp và giọng điệu thân thiện.{context}"""

            full_prompt = f"{system_prompt}\n\nCâu hỏi hiện tại: {user_input}"
            response = llm.invoke(full_prompt)
            return response.content

    except Exception as e:
        return f"🤖 Sgms AI: Xin lỗi, tôi đang gặp vấn đề kỹ thuật. Hãy thử lại sau! ({str(e)[:50]}...)"


def display_welcome():
    """Hiển thị màn hình chào mừng với Rich và RAG features"""
    # Tạo title
    title = Text("🤖 GYM AGENT RAG - Sgms AI với RAG! 🏋️‍♂️", style="bold magenta")

    # Tạo bảng hướng dẫn
    table = Table(show_header=True, header_style="bold blue")
    table.add_column("Tính năng", style="cyan", no_wrap=True)
    table.add_column("Ví dụ", style="green")

    table.add_row("📊 Tính BMI", "'Tính BMI cho tôi 1.75,70'")
    table.add_row("🍜 RAG Dinh dưỡng VN", "'Phở bò có bao nhiêu calories?'")
    table.add_row("💪 RAG Bài tập nhóm cơ", "'Bài tập ngực nào tốt nhất?'")
    table.add_row("🩹 RAG Chấn thương", "'Chấn thương tay nên tập gì?'")
    table.add_row("💬 Hội thoại liên tục", "Tôi nhớ cuộc trò chuyện trước đó!")
    table.add_row("🚪 Thoát", "'exit' hoặc 'quit'")

    # Thông tin về RAG
    rag_info = Text("\n🚀 TÍNH NĂNG RAG MỚI:\n", style="bold yellow")
    rag_info.append(
        "• 🍜 Tư vấn dinh dưỡng 20+ món ăn Việt Nam với calories chi tiết\n",
        style="green",
    )
    rag_info.append(
        "• 💪 Tư vấn bài tập theo 7 nhóm cơ chính + bài tập phục hồi chấn thương\n",
        style="green",
    )
    rag_info.append(
        "• 🧠 AI sử dụng RAG để tìm thông tin chính xác từ database\n", style="green"
    )
    rag_info.append(
        "• 📚 Ví dụ: 'Tôi bị đau vai, nên tập bài tập nào?' hoặc 'Phở có phù hợp giảm cân?'",
        style="cyan",
    )

    # Hiển thị trong panel
    console.print(Panel(title, expand=False, border_style="bright_blue"))
    console.print(rag_info)
    console.print(table)
    console.print("\n" + "=" * 60 + "\n", style="dim")


def extract_user_info(user_input: str):
    """Trích xuất thông tin cá nhân từ tin nhắn của user bao gồm chấn thương"""
    global user_profile

    # Extract height (1.70, 1m70, 170cm, etc.)
    import re

    height_patterns = [
        r"(\d+\.?\d*)\s*m(?:\s|$)",
        r"(\d+)\s*cm",
        r"cao\s+(\d+\.?\d*)",
    ]

    for pattern in height_patterns:
        match = re.search(pattern, user_input.lower())
        if match:
            height = float(match.group(1))
            if height > 10:
                height = height / 100
            user_profile["height"] = height
            break

    # Extract weight
    weight_patterns = [
        r"(\d+\.?\d*)\s*kg",
        r"nặng\s+(\d+\.?\d*)",
        r"cân\s+(\d+\.?\d*)",
    ]

    for pattern in weight_patterns:
        match = re.search(pattern, user_input.lower())
        if match:
            user_profile["weight"] = float(match.group(1))
            break

    # Extract goals
    goal_keywords = {
        "tăng cơ": ["tăng cơ", "build muscle", "muscle", "cơ bắp"],
        "giảm cân": ["giảm cân", "lose weight", "weight loss", "gầy"],
        "tăng sức mạnh": ["mạnh", "strength", "sức mạnh"],
        "giữ dáng": ["giữ dáng", "maintain", "duy trì"],
    }

    for goal, keywords in goal_keywords.items():
        if any(keyword in user_input.lower() for keyword in keywords):
            if goal not in user_profile["goals"]:
                user_profile["goals"].append(goal)

    # Extract injuries - NEW FEATURE
    injury_keywords = {
        "vai": ["đau vai", "chấn thương vai", "vai bị", "shoulder pain"],
        "tay": ["đau tay", "chấn thương tay", "tay bị", "arm pain"],
        "lưng": ["đau lưng", "chấn thương lưng", "lưng bị", "back pain"],
        "gối": ["đau gối", "chấn thương gối", "gối bị", "knee pain"],
        "cổ tay": ["đau cổ tay", "chấn thương cổ tay", "wrist pain"],
        "chân": ["đau chân", "chấn thương chân", "chân bị", "leg pain"],
    }

    for injury, keywords in injury_keywords.items():
        if any(keyword in user_input.lower() for keyword in keywords):
            if injury not in user_profile["injuries"]:
                user_profile["injuries"].append(injury)


def format_chat_history_for_agent():
    """Format conversation history for agent input"""
    chat_messages = []
    for user_msg, ai_msg in conversation_history[-5:]:
        chat_messages.append(("human", user_msg))
        chat_messages.append(("assistant", ai_msg))
    return chat_messages


def get_contextual_suggestions():
    """Đưa ra gợi ý câu hỏi dựa trên user profile và RAG capabilities"""
    suggestions = []

    if user_profile["height"] and user_profile["weight"]:
        bmi = user_profile["weight"] / (user_profile["height"] ** 2)
        user_profile["bmi"] = bmi

        if bmi < 18.5:
            suggestions.append(
                "💡 RAG: Hỏi về 'món ăn tăng cân Việt Nam' hoặc 'bài tập tăng cơ'"
            )
        elif bmi > 25:
            suggestions.append(
                "💡 RAG: Thử 'món ăn ít calories' hoặc 'bài tập giảm cân cardio'"
            )
        else:
            suggestions.append(
                "💡 RAG: Hỏi 'bài tập duy trì' hoặc 'dinh dưỡng cân bằng'"
            )

    if user_profile["injuries"]:
        injury_list = ", ".join(user_profile["injuries"])
        suggestions.append(
            f"💡 RAG Chấn thương: 'Bài tập phù hợp khi bị {injury_list}'"
        )

    if not user_profile["height"] or not user_profile["weight"]:
        suggestions.append(
            "💡 Cần: Cho tôi biết chiều cao và cân nặng để tư vấn tốt hơn!"
        )

    # RAG specific suggestions
    suggestions.append("💡 RAG Dinh dưỡng: 'Phở/Bún bò/Cơm tấm có phù hợp tập gym?'")
    suggestions.append("💡 RAG Bài tập: 'Bài tập ngực/lưng/chân tốt nhất là gì?'")

    return suggestions


def chat_loop(agent_executor, llm):
    """Main chat loop với Rich UI, conversation history và RAG"""
    global conversation_history

    while True:
        try:
            user_input = Prompt.ask("\n[bold cyan]👤 Bạn[/bold cyan]").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "thoát"]:
                console.print(
                    "\n🙋‍♂️ Cảm ơn bạn đã sử dụng RAG Gym Agent! Hẹn gặp lại!",
                    style=STYLE_SUCCESS,
                )
                break

            # Sử dụng agent với RAG nếu có
            if agent_executor:
                try:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[cyan]🤖 Sgms AI đang tra cứu RAG..."),
                        console=console,
                    ) as progress:
                        progress.add_task("", total=None)

                        # Tạo input với chat history
                        agent_input = {
                            "input": user_input,
                            "chat_history": format_chat_history_for_agent(),
                        }
                        result = agent_executor.invoke(agent_input)

                    response = result["output"]
                except Exception as e:
                    console.print(f"⚠️ Agent RAG lỗi: {e}", style="yellow")
                    response = simple_chat(user_input, llm)
            else:
                response = simple_chat(user_input, llm)

            # Extract thông tin từ user input (bao gồm chấn thương)
            extract_user_info(user_input)

            # Lưu vào conversation history
            conversation_history.append((user_input, response))

            if len(conversation_history) > 10:
                conversation_history = conversation_history[-10:]

            # Hiển thị response trong panel đẹp
            response_panel = Panel(
                response,
                title="💬 Sgms AI (RAG)",
                title_align="left",
                border_style="green",
                padding=(1, 2),
            )
            console.print(response_panel)

            # Hiển thị gợi ý RAG dựa trên context
            suggestions = get_contextual_suggestions()
            if suggestions:
                for suggestion in suggestions[-2:]:
                    console.print(f"[dim blue]{suggestion}[/dim blue]")

            # Hiển thị profile info với chấn thương
            if len(conversation_history) > 1:
                injuries_text = (
                    f" | Chấn thương: {','.join(user_profile.get('injuries', []))}"
                    if user_profile.get("injuries")
                    else ""
                )
                console.print(
                    f"[dim]💭 RAG Memory: {len(conversation_history)} cuộc hội thoại | H={user_profile.get('height', '?')}m, W={user_profile.get('weight', '?')}kg{injuries_text}[/dim]"
                )

        except KeyboardInterrupt:
            console.print(
                "\n\n🙋‍♂️ Cảm ơn bạn đã sử dụng RAG Gym Agent!", style=STYLE_SUCCESS
            )
            break
        except Exception as e:
            console.print(f"\n❌ Có lỗi xảy ra: {e}", style=STYLE_ERROR)
            console.print("🔄 Hãy thử lại...", style="yellow")


def main():
    """Main function to run the RAG gym agent"""

    # Initialize LLM
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn(PROGRESS_TEXT_COLUMN),
            console=console,
        ) as progress:
            progress.add_task("[cyan]Đang kết nối Gemini API...", total=None)
            time.sleep(1)

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash", api_key=api_key, temperature=0.7
            )

        console.print("✅ Kết nối Gemini API thành công!", style=STYLE_SUCCESS)
    except Exception as e:
        console.print(f"❌ Lỗi kết nối API: {e}", style=STYLE_ERROR)
        console.print("💡 Hãy kiểm tra GOOGLE_API_KEY trong file .env", style="yellow")
        return

    # Khởi tạo RAG system
    if not initialize_rag():
        console.print("⚠️ Tiếp tục mà không có RAG", style=STYLE_WARNING)

    # Tạo agent với RAG
    agent_executor = create_agent(llm)

    if not agent_executor:
        console.print("⚠️ Sẽ sử dụng chế độ chat đơn giản", style=STYLE_WARNING)

    # Hiển thị màn hình chào mừng
    console.clear()
    display_welcome()

    # Bắt đầu chat loop
    chat_loop(agent_executor, llm)


if __name__ == "__main__":
    main()
