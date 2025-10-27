import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
# Sẽ sử dụng conversation history đơn giản với tuple (user_input, ai_response)
from rich.console import Console
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
    "name": None
}


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
def gym_advice_tool(question: str) -> str:
    """Đưa ra lời khuyên về tập gym, bài tập, dinh dưỡng."""
    advice_db = {
        "tăng cơ": "💪 Gợi ý tăng cơ:\n- Tập trọng lượng nặng, ít rep (6-8 reps)\n- Nghỉ đủ giấc (7-9h/đêm)\n- Ăn nhiều protein (1.6-2.2g/kg cân nặng)",
        "giảm cân": "🔥 Gợi ý giảm cân:\n- Cardio 30-45p/ngày\n- Deficit calories 300-500 cal\n- Tập circuit training\n- Uống đủ nước (2-3L/ngày)",
        "tăng sức mạnh": "💪 Gợi ý tăng sức mạnh:\n- Compound exercises: squat, deadlift, bench press\n- Progressive overload\n- Nghỉ ngơi đủ giữa các set",
        "bụng": "🏋️ Bài tập bụng:\n- Plank (30s-2p)\n- Crunches, bicycle crunches\n- Mountain climbers, russian twists",
        "chân": "🦵 Bài tập chân:\n- Squats, lunges\n- Leg press, calf raises\n- Bulgarian split squats",
        "tay": "💪 Bài tập tay:\n- Push-ups, pull-ups, dips\n- Bicep curls, tricep extensions\n- Overhead press",
        "ngực": "🏋️ Bài tập ngực:\n- Bench press, push-ups\n- Dumbbell flyes, incline press",
        "lưng": "🏋️ Bài tập lưng:\n- Pull-ups, rows\n- Lat pulldowns, deadlifts",
        "dinh dưỡng": "🍗 Dinh dưỡng:\n- Protein: thịt, cá, trứng, sữa\n- Carbs: yến mạch, gạo lứt\n- Fats: nuts, olive oil",
    }

    question_lower = question.lower()
    for key, advice in advice_db.items():
        if key in question_lower:
            return advice

    return "🤔 Hỏi cụ thể hơn về: tăng cơ, giảm cân, tăng sức mạnh, bài tập bụng/chân/tay/ngực/lưng, dinh dưỡng"


def create_agent(llm):
    """Tạo agent với tools và conversation history"""
    # Tạo prompt template
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn(PROGRESS_TEXT_COLUMN),
            console=console,
        ) as progress:
            progress.add_task("[cyan]Đang khởi tạo Agent với tools...", total=None)
            
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        f"""Bạn là Sgms AI - một huấn luyện viên gym chuyên nghiệp và thân thiện. 
                
                QUAN TRỌNG: Bạn có thể nhớ toàn bộ cuộc trò chuyện và thông tin cá nhân để tạo cuộc hội thoại tự nhiên.
                
                THÔNG TIN USER HIỆN TẠI:
                - Chiều cao: {user_profile.get('height', 'chưa có')} m
                - Cân nặng: {user_profile.get('weight', 'chưa có')} kg  
                - BMI: {user_profile.get('bmi', 'chưa tính')}
                - Mục tiêu: {', '.join(user_profile.get('goals', [])) or 'chưa rõ'}
                
                Bạn có các tools để:
                - Tính BMI: calc_bmi(height_weight) với format "chiều_cao,cân_nặng" VD: "1.70,65"
                - Tư vấn gym: gym_advice_tool(question) cho câu hỏi về bài tập, dinh dưỡng
                
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

            # Tạo tools và agent
            tools = [calc_bmi, gym_advice_tool]
            agent = create_tool_calling_agent(llm, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
            
            progress.stop()

        console.print("✅ Agent với tools đã sẵn sàng!", style=STYLE_SUCCESS)
        return agent_executor

    except Exception as e:
        console.print(f"⚠️ Không thể tạo agent với tools: {e}", style=STYLE_WARNING)
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
            time.sleep(0.5)  # Hiệu ứng loading
            
            # Tạo context từ lịch sử cuộc trò chuyện
            context = ""
            if conversation_history:
                context = "\n\nLịch sử cuộc trò chuyện:\n"
                for i, (user_msg, ai_msg) in enumerate(conversation_history[-3:]):  # Chỉ lấy 3 tin nhắn gần nhất
                    context += f"User: {user_msg}\nSgms AI: {ai_msg}\n\n"
            
            profile_info = f"""
            
THÔNG TIN USER:
- Chiều cao: {user_profile.get('height', 'chưa có')} m
- Cân nặng: {user_profile.get('weight', 'chưa có')} kg
- Mục tiêu: {', '.join(user_profile.get('goals', [])) or 'chưa rõ'}"""

            system_prompt = f"""Bạn là Sgms AI - một huấn luyện viên gym chuyên nghiệp và thân thiện. 
            
            QUAN TRỌNG: Sử dụng thông tin cá nhân và lịch sử cuộc trò chuyện để duy trì ngữ cảnh.{profile_info}
            
            Trả lời chuyên nghiệp về:
            - Bài tập gym, thể hình dựa trên thông tin cá nhân
            - Dinh dưỡng thể thao phù hợp với mục tiêu
            - Kế hoạch tập luyện cá nhân hóa
            - Sức khỏe và thể chất
            
            Luôn bắt đầu với emoji phù hợp và giọng điệu thân thiện.{context}"""

            full_prompt = f"{system_prompt}\n\nCâu hỏi hiện tại: {user_input}"
            response = llm.invoke(full_prompt)
            return response.content

    except Exception as e:
        return f"🤖 Sgms AI: Xin lỗi, tôi đang gặp vấn đề kỹ thuật. Hãy thử lại sau! ({str(e)[:50]}...)"


def display_welcome():
    """Hiển thị màn hình chào mừng với Rich"""
    # Tạo title
    title = Text("🤖 GYM AGENT - Sgms AI READY! 🏋️‍♂️", style="bold magenta")
    
    # Tạo bảng hướng dẫn
    table = Table(show_header=True, header_style="bold blue")
    table.add_column("Tính năng", style="cyan", no_wrap=True)
    table.add_column("Ví dụ", style="green")
    
    table.add_row("📊 Tính BMI", "'Tính BMI cho tôi 1.75,70'")
    table.add_row("💪 Tư vấn tập luyện", "'Gợi ý bài tập tăng cơ tay'")
    table.add_row("🍗 Hỏi về dinh dưỡng", "'Ăn gì để tăng cơ?'")
    table.add_row("� Hội thoại liên tục", "Tôi nhớ cuộc trò chuyện trước đó!")
    table.add_row("�🚪 Thoát", "'exit' hoặc 'quit'")
    
    # Thông tin về tính năng mới
    feature_info = Text("\n🚀 TÍNH NĂNG MỚI: Conversation Memory\n", style="bold yellow")
    feature_info.append("• Tôi có thể nhớ cuộc trò chuyện trước đó\n", style="green")
    feature_info.append("• Bạn có thể hỏi tiếp dựa trên thông tin đã cung cấp\n", style="green") 
    feature_info.append("• Ví dụ: 'Tôi cao 1.75m, nặng 70kg' → sau đó 'Gợi ý bài tập cho tôi'", style="cyan")
    
    # Hiển thị trong panel
    console.print(Panel(title, expand=False, border_style="bright_blue"))
    console.print(feature_info)
    console.print(table)
    console.print("\n" + "="*60 + "\n", style="dim")

def main():
    """Main function to run the gym agent"""
    
    # Initialize LLM
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn(PROGRESS_TEXT_COLUMN),
            console=console,
        ) as progress:
            progress.add_task("[cyan]Đang kết nối Gemini API...", total=None)
            time.sleep(1)  # Hiệu ứng loading
            
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash", api_key=api_key, temperature=0.7
            )
            
        console.print("✅ Kết nối Gemini API thành công!", style=STYLE_SUCCESS)
    except Exception as e:
        console.print(f"❌ Lỗi kết nối API: {e}", style=STYLE_ERROR)
        console.print("💡 Hãy kiểm tra GOOGLE_API_KEY trong file .env", style="yellow")
        return

    # Tạo agent
    agent_executor = create_agent(llm)

    if not agent_executor:
        console.print("⚠️ Sẽ sử dụng chế độ chat đơn giản", style=STYLE_WARNING)

    # Hiển thị màn hình chào mừng
    console.clear()
    display_welcome()

    # Bắt đầu chat loop
    chat_loop(agent_executor, llm)

def extract_user_info(user_input: str):
    """Trích xuất thông tin cá nhân từ tin nhắn của user"""
    global user_profile
    
    # Extract height (1.70, 1m70, 170cm, etc.)
    import re
    height_patterns = [
        r'(\d+\.?\d*)\s*m(?:\s|$)',  # 1.70m, 1.7m  
        r'(\d+)\s*cm',  # 170cm
        r'cao\s+(\d+\.?\d*)',  # cao 1.70
    ]
    
    for pattern in height_patterns:
        match = re.search(pattern, user_input.lower())
        if match:
            height = float(match.group(1))
            if height > 10:  # Likely in cm
                height = height / 100
            user_profile["height"] = height
            break
    
    # Extract weight (70kg, 70 kg, nặng 70, etc.)
    weight_patterns = [
        r'(\d+\.?\d*)\s*kg',
        r'nặng\s+(\d+\.?\d*)',
        r'cân\s+(\d+\.?\d*)',
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
        "giữ dáng": ["giữ dáng", "maintain", "duy trì"]
    }
    
    for goal, keywords in goal_keywords.items():
        if any(keyword in user_input.lower() for keyword in keywords):
            if goal not in user_profile["goals"]:
                user_profile["goals"].append(goal)

def format_chat_history_for_agent():
    """Format conversation history for agent input"""
    chat_messages = []
    for user_msg, ai_msg in conversation_history[-5:]:  # Lấy 5 tin nhắn gần nhất
        chat_messages.append(("human", user_msg))
        chat_messages.append(("assistant", ai_msg))
    return chat_messages

def get_contextual_suggestions():
    """Đưa ra gợi ý câu hỏi dựa trên user profile"""
    suggestions = []
    
    if user_profile["height"] and user_profile["weight"]:
        bmi = user_profile["weight"] / (user_profile["height"] ** 2)
        user_profile["bmi"] = bmi
        
        if bmi < 18.5:
            suggestions.append("💡 Gợi ý: Bạn hơi gầy, muốn tăng cơ không?")
        elif bmi > 25:
            suggestions.append("💡 Gợi ý: Có muốn tôi tư vấn kế hoạch giảm cân không?")
        else:
            suggestions.append("💡 Gợi ý: Cân nặng ổn! Muốn tập thể hình hay cardio?")
    
    if not user_profile["height"] or not user_profile["weight"]:
        suggestions.append("💡 Gợi ý: Cho tôi biết chiều cao và cân nặng để tư vấn tốt hơn nhé!")
    
    if user_profile["goals"]:
        suggestions.append(f"💡 Mục tiêu: {', '.join(user_profile['goals'])}")
    
    return suggestions

def chat_loop(agent_executor, llm):
    """Main chat loop với Rich UI và conversation history"""
    global conversation_history
    
    while True:
        try:
            # Sử dụng Rich Prompt thay vì input()
            user_input = Prompt.ask("\n[bold cyan]👤 Bạn[/bold cyan]").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "thoát"]:
                console.print("\n🙋‍♂️ Cảm ơn bạn đã sử dụng! Hẹn gặp lại!", style=STYLE_SUCCESS)
                break

            # Sử dụng agent nếu có, không thì fallback
            if agent_executor:
                try:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[cyan]🤖 Sgms AI đang suy nghĩ..."),
                        console=console,
                    ) as progress:
                        progress.add_task("", total=None)
                        
                        # Tạo input với chat history
                        agent_input = {
                            "input": user_input,
                            "chat_history": format_chat_history_for_agent()
                        }
                        result = agent_executor.invoke(agent_input)
                    
                    response = result["output"]
                except Exception as e:
                    console.print(f"⚠️ Agent lỗi: {e}", style="yellow")
                    response = simple_chat(user_input, llm)
            else:
                response = simple_chat(user_input, llm)

            # Extract thông tin từ user input
            extract_user_info(user_input)
            
            # Lưu vào conversation history
            conversation_history.append((user_input, response))
            
            # Giới hạn history tối đa 10 cuộc hội thoại
            if len(conversation_history) > 10:
                conversation_history = conversation_history[-10:]

            # Hiển thị response trong panel đẹp
            response_panel = Panel(
                response,
                title="💬 Sgms AI",
                title_align="left",
                border_style="green",
                padding=(1, 2)
            )
            console.print(response_panel)
            
            # Hiển thị gợi ý dựa trên context
            suggestions = get_contextual_suggestions()
            if suggestions:
                for suggestion in suggestions[-2:]:  # Chỉ hiển thị 2 gợi ý gần nhất
                    console.print(f"[dim blue]{suggestion}[/dim blue]")
            
            # Hiển thị số tin nhắn trong history
            if len(conversation_history) > 1:
                console.print(f"[dim]💭 Đang nhớ {len(conversation_history)} cuộc hội thoại | Profile: H={user_profile.get('height', '?')}m, W={user_profile.get('weight', '?')}kg[/dim]")

        except KeyboardInterrupt:
            console.print("\n\n🙋‍♂️ Cảm ơn bạn đã sử dụng! Hẹn gặp lại!", style=STYLE_SUCCESS)
            break
        except Exception as e:
            console.print(f"\n❌ Có lỗi xảy ra: {e}", style=STYLE_ERROR)
            console.print("🔄 Hãy thử lại...", style="yellow")


if __name__ == "__main__":
    main()
