import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")


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
    """Tạo agent với tools"""
    # Tạo prompt template
    try:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Bạn là Sgms AI - một huấn luyện viên gym chuyên nghiệp và thân thiện. 
            Bạn có các tools để:
            - Tính BMI: calc_bmi(height_weight) với format "chiều_cao,cân_nặng" VD: "1.70,65"
            - Tư vấn gym: gym_advice_tool(question) cho câu hỏi về bài tập, dinh dưỡng
            
            Khi user hỏi về BMI, hãy sử dụng tool calc_bmi.
            Khi user hỏi về tập gym, hãy sử dụng tool gym_advice_tool.
            Nếu không có tool phù hợp, trả lời trực tiếp bằng kiến thức của bạn.
            
            Luôn trả lời bằng tiếng Việt, thân thiện và chuyên nghiệp!""",
                ),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        # Tạo tools và agent
        tools = [calc_bmi, gym_advice_tool]
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        return agent_executor

    except Exception as e:
        print(f"⚠️ Không thể tạo agent với tools: {e}")
        print("🔄 Sẽ sử dụng chế độ chat đơn giản...")
        return None


def simple_chat(user_input: str, llm) -> str:
    """Fallback chat đơn giản nếu agent không hoạt động"""
    try:
        system_prompt = """Bạn là Sgms AI - một huấn luyện viên gym chuyên nghiệp và thân thiện. 
        Trả lời ngắn gọn (2-4 câu), chuyên nghiệp về:
        - Bài tập gym, thể hình
        - Dinh dưỡng thể thao  
        - Kế hoạch tập luyện
        - Sức khỏe và thể chất
        
        Luôn bắt đầu với emoji phù hợp và giọng điệu thân thiện."""

        full_prompt = f"{system_prompt}\n\nCâu hỏi: {user_input}"
        response = llm.invoke(full_prompt)
        return response.content

    except Exception as e:
        return f"🤖 Sgms AI: Xin lỗi, tôi đang gặp vấn đề kỹ thuật. Hãy thử lại sau! ({str(e)[:50]}...)"


def main():
    """Main function to run the gym agent"""

    # Initialize LLM
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", api_key=api_key, temperature=0.7
        )
        print("✅ Kết nối Gemini API thành công!")
    except Exception as e:
        print(f"❌ Lỗi kết nối API: {e}")
        print("💡 Hãy kiểm tra GOOGLE_API_KEY trong file .env")
        return

    # Tạo agent
    print("🔧 Đang khởi tạo agent với tools...")
    agent_executor = create_agent(llm)

    if agent_executor:
        print("✅ Agent với tools đã sẵn sàng!")
    else:
        print("⚠️ Sẽ sử dụng chế độ chat đơn giản")

    # Start the conversation
    print("\n" + "=" * 50)
    print("🤖 GYM AGENT - Sgms AI READY! 🏋️‍♂️")
    print("=" * 50)
    print("📋 Tôi có thể giúp bạn:")
    print("• Tính BMI: 'Tính BMI cho tôi 1.75,70'")
    print("• Tư vấn tập luyện: 'Gợi ý bài tập tăng cơ tay'")
    print("• Hỏi về dinh dưỡng: 'Ăn gì để tăng cơ?'")
    print("• Gõ 'exit' hoặc 'quit' để thoát")
    print("-" * 50)

    while True:
        try:
            user_input = input("\n👤 Bạn: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "thoát"]:
                print("🙋‍♂️ Cảm ơn bạn đã sử dụng! Hẹn gặp lại!")
                break

            print("\n🤖 Sgms AI đang suy nghĩ...")

            # Sử dụng agent nếu có, không thì fallback
            if agent_executor:
                try:
                    result = agent_executor.invoke({"input": user_input})
                    response = result["output"]
                except Exception as e:
                    print(f"⚠️ Agent lỗi: {e}")
                    response = simple_chat(user_input, llm)
            else:
                response = simple_chat(user_input, llm)

            print(f"\n💬 Sgms AI: {response}")

        except KeyboardInterrupt:
            print("\n\n🙋‍♂️ Cảm ơn bạn đã sử dụng! Hẹn gặp lại!")
            break
        except Exception as e:
            print(f"\n❌ Có lỗi xảy ra: {e}")
            print("🔄 Hãy thử lại...")


if __name__ == "__main__":
    main()
