import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

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


def get_gym_advice(question: str) -> str:
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


def chat_with_gemini(user_input: str, llm) -> str:
    """Chat trực tiếp với Gemini để trả lời câu hỏi về gym"""

    # Kiểm tra xem có phải là yêu cầu tính BMI không
    if (
        any(keyword in user_input.lower() for keyword in ["bmi", "chỉ số", "cân nặng"])
        or "tính" in user_input.lower()
    ):
        # Tìm pattern số trong input
        import re

        numbers = re.findall(r"\d+\.?\d*", user_input)
        if len(numbers) >= 2:
            try:
                height_weight = f"{numbers[0]},{numbers[1]}"
                return calc_bmi(height_weight)
            except Exception:
                return "Để tính BMI, hãy nhập: 'Tính BMI 1.75,70' (chiều cao m, cân nặng kg)"
        else:
            return (
                "Để tính BMI, hãy nhập: 'Tính BMI 1.75,70' (chiều cao m, cân nặng kg)"
            )

    # Kiểm tra advice cơ bản trước
    advice = get_gym_advice(user_input)
    if "🤔" not in advice:  # Nếu tìm thấy advice cụ thể
        return advice

    # Sử dụng Gemini để trả lời
    try:
        system_prompt = """Bạn là PT Nam - một huấn luyện viên gym chuyên nghiệp và thân thiện. 
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
        return f"🤖 PT Nam: Xin lỗi, tôi đang gặp vấn đề kỹ thuật. Hãy thử lại sau! ({str(e)[:50]}...)"


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

    # Start the conversation
    print("\n" + "=" * 50)
    print("🤖 GYM AGENT - PT NAM READY! 🏋️‍♂️")
    print("=" * 50)
    print("📋 Tôi có thể giúp bạn:")
    print("• Tính BMI: 'Tính BMI 1.75,70'")
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

            print("\n🤖 PT Nam đang suy nghĩ...")
            response = chat_with_gemini(user_input, llm)
            print(f"\n💬 PT Nam: {response}")

        except KeyboardInterrupt:
            print("\n\n🙋‍♂️ Cảm ơn bạn đã sử dụng! Hẹn gặp lại!")
            break
        except Exception as e:
            print(f"\n❌ Có lỗi xảy ra: {e}")
            print("🔄 Hãy thử lại...")


if __name__ == "__main__":
    main()
