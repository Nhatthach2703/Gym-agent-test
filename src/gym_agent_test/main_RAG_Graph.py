import os
import re
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import print as rich_print
import time
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
neo4j_uri = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
neo4j_user = os.getenv("NEO4J_USER", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD", "admin1234")
neo4j_database = os.getenv("NEO4J_DATABASE", "test")

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

# Khởi tạo GraphRAG components
graph_driver = None
ingredient_names_cache = []


def connect_neo4j():
    """Kết nối với Neo4j database"""
    global graph_driver
    try:
        console.print("\n🔌 Đang kết nối với Neo4j...", style=STYLE_INFO)

        graph_driver = GraphDatabase.driver(
            neo4j_uri, auth=(neo4j_user, neo4j_password)
        )

        # Test connection
        with graph_driver.session(database=neo4j_database) as session:
            result = session.run("RETURN 1 as test")
            result.single()

        # Lấy thống kê graph database
        stats, dish_count = get_graph_statistics()

        # Tạo thông tin kết nối
        connection_info = f"""📍 URI: {neo4j_uri}
👤 User: {neo4j_user}
🗄️  Database: {neo4j_database}
✅ Status: [bold green]Connected[/bold green]"""

        # Thêm thống kê nếu có
        if stats:
            connection_info += f"\n\n[bold cyan]📊 Thống kê Graph Database:[/bold cyan]"
            if dish_count > 0:
                connection_info += f"\n   🍜 Món ăn (Dish): {dish_count}"
            if stats.get("Cuisine", 0) > 0:
                connection_info += f"\n   🗺️  Cuisine: {stats.get('Cuisine', 0)}"
            if stats.get("Ingredient", 0) > 0:
                connection_info += f"\n   🥘 Ingredients: {stats.get('Ingredient', 0)}"
            if stats.get("Tag", 0) > 0:
                connection_info += f"\n   🏷️  Tags: {stats.get('Tag', 0)}"
            if stats.get("Benefit", 0) > 0:
                connection_info += f"\n   ✨ Benefits: {stats.get('Benefit', 0)}"
            if stats.get("Total Nodes", 0) > 0:
                connection_info += f"\n   📦 Tổng Nodes: {stats.get('Total Nodes', 0)}"
            if stats.get("Total Relationships", 0) > 0:
                connection_info += (
                    f"\n   🔗 Tổng Relationships: {stats.get('Total Relationships', 0)}"
                )
        else:
            connection_info += (
                "\n\n[bold yellow]⚠️ Database trống hoặc chưa có dữ liệu[/bold yellow]"
            )

        success_panel = Panel(
            connection_info,
            title="[bold green]✅ KẾT NỐI THÀNH CÔNG VỚI NEO4J[/bold green]",
            title_align="center",
            border_style="green",
            padding=(1, 2),
            expand=False,
        )
        console.print(success_panel)
        console.print()  # Empty line for spacing

        return True
    except Exception as e:
        # Hiển thị lỗi với Rich Panel
        error_info = f"""[bold red]❌ Lỗi:[/bold red] {str(e)}

[bold yellow]💡 Kiểm tra:[/bold yellow]
   • Neo4j server đang chạy?
   • URI, user, password đúng chưa?
   • Database '{neo4j_database}' đã được tạo chưa?"""

        error_panel = Panel(
            error_info,
            title="[bold red]❌ LỖI KẾT NỐI NEO4J[/bold red]",
            title_align="center",
            border_style="red",
            padding=(1, 2),
            expand=False,
        )
        console.print(error_panel)
        console.print()  # Empty line for spacing
        return False


def get_graph_statistics():
    """Lấy thống kê về graph database để kiểm tra dữ liệu"""
    try:
        with graph_driver.session(database=neo4j_database) as session:
            # Đếm số lượng nodes theo label
            stats = {}

            # Đếm các loại nodes
            node_labels = ["Dish", "Cuisine", "Tag", "Ingredient", "Macro", "Benefit"]
            for label in node_labels:
                result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                count = result.single()["count"]
                if count > 0:
                    stats[label] = count

            # Đếm tổng số nodes
            total_nodes_result = session.run("MATCH (n) RETURN count(n) as total")
            stats["Total Nodes"] = total_nodes_result.single()["total"]

            # Đếm tổng số relationships
            total_rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as total")
            stats["Total Relationships"] = total_rel_result.single()["total"]

            # Đếm số món ăn (Dish)
            dish_count = stats.get("Dish", 0)

            return stats, dish_count
    except Exception as e:
        console.print(f"⚠️ Lỗi lấy thống kê: {e}", style=STYLE_WARNING)
        return None, 0


def get_all_ingredient_names(force_refresh: bool = False):
    """Lấy danh sách tên nguyên liệu (cache để giảm số lần query)"""
    global ingredient_names_cache

    if ingredient_names_cache and not force_refresh:
        return ingredient_names_cache

    if not graph_driver:
        return []

    try:
        with graph_driver.session(database=neo4j_database) as session:
            result = session.run("MATCH (i:Ingredient) RETURN toLower(i.name) as name")
            ingredient_names_cache = [
                record["name"] for record in result if record.get("name")
            ]
    except Exception as e:
        console.print(f"⚠️ Lỗi lấy danh sách nguyên liệu: {e}", style=STYLE_WARNING)

    return ingredient_names_cache


def clear_neo4j():
    """Xóa toàn bộ dữ liệu trong Neo4j (optional, để reset)"""
    try:
        with graph_driver.session(database=neo4j_database) as session:
            session.run("MATCH (n) DETACH DELETE n")
        return True
    except Exception as e:
        console.print(f"⚠️ Lỗi xóa dữ liệu: {e}", style=STYLE_WARNING)
        return False


def initialize_rag():
    """Khởi tạo GraphRAG với Neo4j - chỉ kết nối, không populate data"""
    global graph_driver

    try:
        # Kết nối Neo4j (thông báo sẽ hiển thị trong connect_neo4j)
        if not connect_neo4j():
            return False

        # Lấy thống kê để hiển thị
        stats, dish_count = get_graph_statistics()

        console.print(
            "✅ GraphRAG system với Neo4j đã được khởi tạo!", style=STYLE_SUCCESS
        )
        if stats and dish_count > 0:
            console.print(
                f"📊 Database '{neo4j_database}' có sẵn {dish_count} món ăn và {stats.get('Total Nodes', 0)} nodes",
                style=STYLE_INFO,
            )
        else:
            console.print(
                f"📊 Database '{neo4j_database}' đã sẵn sàng (chưa có dữ liệu hoặc đang kiểm tra...)",
                style=STYLE_INFO,
            )
        return True

    except Exception as e:
        console.print(f"❌ Lỗi khởi tạo GraphRAG: {e}", style=STYLE_ERROR)
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
    """Tư vấn dinh dưỡng món ăn Việt Nam sử dụng GraphRAG với Neo4j.
    Input: Câu hỏi về món ăn, dinh dưỡng, calories, etc."""
    try:
        if not graph_driver:
            return "❌ GraphRAG chưa được khởi tạo. Hãy khởi động lại ứng dụng."

        query_lower = query.lower()
        results = []
        ingredient_matches = extract_ingredients_from_query(query_lower)

        # Log để biết AI đang query graph database
        console.print(
            f"[dim]🔍 GraphRAG: Đang query Neo4j database '{neo4j_database}'...[/dim]"
        )

        # Extract tên món ăn từ query (tìm các từ khóa món ăn)
        dish_keywords = [
            "phở bò",
            "phở gà",
            "phở",
            "bún bò",
            "bún chả",
            "bún",
            "cơm tấm",
            "cơm",
            "bánh mì",
            "bánh cuốn",
            "bánh tét",
            "bánh",
            "gỏi cuốn",
            "gỏi",
            "chả cá",
            "chả",
            "canh chua",
            "canh",
            "thịt kho",
            "thịt",
            "gà luộc",
            "gà",
            "cháo gà",
            "cháo",
            "nem nướng",
            "nem",
            "bò lúc lắc",
            "bò",
            "chè đậu",
            "chè",
            "nước mía",
            "nước",
            "trà đá",
            "trà",
            "cà phê",
            "tôm rang",
            "tôm",
        ]

        # Tìm từ khóa món ăn trong query (ưu tiên cụm từ dài hơn)
        dish_keyword = None
        for keyword in sorted(dish_keywords, key=len, reverse=True):
            if keyword in query_lower:
                dish_keyword = keyword
                break

        with graph_driver.session(database=neo4j_database) as session:
            # Query 1: Tìm món ăn theo tên (sử dụng từ khóa đã extract)
            if dish_keyword:
                try:
                    console.print(
                        f"[dim]🔍 GraphRAG: Tìm kiếm với keyword: '{dish_keyword}'[/dim]"
                    )
                    # Query với các relationships theo schema - Match Dish trước, sau đó optional relationships
                    dish_query = """
                        MATCH (d:Dish)
                        WHERE toLower(d.name) CONTAINS toLower($dish_keyword)
                        OPTIONAL MATCH (d)-[:BELONGS_TO]->(c:Cuisine)
                        OPTIONAL MATCH (d)-[rel:CONTAINS]->(i:Ingredient)
                        OPTIONAL MATCH (i)-[:PROVIDES_BENEFIT]->(ib:Benefit)
                        WITH d, c, rel, i, collect(DISTINCT ib.name) AS ingredient_benefits
                        WITH d,
                             c,
                             collect(
                                 DISTINCT {
                                     name: i.name,
                                     quantity: rel.quantity_g,
                                     benefits: ingredient_benefits
                                 }
                             ) AS ingredients
                        OPTIONAL MATCH (d)-[:HAS_BENEFIT]->(b:Benefit)
                        RETURN d.name as dish_name, 
                               d.calories as calories,
                               d.protein_g as protein,
                               d.carbs_g as carbs,
                               d.fat_g as fat,
                               c.name as cuisine,
                               ingredients,
                               collect(DISTINCT b.name) as benefits
                        LIMIT 5
                    """
                    dish_results = session.run(dish_query, dish_keyword=dish_keyword)
                    record_count = 0
                    for record in dish_results:
                        record_count += 1
                        # Xử lý None values
                        ingredients_raw = record.get("ingredients") or []
                        ingredients = []
                        for ing in ingredients_raw:
                            if not ing:
                                continue
                            if isinstance(ing, dict):
                                name = ing.get("name")
                                quantity = ing.get("quantity")
                                ing_benefits = [
                                    b for b in (ing.get("benefits") or []) if b
                                ]
                            else:
                                name = ing
                                quantity = None
                                ing_benefits = []
                            if name:
                                ingredients.append(
                                    {
                                        "name": name,
                                        "quantity": quantity,
                                        "benefits": ing_benefits,
                                    }
                                )

                        benefits = record.get("benefits") or []

                        # Filter None values từ collections
                        benefits = [b for b in benefits if b is not None]

                        results.append(
                            {
                                "type": "dish",
                                "name": record.get("dish_name"),
                                "calories": record.get("calories"),
                                "protein": record.get("protein_g"),
                                "carbs": record.get("carbs_g"),
                                "fat": record.get("fat_g"),
                                "cuisine": record.get("cuisine"),
                                "benefits": benefits,
                                "ingredients": ingredients,
                            }
                        )

                    if record_count == 0:
                        console.print(
                            f"[dim]⚠️ GraphRAG: Không tìm thấy với keyword '{dish_keyword}', thử fallback...[/dim]"
                        )
                    else:
                        console.print(
                            f"[dim]✅ GraphRAG: Tìm thấy {record_count} record với keyword '{dish_keyword}'[/dim]"
                        )
                except Exception as e:
                    console.print(f"[dim]❌ GraphRAG: Lỗi query dish: {str(e)}[/dim]")

            # Query fallback: Nếu không tìm thấy với tên chính xác, thử tìm với từ khóa
            if not results and any(
                keyword in query_lower
                for keyword in [
                    "phở",
                    "bún",
                    "cơm",
                    "bánh",
                    "gỏi",
                    "chả",
                    "canh",
                    "thịt",
                    "gà",
                    "cháo",
                    "nem",
                    "bò",
                    "chè",
                    "nước",
                    "trà",
                    "cà phê",
                    "tôm",
                ]
            ):
                try:
                    # Extract từ khóa đơn giản hơn
                    simple_keywords = [
                        "phở",
                        "bún bò",
                        "bún chả",
                        "bún",
                        "cơm tấm",
                        "cơm",
                        "bánh mì",
                        "bánh cuốn",
                        "bánh tét",
                        "bánh",
                        "gỏi cuốn",
                        "gỏi",
                        "chả cá",
                        "chả",
                        "canh chua",
                        "canh",
                        "thịt kho",
                        "thịt",
                        "gà luộc",
                        "gà",
                        "cháo gà",
                        "cháo",
                        "nem nướng",
                        "nem",
                        "bò lúc lắc",
                        "bò",
                        "chè đậu",
                        "chè",
                        "nước mía",
                        "nước",
                        "trà đá",
                        "trà",
                        "cà phê",
                        "tôm rang",
                        "tôm",
                    ]

                    search_keyword = None
                    for keyword in sorted(simple_keywords, key=len, reverse=True):
                        if keyword in query_lower:
                            search_keyword = keyword
                            break

                    if search_keyword:
                        console.print(
                            f"[dim]🔄 GraphRAG: Fallback query với keyword: '{search_keyword}'[/dim]"
                        )
                        fallback_query = """
                            MATCH (d:Dish)
                            WHERE toLower(d.name) CONTAINS toLower($search_keyword)
                            OPTIONAL MATCH (d)-[:BELONGS_TO]->(c:Cuisine)
                            OPTIONAL MATCH (d)-[rel:CONTAINS]->(i:Ingredient)
                            OPTIONAL MATCH (i)-[:PROVIDES_BENEFIT]->(ib:Benefit)
                            WITH d, c, rel, i, collect(DISTINCT ib.name) AS ingredient_benefits
                            WITH d,
                                 c,
                                 collect(
                                     DISTINCT {
                                         name: i.name,
                                         quantity: rel.quantity_g,
                                         benefits: ingredient_benefits
                                     }
                                 ) AS ingredients
                            OPTIONAL MATCH (d)-[:HAS_BENEFIT]->(b:Benefit)
                            RETURN d.name as dish_name, 
                                   d.calories as calories,
                                   d.protein_g as protein,
                                   d.carbs_g as carbs,
                                   d.fat_g as fat,
                                   c.name as cuisine,
                                   ingredients,
                                   collect(DISTINCT b.name) as benefits
                            LIMIT 5
                        """
                        fallback_results = session.run(
                            fallback_query, search_keyword=search_keyword
                        )
                        fallback_count = 0
                        for record in fallback_results:
                            fallback_count += 1
                            ingredients_raw = record.get("ingredients") or []
                            ingredients = []
                            for ing in ingredients_raw:
                                if not ing:
                                    continue
                                if isinstance(ing, dict):
                                    name = ing.get("name")
                                    quantity = ing.get("quantity")
                                    ing_benefits = [
                                        b for b in (ing.get("benefits") or []) if b
                                    ]
                                else:
                                    name = ing
                                    quantity = None
                                    ing_benefits = []
                                if name:
                                    ingredients.append(
                                        {
                                            "name": name,
                                            "quantity": quantity,
                                            "benefits": ing_benefits,
                                        }
                                    )
                            benefits = [
                                b
                                for b in (record.get("benefits") or [])
                                if b is not None
                            ]

                            results.append(
                                {
                                    "type": "dish",
                                    "name": record.get("dish_name"),
                                    "calories": record.get("calories"),
                                    "protein": record.get("protein_g"),
                                    "carbs": record.get("carbs_g"),
                                    "fat": record.get("fat_g"),
                                    "cuisine": record.get("cuisine"),
                                    "benefits": benefits,
                                    "ingredients": ingredients,
                                }
                            )

                        if fallback_count > 0:
                            console.print(
                                f"[dim]✅ GraphRAG: Fallback tìm thấy {fallback_count} record[/dim]"
                            )
                        else:
                            console.print(
                                f"[dim]⚠️ GraphRAG: Fallback cũng không tìm thấy kết quả[/dim]"
                            )

                            # Debug: Thử query tất cả dishes để xem có data không
                            try:
                                debug_query = (
                                    "MATCH (d:Dish) RETURN d.name as name LIMIT 5"
                                )
                                debug_results = session.run(debug_query)
                                debug_dishes = [r["name"] for r in debug_results]
                                if debug_dishes:
                                    console.print(
                                        f"[dim]🔍 GraphRAG Debug: Tìm thấy {len(debug_dishes)} dishes trong DB: {', '.join(debug_dishes)}[/dim]"
                                    )
                                else:
                                    console.print(
                                        f"[dim]⚠️ GraphRAG Debug: Database trống hoặc không có Dish nodes[/dim]"
                                    )
                            except Exception as debug_e:
                                console.print(
                                    f"[dim]⚠️ GraphRAG Debug: Lỗi debug query: {str(debug_e)}[/dim]"
                                )
                except Exception as e:
                    console.print(
                        f"[dim]❌ GraphRAG: Lỗi query fallback: {str(e)}[/dim]"
                    )

            # Query ingredient: Gợi ý món ăn dựa trên nguyên liệu user có
            if ingredient_matches:
                ingredient_trigger = (
                    len(ingredient_matches) >= 2
                    or "nguyên liệu" in query_lower
                    or "ingredient" in query_lower
                    or (
                        "món" in query_lower
                        and any(
                            keyword in query_lower
                            for keyword in ["có", "làm", "nấu", "từ", "với"]
                        )
                    )
                )

                if ingredient_trigger:
                    try:
                        console.print(
                            f"[dim]🧾 GraphRAG: Gợi ý món từ nguyên liệu {', '.join(ingredient_matches)}[/dim]"
                        )
                        ingredient_query = """
                            MATCH (d:Dish)-[rel:CONTAINS]->(i:Ingredient)
                            WHERE toLower(i.name) IN $ingredient_names
                            OPTIONAL MATCH (i)-[:PROVIDES_BENEFIT]->(ib:Benefit)
                            WITH d,
                                 rel,
                                 i,
                                 collect(DISTINCT ib.name) AS ingredient_benefits
                            WITH d,
                                 collect(
                                     DISTINCT {
                                         name: i.name,
                                         quantity: rel.quantity_g,
                                         benefits: ingredient_benefits
                                     }
                                 ) AS matched_ingredients,
                                 count(DISTINCT i) AS match_count
                            OPTIONAL MATCH (d)-[rel_all:CONTAINS]->(all_i:Ingredient)
                            OPTIONAL MATCH (all_i)-[:PROVIDES_BENEFIT]->(all_ib:Benefit)
                            WITH d,
                                 matched_ingredients,
                                 match_count,
                                 rel_all,
                                 all_i,
                                 collect(DISTINCT all_ib.name) AS all_ingredient_benefits
                            WITH d,
                                 matched_ingredients,
                                 match_count,
                                 collect(
                                     DISTINCT {
                                         name: all_i.name,
                                         quantity: rel_all.quantity_g,
                                         benefits: all_ingredient_benefits
                                     }
                                 ) AS ingredients
                            OPTIONAL MATCH (d)-[:HAS_BENEFIT]->(b:Benefit)
                            RETURN d.name AS dish_name,
                                   d.calories AS calories,
                                   d.protein_g AS protein,
                                   d.carbs_g AS carbs,
                                   d.fat_g AS fat,
                                   matched_ingredients,
                                   ingredients,
                                   collect(DISTINCT b.name) AS benefits,
                                   match_count
                            ORDER BY match_count DESC, d.calories ASC
                            LIMIT 5
                        """
                        ingredient_results = session.run(
                            ingredient_query, ingredient_names=ingredient_matches
                        )

                        for record in ingredient_results:
                            matched_ing_raw = record.get("matched_ingredients") or []
                            matched_ing = []
                            for ing in matched_ing_raw:
                                if not ing:
                                    continue
                                if isinstance(ing, dict):
                                    name = ing.get("name")
                                    quantity = ing.get("quantity")
                                    ing_benefits = [
                                        b for b in (ing.get("benefits") or []) if b
                                    ]
                                else:
                                    name = ing
                                    quantity = None
                                    ing_benefits = []
                                if name:
                                    matched_ing.append(
                                        {
                                            "name": name,
                                            "quantity": quantity,
                                            "benefits": ing_benefits,
                                        }
                                    )

                            all_ingredients_raw = record.get("ingredients") or []
                            all_ingredients = []
                            for ing in all_ingredients_raw:
                                if not ing:
                                    continue
                                if isinstance(ing, dict):
                                    name = ing.get("name")
                                    quantity = ing.get("quantity")
                                    ing_benefits = [
                                        b for b in (ing.get("benefits") or []) if b
                                    ]
                                else:
                                    name = ing
                                    quantity = None
                                    ing_benefits = []
                                if name:
                                    all_ingredients.append(
                                        {
                                            "name": name,
                                            "quantity": quantity,
                                            "benefits": ing_benefits,
                                        }
                                    )
                            benefits = [
                                b
                                for b in (record.get("benefits") or [])
                                if b is not None
                            ]
                            results.append(
                                {
                                    "type": "dish_by_ingredient",
                                    "name": record.get("dish_name"),
                                    "calories": record.get("calories"),
                                    "protein": record.get("protein"),
                                    "carbs": record.get("carbs"),
                                    "fat": record.get("fat"),
                                    "ingredients": all_ingredients,
                                    "matched_ingredients": matched_ing,
                                    "match_count": record.get(
                                        "match_count", len(matched_ing)
                                    ),
                                    "benefits": benefits,
                                }
                            )
                    except Exception as e:
                        console.print(
                            f"[dim]❌ GraphRAG: Lỗi query món theo nguyên liệu: {str(e)}[/dim]"
                        )

            # Query 2: Tìm món ăn theo calories
            if "calories" in query_lower or "cal" in query_lower:
                cal_match = re.search(r"(\d+)\s*cal", query_lower)
                if cal_match:
                    target_cal = int(cal_match.group(1))
                    cal_query = """
                        MATCH (d:Dish)
                        WHERE d.calories <= $target_cal + 50 AND d.calories >= $target_cal - 50
                        OPTIONAL MATCH (d)-[:HAS_BENEFIT]->(b:Benefit)
                        RETURN d.name as dish_name, d.calories as calories, d.protein_g as protein,
                               d.carbs_g as carbs, d.fat_g as fat,
                               collect(DISTINCT b.name) as benefits
                        ORDER BY abs(d.calories - $target_cal)
                        LIMIT 5
                    """
                    cal_results = session.run(cal_query, target_cal=target_cal)
                    for record in cal_results:
                        results.append(
                            {
                                "type": "dish_by_cal",
                                "name": record["dish_name"],
                                "calories": record["calories"],
                                "protein": record["protein_g"],
                                "carbs": record["carbs_g"],
                                "fat": record["fat_g"],
                                "benefits": record["benefits"],
                            }
                        )

            # Query 3: Tìm món ăn theo benefit (match với benefit names trong database)
            benefit_keywords = {
                "tăng cơ": ["tăng cơ", "protein cao", "protein"],
                "giảm cân": ["giảm cân", "ít calories", "rau xanh"],
                "khớp": ["khớp", "collagen", "da"],
                "tim mạch": ["tim mạch", "omega-3", "não bộ"],
                "miễn dịch": ["miễn dịch", "vitamin c"],
                "năng lượng": ["năng lượng", "pre-workout", "caffeine"],
                "tiêu hóa": ["tiêu hóa", "dễ tiêu"],
            }

            for keyword, search_terms in benefit_keywords.items():
                if any(term in query_lower for term in search_terms):
                    # Query với LIKE pattern để match benefit names
                    benefit_query = """
                        MATCH (d:Dish)-[:HAS_BENEFIT]->(b:Benefit)
                        WHERE any(term IN $search_terms WHERE toLower(b.name) CONTAINS toLower(term))
                        RETURN d.name as dish_name, d.calories as calories, 
                               d.protein_g as protein, d.carbs_g as carbs, d.fat_g as fat
                        LIMIT 5
                    """
                    benefit_results = session.run(
                        benefit_query, search_terms=search_terms
                    )
                    for record in benefit_results:
                        results.append(
                            {
                                "type": "dish_by_benefit",
                                "name": record["dish_name"],
                                "calories": record["calories"],
                                "protein": record["protein_g"],
                                "carbs": record["carbs_g"],
                                "fat": record["fat_g"],
                            }
                        )

            # Query 4: Tìm món ăn theo tag (nếu HAS_TAG relationship tồn tại)
            # Lưu ý: HAS_TAG có thể không tồn tại trong database, nên bỏ qua query này
            # Hoặc có thể query qua Dish properties nếu có tag field
            # if "breakfast" in query_lower or "bữa sáng" in query_lower:
            #     # Skip HAS_TAG query vì relationship có thể không tồn tại
            #     pass

            # Query 5: Tìm món ăn theo protein cao
            if "protein" in query_lower and (
                "cao" in query_lower or "nhiều" in query_lower
            ):
                protein_query = """
                    MATCH (d:Dish)
                    WHERE d.protein_g >= 20
                    RETURN d.name as dish_name, d.calories as calories,
                           d.protein_g as protein, d.carbs_g as carbs, d.fat_g as fat
                    ORDER BY d.protein_g DESC
                    LIMIT 5
                """
                protein_results = session.run(protein_query)
                for record in protein_results:
                    results.append(
                        {
                            "type": "dish_by_protein",
                            "name": record["dish_name"],
                            "calories": record["calories"],
                            "protein": record["protein_g"],
                            "carbs": record["carbs_g"],
                            "fat": record["fat_g"],
                        }
                    )

            # Query 6: Tìm món ăn ít calories
            if (
                "ít calories" in query_lower
                or "low calorie" in query_lower
                or "giảm cân" in query_lower
            ):
                lowcal_query = """
                    MATCH (d:Dish)
                    WHERE d.calories <= 250
                    RETURN d.name as dish_name, d.calories as calories,
                           d.protein_g as protein, d.carbs_g as carbs, d.fat_g as fat
                    ORDER BY d.calories ASC
                    LIMIT 5
                """
                lowcal_results = session.run(lowcal_query)
                for record in lowcal_results:
                    results.append(
                        {
                            "type": "dish_low_cal",
                            "name": record["dish_name"],
                            "calories": record["calories"],
                            "protein": record["protein_g"],
                            "carbs": record["carbs_g"],
                            "fat": record["fat_g"],
                        }
                    )

            # Query 7: Tìm ingredient và macro
            if (
                "ingredient" in query_lower
                or "nguyên liệu" in query_lower
                or "thành phần" in query_lower
            ):
                # Sử dụng search_term để tránh conflict với parameter name 'query'
                ing_query = """
                    MATCH (i:Ingredient)-[:HAS_MACRO]->(m:Macro)
                    WHERE toLower(i.name) CONTAINS toLower($search_term)
                    RETURN i.name as ingredient, m.calories_per_100g as calories,
                           m.protein_g_per_100g as protein, m.carbs_g_per_100g as carbs,
                           m.fat_g_per_100g as fat
                    LIMIT 5
                """
                ing_results = session.run(ing_query, search_term=query)
                for record in ing_results:
                    results.append(
                        {
                            "type": "ingredient",
                            "name": record["ingredient"],
                            "calories": record["calories"],
                            "protein": record["protein"],
                            "carbs": record["carbs"],
                            "fat": record["fat"],
                        }
                    )

        # Format response
        if not results:
            console.print(
                f"[dim]⚠️ GraphRAG: Không tìm thấy kết quả trong database[/dim]"
            )
            return "❌ Không tìm thấy thông tin phù hợp. Hãy thử hỏi về tên món ăn, calories, protein, hoặc benefits."

        # Log số lượng kết quả tìm được
        unique_count = len(set(r.get("name", "") for r in results if r.get("name")))
        console.print(
            f"[dim]✅ GraphRAG: Tìm thấy {unique_count} kết quả từ Neo4j database[/dim]"
        )

        response = "🍜 **TƯ VẤN DINH DƯỠNG MÓN ĂN VIỆT NAM (GraphRAG):**\n\n"

        # Remove duplicates based on dish name
        seen_dishes = set()
        unique_results = []
        for r in results:
            if r.get("name") and r["name"] not in seen_dishes:
                seen_dishes.add(r["name"])
                unique_results.append(r)

        for i, result in enumerate(unique_results[:5], 1):
            if result["type"] == "dish" or result["type"].startswith("dish"):
                response += f"**{i}. {result['name']}**\n"
                response += f"   📊 Calories: {result['calories']} | Protein: {result['protein']}g | Carbs: {result['carbs']}g | Fat: {result['fat']}g\n"
                if result.get("cuisine"):
                    response += f"   🗺️ Cuisine: {result['cuisine']}\n"
                if result.get("benefits"):
                    response += f"   ✨ Benefits: {', '.join(result['benefits'])}\n"
                matched_text = format_ingredient_list(
                    result.get("matched_ingredients") or []
                )
                if matched_text:
                    response += f"   ✅ Nguyên liệu khớp: {matched_text}\n"
                ingredients_text = format_ingredient_list(
                    result.get("ingredients") or []
                )
                if ingredients_text:
                    response += f"   🥘 Ingredients: {ingredients_text}\n"
                response += "\n"
            elif result["type"] == "ingredient":
                response += f"**{i}. Nguyên liệu: {result['name']}**\n"
                response += f"   📊 (per 100g) Calories: {result['calories']} | Protein: {result['protein']}g | Carbs: {result['carbs']}g | Fat: {result['fat']}g\n\n"

        response += "💡 **Gợi ý:** Dựa trên thông tin GraphRAG, bạn có thể chọn món phù hợp với mục tiêu của mình."

        return response

    except Exception as e:
        error_msg = str(e)
        console.print(f"[dim]❌ GraphRAG: Lỗi khi query database: {error_msg}[/dim]")
        # Trả về message thân thiện hơn
        return f"❌ Xin lỗi, có lỗi khi truy vấn GraphRAG database. Lỗi: {error_msg[:100]}. Hãy thử lại với câu hỏi khác hoặc kiểm tra kết nối Neo4j."


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
                        f"""Bạn là Sgms AI - một huấn luyện viên gym chuyên nghiệp với hệ thống GraphRAG tiên tiến sử dụng Neo4j.
                
                QUAN TRỌNG: Bạn có thể nhớ toàn bộ cuộc trò chuyện và thông tin cá nhân để tạo cuộc hội thoại tự nhiên.
                
                THÔNG TIN USER HIỆN TẠI:
                - Chiều cao: {user_profile.get('height', 'chưa có')} m
                - Cân nặng: {user_profile.get('weight', 'chưa có')} kg  
                - BMI: {user_profile.get('bmi', 'chưa tính')}
                - Mục tiêu: {', '.join(user_profile.get('goals', [])) or 'chưa rõ'}
                - Chấn thương: {', '.join(user_profile.get('injuries', [])) or 'không có'}
                
                TOOLS CÓ SẴN:
                1. calc_bmi(height_weight) - Tính BMI với format "chiều_cao,cân_nặng" 
                2. nutrition_advisor_rag(query) - GraphRAG tư vấn dinh dưỡng món ăn Việt Nam từ Neo4j
                3. gym_advice_tool(question) - Tư vấn gym tổng quát (backup)
                
                HƯỚNG DẪN SỬ DỤNG GraphRAG:
                - Với câu hỏi về món ăn, calories, dinh dưỡng → dùng nutrition_advisor_rag
                - GraphRAG có thể query theo: tên món, calories, protein, benefits, tags, ingredients
                - Ưu tiên GraphRAG tools trước, fallback sang gym_advice_tool nếu cần
                
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
                gym_advice_tool,
            ]
            agent = create_tool_calling_agent(llm, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

            progress.stop()

        console.print("✅ Agent với GraphRAG tools đã sẵn sàng!", style=STYLE_SUCCESS)
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

            system_prompt = f"""Bạn là Sgms AI - một huấn luyện viên gym chuyên nghiệp và thân thiện với GraphRAG. 
            
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
    title = Text("🤖 GYM AGENT GraphRAG - Sgms AI với Neo4j! 🏋️‍♂️", style="bold magenta")

    # Tạo bảng hướng dẫn
    table = Table(show_header=True, header_style="bold blue")
    table.add_column("Tính năng", style="cyan", no_wrap=True)
    table.add_column("Ví dụ", style="green")

    table.add_row("📊 Tính BMI", "'Tính BMI cho tôi 1.75,70'")
    table.add_row("🍜 GraphRAG Dinh dưỡng VN", "'Phở bò có bao nhiêu calories?'")
    table.add_row("💬 Hội thoại liên tục", "Tôi nhớ cuộc trò chuyện trước đó!")
    table.add_row("🚪 Thoát", "'exit' hoặc 'quit'")

    # Thông tin về GraphRAG
    rag_info = Text("\n🚀 TÍNH NĂNG GraphRAG MỚI:\n", style="bold yellow")
    rag_info.append(
        "• 🍜 Tư vấn dinh dưỡng 20+ món ăn Việt Nam với Neo4j Graph Database\n",
        style="green",
    )
    rag_info.append(
        "• 🧠 AI sử dụng GraphRAG để query relationships: Dish → Ingredient → Macro → Benefit\n",
        style="green",
    )
    rag_info.append(
        "• 📚 Ví dụ: 'Phở có phù hợp giảm cân?' hoặc 'Món ăn nào nhiều protein?' hoặc 'Tìm món ăn có benefit tăng cơ'",
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


def extract_ingredients_from_query(query_lower: str):
    """Tìm các nguyên liệu xuất hiện trong câu hỏi"""
    ingredient_names = get_all_ingredient_names()
    if not ingredient_names:
        return []

    matches = []
    temp_query = query_lower
    for name in sorted(ingredient_names, key=len, reverse=True):
        if name in temp_query:
            matches.append(name)
            temp_query = temp_query.replace(name, " ")

    # Loại bỏ trùng lặp nhưng giữ nguyên thứ tự
    seen = set()
    unique_matches = []
    for item in matches:
        if item not in seen:
            unique_matches.append(item)
            seen.add(item)

    return unique_matches


def format_ingredient_list(ingredients, limit=10):
    """Hiển thị danh sách nguyên liệu với định lượng"""
    if not ingredients:
        return ""

    formatted = []
    for ing in ingredients[:limit]:
        if isinstance(ing, dict):
            name = ing.get("name")
            quantity = ing.get("quantity")
            benefits = [b for b in (ing.get("benefits") or []) if b]
            if name:
                details = []
                if quantity is not None:
                    details.append(f"{quantity}g")
                if benefits:
                    details.append(f"lợi ích: {', '.join(benefits[:5])}")
                if details:
                    formatted.append(f"{name} ({'; '.join(details)})")
                else:
                    formatted.append(name)
        else:
            formatted.append(str(ing))

    if len(ingredients) > limit:
        formatted.append("...")

    return ", ".join([item for item in formatted if item])


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
            suggestions.append("💡 GraphRAG: Hỏi về 'món ăn tăng cân Việt Nam'")
        elif bmi > 25:
            suggestions.append("💡 GraphRAG: Thử 'món ăn ít calories'")
        else:
            suggestions.append("💡 GraphRAG: Hỏi 'dinh dưỡng cân bằng'")

    if not user_profile["height"] or not user_profile["weight"]:
        suggestions.append(
            "💡 Cần: Cho tôi biết chiều cao và cân nặng để tư vấn tốt hơn!"
        )

        # GraphRAG specific suggestions
        suggestions.append(
            "💡 GraphRAG Dinh dưỡng: 'Phở/Bún bò/Cơm tấm có phù hợp tập gym?'"
        )

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
                    "\n🙋‍♂️ Cảm ơn bạn đã sử dụng GraphRAG Gym Agent! Hẹn gặp lại!",
                    style=STYLE_SUCCESS,
                )
                break

            # Sử dụng agent với RAG nếu có
            if agent_executor:
                try:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[cyan]🤖 Sgms AI đang tra cứu GraphRAG..."),
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

            if len(conversation_history) > 30:
                conversation_history = conversation_history[-30:]

            # Hiển thị response trong panel đẹp
            response_panel = Panel(
                response,
                title="💬 Sgms AI (GraphRAG)",
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
                    f"[dim]💭 GraphRAG Memory: {len(conversation_history)} cuộc hội thoại | H={user_profile.get('height', '?')}m, W={user_profile.get('weight', '?')}kg{injuries_text}[/dim]"
                )

        except KeyboardInterrupt:
            console.print(
                "\n\n🙋‍♂️ Cảm ơn bạn đã sử dụng GraphRAG Gym Agent!", style=STYLE_SUCCESS
            )
            break
        except Exception as e:
            console.print(f"\n❌ Có lỗi xảy ra: {e}", style=STYLE_ERROR)
            console.print("🔄 Hãy thử lại...", style="yellow")


def main():
    """Main function to run the RAG gym agent"""

    # Clear screen trước khi bắt đầu
    console.clear()

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

    # Khởi tạo GraphRAG system
    if not initialize_rag():
        console.print("⚠️ Tiếp tục mà không có GraphRAG", style=STYLE_WARNING)

    # Tạo agent với GraphRAG
    agent_executor = create_agent(llm)

    if not agent_executor:
        console.print("⚠️ Sẽ sử dụng chế độ chat đơn giản", style=STYLE_WARNING)

    # Hiển thị màn hình chào mừng
    console.print("\n")  # Thêm khoảng trống trước welcome screen
    display_welcome()

    # Bắt đầu chat loop
    chat_loop(agent_executor, llm)


if __name__ == "__main__":
    main()
