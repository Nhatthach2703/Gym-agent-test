#!/usr/bin/env python3
"""
Test script để kiểm tra Poetry environment và RAG dependencies
"""


def test_imports():
    """Test tất cả imports cần thiết"""
    print("🔍 Testing imports...")

    try:
        import langchain

        print(f"✅ langchain: {langchain.__version__}")
    except ImportError as e:
        print(f"❌ langchain: {e}")
        return False

    try:
        import langchain_google_genai

        print("✅ langchain_google_genai: OK")
    except ImportError as e:
        print(f"❌ langchain_google_genai: {e}")
        return False

    try:
        import chromadb

        print(f"✅ chromadb: {chromadb.__version__}")
    except ImportError as e:
        print(f"❌ chromadb: {e}")
        return False

    try:
        import numpy as np

        print(f"✅ numpy: {np.__version__}")
    except ImportError as e:
        print(f"❌ numpy: {e}")
        return False

    try:
        import sentence_transformers

        print(f"✅ sentence_transformers: {sentence_transformers.__version__}")
    except ImportError as e:
        print(f"❌ sentence_transformers: {e}")
        return False

    try:
        import rich

        print("✅ rich: OK")
    except ImportError as e:
        print(f"❌ rich: {e}")
        return False

    try:
        import dotenv

        print("✅ python-dotenv: OK")
    except ImportError as e:
        print(f"❌ python-dotenv: {e}")
        return False

    return True


def test_environment():
    """Test environment variables"""
    print("\n🌍 Testing environment...")

    import os
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    if api_key:
        print("✅ GOOGLE_API_KEY: Found")
        print(f"   Key preview: {api_key[:10]}...{api_key[-4:]}")
        return True
    else:
        print("❌ GOOGLE_API_KEY: Not found")
        print("   Please create .env file with GOOGLE_API_KEY=your_key")
        return False


def check_poetry_setup():
    """Kiểm tra Poetry setup"""
    print("\n📦 Checking Poetry setup...")

    import sys
    import subprocess

    try:
        # Check if running in poetry environment
        result = subprocess.run(
            ["poetry", "env", "info", "--path"],
            capture_output=True,
            text=True,
            check=True,
        )
        poetry_path = result.stdout.strip()
        current_path = sys.executable

        if poetry_path in current_path:
            print("✅ Running in Poetry virtual environment")
            print(f"   Poetry env: {poetry_path}")
            return True
        else:
            print("⚠️ Not running in Poetry environment")
            print("   Run: poetry shell")
            return False

    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Poetry not found or not in Poetry environment")
        print("   Install Poetry: https://python-poetry.org/docs/#installation")
        return False


def main():
    """Main test function"""
    print("🚀 GYM AGENT RAG - Poetry Environment Test\n")

    # Test Poetry setup
    poetry_ok = check_poetry_setup()

    # Test imports
    imports_ok = test_imports()

    # Test environment
    env_ok = test_environment()

    print(f"\n📊 Test Results:")
    print(f"Poetry Setup: {'✅ PASSED' if poetry_ok else '❌ FAILED'}")
    print(f"Dependencies: {'✅ PASSED' if imports_ok else '❌ FAILED'}")
    print(f"Environment: {'✅ PASSED' if env_ok else '❌ FAILED'}")

    if poetry_ok and imports_ok and env_ok:
        print("\n🎉 All tests passed! Ready to run RAG Gym Agent!")
        print("\nNext steps:")
        print("  poetry run python src/gym_agent_test/main_RAG.py")
    else:
        print("\n⚠️ Some tests failed. Fix issues above.")

        if not poetry_ok:
            print("\n📝 Poetry setup:")
            print(
                "  1. Install Poetry: curl -sSL https://install.python-poetry.org | python3 -"
            )
            print("  2. poetry install")
            print("  3. poetry shell")

        if not imports_ok:
            print("\n📝 Dependencies:")
            print("  poetry install")

        if not env_ok:
            print("\n📝 Environment:")
            print("  Create .env file with GOOGLE_API_KEY=your_key")


if __name__ == "__main__":
    main()
