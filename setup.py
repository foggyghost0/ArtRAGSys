#!/usr/bin/env python3
"""
Setup script for ArtRAGSys with Ollama integration.
Automates the installation and configuration process.
"""

import subprocess
import sys
import os
from pathlib import Path
import requests
import json


def run_command(command, description, check=True):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(
            command, shell=True, check=check, capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed: {result.stderr}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    print("🔍 Checking Python version...")
    if sys.version_info < (3.8, 0):
        print("❌ Python 3.8 or higher is required")
        return False
    print(
        f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} is compatible"
    )
    return True


def install_python_dependencies():
    """Install Python dependencies."""
    print("📦 Installing Python dependencies...")

    # Install from requirements.txt
    if Path("requirements.txt").exists():
        success = run_command(
            f"{sys.executable} -m pip install -r requirements.txt",
            "Installing requirements from requirements.txt",
        )
        if not success:
            return False

    # Install spaCy model
    success = run_command(
        f"{sys.executable} -m spacy download en_core_web_trf",
        "Installing spaCy transformer model",
    )

    return success


def check_ollama_installation():
    """Check if Ollama is installed and running."""
    print("🔍 Checking Ollama installation...")

    # Check if Ollama is installed
    result = subprocess.run("which ollama", shell=True, capture_output=True)
    if result.returncode != 0:
        print("❌ Ollama is not installed")
        print("Please install Ollama from: https://ollama.com/download")
        return False

    print("✅ Ollama is installed")

    # Check if Ollama is running
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama server is running")
            return True
        else:
            print("❌ Ollama server is not responding")
            return False
    except requests.RequestException:
        print("❌ Ollama server is not running")
        print("Please start Ollama with: ollama serve")
        return False


def install_ollama_models():
    """Install required Ollama models."""
    print("🤖 Installing Ollama models...")

    models = [
        ("gemma2:4b", "Gemma 3:4b text model"),
        # ("llava:latest", "LLaVA vision model"),
    ]

    for model, description in models:
        print(f"📥 Installing {description}...")
        success = run_command(
            f"ollama pull {model}",
            f"Installing {model}",
            check=False,  # Don't fail immediately on model download issues
        )
        if not success:
            print(
                f"⚠️  Failed to install {model}. You can install it later with: ollama pull {model}"
            )


def setup_database():
    """Initialize the database if needed."""
    print("🗄️  Setting up database...")

    # Check if database files exist
    db_path = Path("src/art_database.db")
    chroma_path = Path("src/chroma_db")

    if db_path.exists() and chroma_path.exists():
        print("✅ Database files already exist")
        return True

    # Check if initialization script exists
    init_script = Path("src/init_databases.py")
    if not init_script.exists():
        print("❌ Database initialization script not found")
        return False

    # Run database initialization
    success = run_command(
        f"cd src && {sys.executable} init_databases.py", "Initializing databases"
    )

    return success


def verify_installation():
    """Verify the installation by running a test."""
    print("🧪 Verifying installation...")

    # Test import
    try:
        sys.path.append("src")
        from ollama_module import OllamaArtRAG

        print("✅ Python modules import successfully")
    except ImportError as e:
        print(f"❌ Failed to import modules: {e}")
        return False

    # Test Ollama connection
    try:
        rag = OllamaArtRAG()
        print("✅ OllamaArtRAG initialization successful")
        rag.close()
        return True
    except Exception as e:
        print(f"❌ Failed to initialize OllamaArtRAG: {e}")
        return False


def create_example_scripts():
    """Create example usage scripts."""
    print("📝 Creating example scripts...")

    # Create a simple example script
    example_script = """#!/usr/bin/env python3
\"\"\"
Simple example of using ArtRAGSys with Ollama.
\"\"\"

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from ollama_module import chat_about_art, quick_art_search, analyze_artwork_image

def main():
    print("ArtRAGSys Example")
    print("=" * 30)
    
    # Example 1: Simple art search
    print("\\n1. Simple Art Search:")
    results = quick_art_search("portrait painting", k=3)
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['title']} by {result['author']}")
    
    # Example 2: Chat about art
    print("\\n2. Chat about Art:")
    response = chat_about_art("What are the characteristics of Renaissance art?")
    print(f"  {response[:200]}...")
    
    # Example 3: Analyze an image (if available)
    image_dir = Path("data/img")
    if image_dir.exists():
        images = list(image_dir.glob("*.jpg"))
        if images:
            print("\\n3. Image Analysis:")
            result = analyze_artwork_image(str(images[0]))
            if 'error' not in result:
                print(f"  Analysis: {result['image_analysis'][:200]}...")
            else:
                print(f"  Error: {result['error']}")

if __name__ == "__main__":
    main()
"""

    with open("example_usage.py", "w") as f:
        f.write(example_script)

    print("✅ Created example_usage.py")


def main():
    """Main setup function."""
    print("🎨 ArtRAGSys with Ollama Setup")
    print("=" * 40)

    steps = [
        ("Python Version Check", check_python_version),
        ("Python Dependencies", install_python_dependencies),
        ("Ollama Installation Check", check_ollama_installation),
        ("Ollama Models Installation", install_ollama_models),
        ("Database Setup", setup_database),
        ("Installation Verification", verify_installation),
        ("Example Scripts Creation", create_example_scripts),
    ]

    failed_steps = []

    for step_name, step_function in steps:
        print(f"\n📋 Step: {step_name}")
        print("-" * 30)

        success = step_function()
        if not success:
            failed_steps.append(step_name)
            print(f"⚠️  Step '{step_name}' completed with issues")
        else:
            print(f"✅ Step '{step_name}' completed successfully")

    print("\n🎯 Setup Summary")
    print("=" * 40)

    if not failed_steps:
        print("🎉 All steps completed successfully!")
        print("\nYou can now use ArtRAGSys with Ollama:")
        print("  • Run CLI: python src/art_cli.py --help")
        print("  • Run example: python example_usage.py")
        print("  • Open Jupyter notebook: multimodal_demo.ipynb")
    else:
        print(f"⚠️  Setup completed with {len(failed_steps)} issues:")
        for step in failed_steps:
            print(f"  • {step}")
        print("\nPlease resolve these issues before using the system.")

    print("\n📚 Next Steps:")
    print("  1. Start Ollama server: ollama serve")
    print("  2. Test the installation: python example_usage.py")
    print("  3. Explore the CLI: python src/art_cli.py interactive")
    print("  4. Try the Jupyter notebook: multimodal_demo.ipynb")


if __name__ == "__main__":
    main()
