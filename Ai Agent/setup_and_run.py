"""
Setup and Run Script
====================
Automated setup for the Enhanced AI Assistant
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and print status."""
    print(f"\n{'='*50}")
    print(f"🔄 {description}")
    print(f"{'='*50}")
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"⚠️  Warning: {description} may have had issues")
    return result.returncode == 0

def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           ENHANCED AI ASSISTANT - SETUP & RUN                    ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Install requirements
    print("\n📦 Step 1: Installing Python dependencies...")
    run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing requirements")
    
    # Step 2: Download spaCy model
    print("\n📦 Step 2: Downloading spaCy English model...")
    run_command(f"{sys.executable} -m spacy download en_core_web_sm", "Downloading spaCy model")
    
    # Step 3: Create directories
    print("\n📁 Step 3: Creating directories...")
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    print("✅ Directories created")
    
    # Step 4: Check if model needs training
    if not os.path.exists("models/intent_classifier"):
        print("\n🧠 Step 4: Training intent classifier...")
        print("(This may take a few minutes)")
        success = run_command(f"{sys.executable} train_intent_classifier.py", "Training intent classifier")
        
        if success and os.path.exists("evaluate_models.py"):
            print("\n📊 Step 5: Running model comparison...")
            run_command(f"{sys.executable} evaluate_models.py", "Comparing models")
    else:
        print("\n✅ Step 4: Intent classifier already trained!")
        print("   (Delete models/intent_classifier to retrain)")
    
    # Step 5: Test entity extraction
    print("\n🧪 Step 6: Testing entity extraction...")
    run_command(f"{sys.executable} entity_extractor.py", "Testing entity extraction")
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                    SETUP COMPLETE! 🎉                            ║
╚══════════════════════════════════════════════════════════════════╝

To start the Enhanced AI Assistant:

    uvicorn app_enhanced:app --reload --host 0.0.0.0 --port 8000

Then open: http://localhost:8000/docs

Or with UI: http://localhost:8000/ui

API Endpoints:
    POST /infer     - Process text input
    POST /voice     - Process voice input  
    GET  /context   - Get user session
    GET  /history   - Get conversation history
    GET  /reset     - Reset user session
    """)
    
    # Ask to start server
    response = input("\n🚀 Start the server now? (y/n): ").lower().strip()
    if response == 'y':
        print("\n🚀 Starting Enhanced AI Assistant...")
        subprocess.run(f"{sys.executable} -m uvicorn app_enhanced:app --reload --host 0.0.0.0 --port 8000", shell=True)

if __name__ == "__main__":
    main()

