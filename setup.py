"""
Run this ONCE before launching the app to train the PyTorch difficulty classifier.

Usage:
    python setup.py
"""

import os
import sys

def main():
    print("=" * 60)
    print("  AdaptML — First-time setup")
    print("=" * 60)

    # 1. Check .env
    if not os.path.exists(".env"):
        if os.path.exists(".env.example"):
            import shutil
            shutil.copy(".env.example", ".env")
            print("\n⚠️  Created .env from .env.example.")
            print("    Edit .env and add your GROQ_API_KEY before running the app!\n")
        else:
            print("\n⚠️  No .env file found. Create one with GROQ_API_KEY=...\n")

    # 2. Create directories
    os.makedirs("data",   exist_ok=True)
    os.makedirs("models", exist_ok=True)
    print("✅ Directories created: data/, models/")

    # 3. Generate dataset
    print("\n--- Step 1: Generating training dataset ---")
    from src.generate_dataset import generate_dataset
    generate_dataset()

    # 4. Train model
    print("\n--- Step 2: Training PyTorch difficulty classifier ---")
    print("    (This downloads ~80 MB sentence-transformer on first run)")
    from src.train_model import train
    stats = train()

    print("\n" + "=" * 60)
    print(f"  ✅ Setup complete!")
    print(f"  Model accuracy: {round(stats['accuracy']*100, 1)}%")
    print(f"  Model saved to: models/difficulty_model.pt")
    print(f"  Stats saved to: models/training_stats.json")
    print("=" * 60)
    print("\nNow run:  streamlit run app.py")


if __name__ == "__main__":
    main()
