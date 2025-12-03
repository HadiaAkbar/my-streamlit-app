"""
Setup script for MH[FG] project
"""

import os
import sys
import subprocess
from pathlib import Path

def create_project_structure():
    """Create the complete project structure"""
    print("🚀 Setting up MH[FG] project structure...")
    
    base_dir = Path("MH[FG]")
    directories = [
        base_dir / "data" / "datasets",
        base_dir / "data" / "models",
        base_dir / "src",
        base_dir / "notebooks",
        base_dir / "assets"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {directory}")
    
    # Create empty __init__.py files
    init_files = [
        base_dir / "src" / "__init__.py",
        base_dir / "__init__.py"
    ]
    
    for init_file in init_files:
        init_file.touch()
        print(f"✓ Created: {init_file}")
    
    print("\n✅ Project structure created successfully!")

def install_requirements():
    """Install project requirements"""
    print("\n📦 Installing requirements...")
    
    requirements_file = Path("MH[FG]") / "requirements.txt"
    
    if requirements_file.exists():
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
            print("✅ Requirements installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing requirements: {e}")
    else:
        print("❌ requirements.txt not found!")

def download_nltk_data():
    """Download NLTK data files"""
    print("\n📥 Downloading NLTK data...")
    
    try:
        import nltk
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('vader_lexicon')
        print("✅ NLTK data downloaded successfully!")
    except Exception as e:
        print(f"❌ Error downloading NLTK data: {e}")

def create_sample_data():
    """Create sample dataset for testing"""
    print("\n📊 Creating sample data...")
    
    import pandas as pd
    import numpy as np
    
    sample_data = pd.DataFrame({
        'text': [
            "The government announced new economic policies today focusing on sustainable development.",
            "BREAKING: Aliens landed in New York City, government covering it up!",
            "Scientists have discovered a new species in the Amazon rainforest.",
            "Miracle cure discovered that can cure all cancers in 24 hours!",
            "According to official reports, the vaccination campaign has reached 70% of the population."
        ],
        'label': [1, 0, 1, 0, 1],  # 1 = Real, 0 = Fake
        'title': [
            "Government Announces Economic Policies",
            "Aliens Land in New York",
            "New Species Discovered",
            "Miracle Cancer Cure",
            "Vaccination Campaign Success"
        ],
        'source': [
            "Reuters",
            "Conspiracy News",
            "Science Journal",
            "Fake Health Blog",
            "Health Ministry"
        ]
    })
    
    data_dir = Path("MH[FG]") / "data" / "datasets"
    sample_data.to_csv(data_dir / "sample_dataset.csv", index=False)
    print(f"✅ Sample data created at: {data_dir / 'sample_dataset.csv'}")

def main():
    """Main setup function"""
    print("=" * 50)
    print("MH[FG] - FAKE NEWS DETECTOR PROJECT SETUP")
    print("=" * 50)
    
    # Create project structure
    create_project_structure()
    
    # Ask user if they want to install requirements
    response = input("\nDo you want to install requirements? (y/n): ")
    if response.lower() == 'y':
        install_requirements()
    
    # Download NLTK data
    response = input("\nDo you want to download NLTK data? (y/n): ")
    if response.lower() == 'y':
        download_nltk_data()
    
    # Create sample data
    response = input("\nDo you want to create sample data? (y/n): ")
    if response.lower() == 'y':
        create_sample_data()
    
    print("\n" + "=" * 50)
    print("🎉 SETUP COMPLETE!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Add your API keys to the .env file")
    print("2. Run: python train_model.py (to train models)")
    print("3. Run: streamlit run app.py (to start dashboard)")
    print("\nProject ready at: MH[FG]/")

if __name__ == "__main__":
    main()