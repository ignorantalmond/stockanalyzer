#!/usr/bin/env python3
"""
This script shows you which files still need to be created.
"""

import os

REMAINING_FILES = [
    "analyzer/enums.py",
    "analyzer/reddit_analyzer.py", 
    "analyzer/news_analyzer.py",
    "analyzer/stock_analyzer.py",
    "analyzer/phase_detector.py",
    "analyzer/options_strategy.py",
    "analyzer/meme_stock_analyzer.py",
    "analyzer/display_manager.py"
]

def main():
    print("📋 REMAINING FILES TO CREATE:")
    print("=" * 50)
    
    missing_files = []
    for i, filename in enumerate(REMAINING_FILES, 1):
        if os.path.exists(filename):
            print(f"✅ {filename}")
        else:
            print(f"❌ {filename}")
            missing_files.append(filename)
    
    if missing_files:
        print(f"\n💡 {len(missing_files)} files still needed")
        print("\n🎯 Next: Ask the assistant for these files:")
        for i, filename in enumerate(missing_files[:3], 1):
            print(f"{i}. {filename}")
        
        if len(missing_files) > 3:
            print(f"... and {len(missing_files) - 3} more")
    else:
        print("\n🎉 ALL FILES CREATED! Ready to run:")
        print("python main.py")

if __name__ == "__main__":
    main()
