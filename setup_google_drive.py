#!/usr/bin/env python3
"""
Setup script to help configure Google Drive model download
"""
import os

def print_instructions():
    """Print instructions for setting up Google Drive model"""
    print("=" * 60)
    print("RideSense - Google Drive Model Setup")
    print("=" * 60)
    print()
    print("To use Random Forest model from Google Drive:")
    print()
    print("1. Upload your random_forest.pkl file to Google Drive")
    print("2. Right-click on the file and select 'Get link'")
    print("3. Copy the link - it will look like:")
    print("   https://drive.google.com/file/d/FILE_ID_HERE/view?usp=sharing")
    print()
    print("4. Extract the FILE_ID_HERE part (the long string between /d/ and /view)")
    print("5. Open src/logic.py and replace 'YOUR_GOOGLE_DRIVE_FILE_ID_HERE' with your file ID")
    print()
    print("Example:")
    print("   If your link is: https://drive.google.com/file/d/1ABC123DEF456GHI789JKL/view?usp=sharing")
    print("   Your file ID is: 1ABC123DEF456GHI789JKL")
    print()
    print("6. Make sure the file is set to 'Anyone with the link can view'")
    print("7. Run your app - it will automatically download the model on first run")
    print()
    print("=" * 60)

if __name__ == "__main__":
    print_instructions()
