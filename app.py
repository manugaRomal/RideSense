"""
RideSense - Vehicle Condition Prediction System
Main application entry point using separated logic and UI modules
"""
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.ui import RideSenseUI

def main():
    """Main function to run the RideSense application"""
    ui = RideSenseUI()
    ui.run()

if __name__ == "__main__":
    main()
