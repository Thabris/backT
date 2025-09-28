"""
Launch the BackT Streamlit Web Interface

Simple launcher script to start the web interface.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch Streamlit app"""

    # Get the path to the streamlit app
    app_path = Path(__file__).parent / "streamlit_app.py"

    print("🚀 Launching BackT Streamlit Web Interface...")
    print(f"📁 App location: {app_path}")
    print("🌐 Opening browser at: http://localhost:8501")
    print("\n💡 To stop the server, press Ctrl+C in this terminal")
    print("=" * 50)

    try:
        # Launch streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(app_path),
            "--server.headless", "false",
            "--server.runOnSave", "true"
        ])
    except KeyboardInterrupt:
        print("\n👋 Streamlit server stopped.")
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")
        print("\n💡 Make sure Streamlit is installed:")
        print("   pip install streamlit plotly")

if __name__ == "__main__":
    main()