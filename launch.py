import sys
import shared

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(f"Launching with arguments: {' '.join(sys.argv[1:])}")
    from ui import create_ui
    create_ui()
