"""
main.py

What this file does:
    Entry point of the FaceVault application.
    Displays an interactive console menu and dispatches to the
    appropriate module.

Input:  User choice from the console.
Output: Runs the selected pipeline stage.
How it fits into the project:
    Ties all modules together into a single, easy-to-use application.
"""

import sys
from src.utils import setup_project_structure, get_project_status


def display_menu():
    """Print the main menu."""
    print()
    print("=" * 48)
    print("       FACEVAULT")
    print("   AI FACIAL RECOGNITION SYSTEM")
    print("=" * 48)
    print("  1.  Collect Face Dataset")
    print("  2.  Preprocess Dataset")
    print("  3.  Generate Face Encodings")
    print("  4.  Recognize an Image")
    print("  5.  Start Real-Time Recognition")
    print("  6.  Evaluate Model")
    print("  7.  Show Project Status")
    print("  8.  Exit")
    print("-" * 48)


def show_status():
    """Display a summary of the current project state."""
    s = get_project_status()
    print()
    print("--- Project Status ---")
    print(f"  Registered people (raw)  : {s['registered_people']}")
    print(f"  Raw images               : {s['raw_images']}")
    print(f"  Processed images         : {s['processed_images']}")
    print(f"  Encoding model ready     : {'Yes' if s['encoding_exists'] else 'No'}")
    print(f"  Test people              : {s['test_people']}")
    print(f"  Test images              : {s['test_images']}")
    print(f"  Evaluation completed     : {'Yes' if s['evaluation_done'] else 'No'}")
    print()


def main():
    """Main application loop."""
    setup_project_structure()

    while True:
        display_menu()
        try:
            choice = input("  Enter your choice: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nExiting FaceVault.  Goodbye!")
            sys.exit(0)

        if choice == "1":
            print("\n--- Dataset Collection ---")
            try:
                from src import collect_faces
                collect_faces.main()
            except Exception as e:
                print(f"ERROR: {e}")

        elif choice == "2":
            print("\n--- Preprocessing ---")
            try:
                from src import preprocess
                preprocess.main()
            except Exception as e:
                print(f"ERROR: {e}")

        elif choice == "3":
            print("\n--- Face Encoding ---")
            try:
                from src import encode_faces
                encode_faces.main()
            except Exception as e:
                print(f"ERROR: {e}")

        elif choice == "4":
            print("\n--- Image Recognition ---")
            image_path = input("  Enter the path to the image: ").strip()
            if not image_path:
                print("ERROR: No path entered.")
                continue
            try:
                from src import recognize_image
                recognize_image.main(image_path)
            except Exception as e:
                print(f"ERROR: {e}")

        elif choice == "5":
            print("\n--- Real-Time Webcam Recognition ---")
            try:
                from src import recognize_webcam
                recognize_webcam.main()
            except Exception as e:
                print(f"ERROR: {e}")

        elif choice == "6":
            print("\n--- Model Evaluation ---")
            try:
                from src import evaluate
                evaluate.main()
            except Exception as e:
                print(f"ERROR: {e}")

        elif choice == "7":
            show_status()

        elif choice == "8":
            print("\nExiting FaceVault.  Goodbye!")
            sys.exit(0)

        else:
            print("Invalid choice.  Please enter a number between 1 and 8.\n")


if __name__ == "__main__":
    main()
