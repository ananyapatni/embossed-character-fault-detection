import requests
import subprocess
import os

IP_CAM_URL = "http://192.168.1.5:8020/shot.jpg"   # phone IP cam

# Windows-safe save path (CHANGE if needed)
SAVE_PATH = r"D:\Embossing\images\captured_image.jpg"

# Next script (also Windows path)
NEXT_SCRIPT = r"D:\Embossing\dual_ocr.py"


def capture_image():
    try:
        response = requests.get(IP_CAM_URL, timeout=5)

        if response.status_code == 200:
            # ensure folder exists
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

            with open(SAVE_PATH, "wb") as f:
                f.write(response.content)

            print(f"Image saved successfully at {SAVE_PATH}")
        else:
            print("Failed to get image. Status:", response.status_code)

    except Exception as e:
        print("Error while capturing image:", e)


VENV_PYTHON = r"D:\Embossing\.venv\Scripts\python.exe"

def run_next_script():
    try:
        subprocess.run([VENV_PYTHON, NEXT_SCRIPT], check=True)
    except Exception as e:
        print("Error running next script:", e)


if __name__ == "__main__":
    capture_image()
    run_next_script()