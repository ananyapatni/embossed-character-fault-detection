import requests
import subprocess

IP_CAM_URL = "http://192.168.1.5:8020/"   # your phone IP cam
SAVE_PATH = "/home/pi/captured_image.jpg"

def capture_image():
    try:
        response = requests.get(IP_CAM_URL, timeout=5)
        if response.status_code == 200:
            with open(SAVE_PATH, "wb") as f:
                f.write(response.content)
            print("Image saved successfully.")
        else:
            print("Failed to get image.")
    except Exception as e:
        print("Error:", e)

def run_next_script():
    subprocess.run(["python3", "/home/pi/next_script.py"])

if __name__ == "__main__":
    try:
        capture_image()
    finally:
        run_next_script()