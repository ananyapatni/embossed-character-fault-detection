import cv2
import numpy as np
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en')

def run_raw_ocr(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Image not found.")

    # ---- Interactive ROI Selection ----
    roi = cv2.selectROI("Select ROI (Press ENTER when done)", img, fromCenter=False, showCrosshair=True)
    print("Returned ROI:", roi)  
    cv2.destroyWindow("Select ROI (Press ENTER when done)")

    x, y, w, h = roi

    if w == 0 or h == 0:
        print("No ROI selected.")
        return

    cropped = img[y:y+h, x:x+w]

    cv2.imshow("Cropped ROI", cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Run OCR on cropped region
    result = ocr.ocr(cropped, cls=True)

    print("\n===== OCR RESULTS =====\n")

    if not result or result[0] is None:
        print("No text detected.")
        return

    for line in result[0]:
        bbox = line[0]
        text = line[1][0]
        confidence = line[1][1]

        print("Text:", text)
        print("Confidence:", round(confidence, 4))
        print("-" * 40)

        pts = np.array(bbox).astype(int)
        cv2.polylines(cropped, [pts], True, (0, 255, 0), 2)

    cv2.imshow("OCR Result on ROI", cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_raw_ocr("images/i1.jpeg")