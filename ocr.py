import cv2
import numpy as np
from paddleocr import PaddleOCR

# Initialize PaddleOCR (2.7 API)
ocr = PaddleOCR(use_angle_cls=True, lang='en')

def run_raw_ocr(image_path):
    # Load image
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Image not found. Check filename and path.")

    # Run OCR
    result = ocr.ocr(img, cls=True)

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
        print("Bounding Box:", bbox)
        print("-" * 40)

        pts = np.array(bbox).astype(int)
        cv2.polylines(img, [pts], True, (0, 255, 0), 2)

        cv2.putText(
            img,
            text,
            (pts[0][0], pts[0][1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        # Show result image
        cv2.imshow("Raw OCR Result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_raw_ocr("photo.jpeg")