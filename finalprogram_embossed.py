#imports
import cv2
import numpy as np
import pytesseract
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize

# Point pytesseract at the Tesseract binary (remove if it's on your PATH)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

img = cv2.imread(r'D:\Embossing\images\top.jpeg')

# --- ROI SELECTION ---
# Optional: shrink the display window if the image is larger than your screen.
# selectROI shows the image at native size, so big images can go off-screen.
display_img = img.copy()
max_display_dim = 1200
h0, w0 = display_img.shape[:2]
scale = 1.0
if max(h0, w0) > max_display_dim:
    scale = max_display_dim / max(h0, w0)
    display_img = cv2.resize(display_img, (int(w0 * scale), int(h0 * scale)))

# Drag a box with the mouse, then press ENTER or SPACE to confirm.
# Press 'c' to cancel and process the whole image.
roi = cv2.selectROI("Select ROI (ENTER=confirm, c=cancel)", display_img,
                    showCrosshair=True, fromCenter=False)
cv2.destroyWindow("Select ROI (ENTER=confirm, c=cancel)")

x, y, w, h = roi
if w == 0 or h == 0:
    print("No ROI selected — processing full image.")
    cropped = img
else:
    # Scale coords back up to original image size if we downscaled for display
    if scale != 1.0:
        x, y, w, h = (int(v / scale) for v in (x, y, w, h))
    cropped = img[y:y + h, x:x + w]
    print(f"ROI selected: x={x}, y={y}, w={w}, h={h}")

# --- ORIGINAL PIPELINE, NOW OPERATING ON `cropped` ---
# Convert to grayscale for morphological operations
gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

# 1. Define the kernel (Size 15-25 is usually good for license plate text)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

# 2. Apply White Top-Hat (extracts bright/raised features from dark background)
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

# 3. Contrast Enhancement
enhanced = cv2.normalize(tophat, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

# 4. Thresholding to create Binary Image
_, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 5. INVERT: Make text BLACK and background WHITE
nakki = cv2.bitwise_not(thresh)

# 6. Tesseract OCR Attempt
custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/'
text = pytesseract.image_to_string(nakki, config=custom_config)

# --- Visualization & Output ---
print("-" * 30)
print(f"Recognized Text: {text.strip()}")
print("-" * 30)

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
plt.title("0. Selected ROI")
plt.subplot(1, 3, 2)
plt.imshow(tophat, cmap='gray')
plt.title("1. Top-Hat")
plt.subplot(1, 3, 3)
plt.imshow(nakki, cmap='gray')
plt.title("2. Inverted (OCR input)")
plt.show()