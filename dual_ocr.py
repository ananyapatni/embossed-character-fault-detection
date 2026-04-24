import cv2
import numpy as np
from paddleocr import PaddleOCR
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SAVE_PATH = r"D:\Embossing\images\captured_image.jpg"

# ── load & pre-rotate (camera is always mounted 90 CW off) ───────────────────
img = cv2.imread(SAVE_PATH)
if img is None:
    raise FileNotFoundError(f"No image at {SAVE_PATH}")
rotated_bgr = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
gray = cv2.cvtColor(rotated_bgr, cv2.COLOR_BGR2GRAY)
denoised = cv2.bilateralFilter(gray, d=11, sigmaColor=40, sigmaSpace=40)

# ── compute tophat & blackhat for display ─────────────────────────────────────
k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
tophat_img   = cv2.morphologyEx(denoised, cv2.MORPH_TOPHAT,   k)
blackhat_img = cv2.morphologyEx(denoised, cv2.MORPH_BLACKHAT, k)

# ── preprocessing variants ───────────────────────────────────────────────────
def clahe(img, clip, tile):
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(img)

# global normalise first — fixes raking-light images with full-image brightness gradients
normalised = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)

variants = {
    "gentle":      clahe(denoised,    2.0, 8),
    "medium":      clahe(denoised,    3.5, 6),
    "strong":      clahe(denoised,    5.0, 4),
    "norm_gentle": clahe(normalised,  2.0, 8),
    "norm_medium": clahe(normalised,  3.5, 6),
    "norm_strong": clahe(normalised,  5.0, 4),
}

# ── OCR ───────────────────────────────────────────────────────────────────────
ocr = PaddleOCR(use_angle_cls=False, lang='en',
                det_db_thresh=0.05, det_db_box_thresh=0.15,
                det_db_unclip_ratio=2.5,
                use_gpu=False, show_log=False)

def rotate_img(img, angle):
    if angle == 0:   return img
    if angle == 90:  return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180: return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 270: return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

def run_ocr(gray_img, name):
    h, w = gray_img.shape
    scaled = cv2.resize(gray_img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    scaled = cv2.morphologyEx(scaled, cv2.MORPH_CLOSE, k5, iterations=1)
    PAD = 20
    padded = cv2.copyMakeBorder(scaled, PAD, PAD, PAD, PAD,
                                cv2.BORDER_CONSTANT, value=255)
    rgb = cv2.cvtColor(padded, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(f"_tmp_{name}.png", rgb)
    result = ocr.ocr(f"_tmp_{name}.png", cls=False)
    if result and result[0]:
        texts = [(r[1][0], r[1][1]) for r in result[0]]
        return texts, sum(c for _, c in texts)
    return [], 0.0

all_results = []
for strength, enhanced in variants.items():
    for polarity, processed in [("normal", enhanced), ("inverted", cv2.bitwise_not(enhanced))]:
        for angle in [0, 90, 180, 270]:
            rotated = rotate_img(processed, angle)
            texts, total_conf = run_ocr(rotated, f"{strength}_{polarity}_{angle}")
            if total_conf > 0:
                all_results.append((texts, total_conf))

if not all_results:
    final_text = "Nothing detected"
    final_conf = 0.0
else:
    # sort by average per-box confidence (prevents multi-box sums dominating)
    all_results.sort(key=lambda x: -(x[1] / max(1, len(x[0]))))
    best_texts, best_conf = all_results[0]
    best_str = "".join(t for t, _ in best_texts)

    if best_conf / max(1, len(best_texts)) < 0.85:
        # low confidence — use majority vote (length x count) instead
        vote_scores = defaultdict(float)
        vote_best   = {}
        for texts, conf in all_results:
            joined = "".join(t for t, _ in texts)
            if not joined:
                continue
            vote_scores[joined] += len(joined)
            if joined not in vote_best or conf > vote_best[joined][1]:
                vote_best[joined] = (texts, conf)
        best_str = max(vote_scores, key=lambda k: vote_scores[k])
        best_texts, best_conf = vote_best[best_str]

    # prefix/suffix correction: strip false-positive chars at either end
    for texts, conf in all_results:
        candidate = "".join(t for t, _ in texts)
        if conf > best_conf and candidate:
            if best_str.startswith(candidate) or best_str.endswith(candidate):
                best_str   = candidate
                best_texts = texts
                best_conf  = conf
                break

    final_text = " ".join(t for t, _ in best_texts)
    final_conf = best_conf

# ── display ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 6), facecolor='#1a1a2e')
fig.suptitle("Embossed Text OCR", color='white', fontsize=16, fontweight='bold')
gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.05)

def show(ax, image, title, cmap=None):
    ax.imshow(image, cmap=cmap)
    ax.set_title(title, color='white', fontsize=11, pad=8)
    ax.axis('off')

show(fig.add_subplot(gs[0]), cv2.cvtColor(rotated_bgr, cv2.COLOR_BGR2RGB),
     "Captured (corrected)")
show(fig.add_subplot(gs[1]), tophat_img,   "Top-hat",   cmap='gray')
show(fig.add_subplot(gs[2]), blackhat_img, "Black-hat", cmap='gray')

ax_result = fig.add_subplot(gs[3])
ax_result.set_facecolor('#0f3460')
ax_result.text(0.5, 0.55, final_text,
               ha='center', va='center', fontsize=28, fontweight='bold',
               color='#e94560', transform=ax_result.transAxes)
ax_result.text(0.5, 0.25, f"conf: {final_conf:.2f}",
               ha='center', va='center', fontsize=13,
               color='#aaaaaa', transform=ax_result.transAxes)
ax_result.set_title("OCR Result", color='white', fontsize=11, pad=8)
ax_result.axis('off')

plt.tight_layout()
plt.show()

print(f"Result: '{final_text}'  (confidence {final_conf:.2f})")
