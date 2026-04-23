
import cv2
import numpy as np
from skimage.morphology import skeletonize


def preprocess_embossed_text(image_path, output_path="output.jpg", min_blob_area=300, max_gap=50):
    """
    Preprocessing pipeline for embossed/stamped text images.
    Steps:
      1. Load & convert to grayscale
      2. Binary threshold
      3. Remove noise (keep only large blobs = actual letters)
      4. Skeletonize to find stroke centerlines
      5. Find stroke endpoints
      6. For each endpoint, walk to letter edge then across gap
      7. If gap hits another stroke, fill it with a thick black bar
      8. Return clean black-on-white result
    """

    # ── 1. Load ──────────────────────────────────────────────────────────────
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # ── 2. Binary threshold ──────────────────────────────────────────────────
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # ── 3. Remove noise blobs ────────────────────────────────────────────────
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    clean = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > min_blob_area:
            clean[labels == i] = 255

    # ── 4. Skeletonize ───────────────────────────────────────────────────────
    skeleton = skeletonize(clean // 255).astype(np.uint8) * 255

    # ── 5. Find endpoints (pixels with exactly 1 skeleton neighbor) ──────────
    kernel = np.ones((3, 3), np.uint8)
    neighbor_count = cv2.filter2D(skeleton // 255, -1, kernel)
    endpoints_mask = (skeleton == 255) & (neighbor_count == 2)
    endpoint_coords = np.column_stack(np.where(endpoints_mask))

    # ── 6. Direction: trace back along skeleton ──────────────────────────────
    def get_stroke_direction(skel, row, col, lookback=12):
        visited = set()
        path = [(row, col)]
        visited.add((row, col))
        for _ in range(lookback):
            cr, cc = path[-1]
            found = False
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = cr + dr, cc + dc
                    if (nr, nc) in visited:
                        continue
                    if 0 <= nr < skel.shape[0] and 0 <= nc < skel.shape[1]:
                        if skel[nr, nc] == 255:
                            path.append((nr, nc))
                            visited.add((nr, nc))
                            found = True
                            break
                if found:
                    break
            if not found:
                break
        if len(path) < 2:
            return (0, 0)
        r1, c1 = path[-1]
        r0, c0 = path[0]
        dr = r0 - r1
        dc = c0 - c1
        length = np.sqrt(dr ** 2 + dc ** 2)
        if length == 0:
            return (0, 0)
        return (dr / length, dc / length)

    # ── 7. Estimate local stroke width ───────────────────────────────────────
    def estimate_stroke_width(clean_img, r, c):
        widths = []
        for direction in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            dr, dc = direction
            w = 1
            for sign in [1, -1]:
                for step in range(1, 30):
                    nr = r + sign * dr * step
                    nc = c + sign * dc * step
                    if not (0 <= nr < clean_img.shape[0] and 0 <= nc < clean_img.shape[1]):
                        break
                    if clean_img[nr, nc] == 0:
                        break
                    w += 1
            widths.append(w)
        return max(6, min(20, int(np.median(widths))))

    # ── 8. Bridge gaps ───────────────────────────────────────────────────────
    result = clean.copy()
    bridges = 0

    for (r, c) in endpoint_coords:
        dr, dc = get_stroke_direction(skeleton, r, c, lookback=12)
        if dr == 0 and dc == 0:
            continue

        # Walk forward until we exit the letter stroke
        cr, cc = float(r), float(c)
        edge_r, edge_c = r, c
        for _ in range(60):
            cr += dr
            cc += dc
            nr, nc = int(round(cr)), int(round(cc))
            if not (0 <= nr < clean.shape[0] and 0 <= nc < clean.shape[1]):
                break
            if clean[nr, nc] == 0:
                edge_r, edge_c = nr, nc
                break

        # Continue until we hit another stroke
        gap_path = []
        hit = False
        hit_r, hit_c = edge_r, edge_c
        for _ in range(max_gap):
            cr += dr
            cc += dc
            nr, nc = int(round(cr)), int(round(cc))
            if not (0 <= nr < clean.shape[0] and 0 <= nc < clean.shape[1]):
                break
            if clean[nr, nc] == 255:
                hit = True
                hit_r, hit_c = nr, nc
                break
            gap_path.append((nr, nc))

        if hit and len(gap_path) >= 1:
            stroke_w = estimate_stroke_width(clean, edge_r, edge_c)
            cv2.line(result, (edge_c, edge_r), (hit_c, hit_r), 255, thickness=stroke_w)
            bridges += 1

    print(f"Gaps bridged: {bridges}")

    # ── 9. Final output: black letters on white ──────────────────────────────
    output = cv2.bitwise_not(result)
    cv2.imwrite(output_path, output)
    print(f"Saved to: {output_path}")
    return output


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    preprocess_embossed_text(
        image_path="photo.jpeg",
        output_path="output.jpg",
        min_blob_area=300,   # increase to filter more noise
        max_gap=50           # max gap in px to bridge
    )

preprocess_embossed_text("photo2.jpg", "output5.jpg")