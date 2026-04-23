import cv2
import numpy as np
from skimage.morphology import skeletonize

img = cv2.imread("photo.jpeg", cv2.IMREAD_GRAYSCALE)

# Binary
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

# Skeletonize
skeleton = skeletonize(binary // 255).astype(np.uint8) * 255

# Find endpoints
kernel = np.ones((3, 3), np.uint8)
neighbor_count = cv2.filter2D(skeleton // 255, -1, kernel)
endpoints = (skeleton == 255) & (neighbor_count == 2)
endpoint_coords = np.column_stack(np.where(endpoints))  # (row, col)

# Draw big green arrows on original
output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

for (row, col) in endpoint_coords:
    tip = (col, row)
    tail = (col, row - 40)  # arrow comes from above pointing down
    cv2.arrowedLine(output, tail, tip, (0, 255, 0), 2, tipLength=0.4)

cv2.imwrite("endpoints.jpg", output)
cv2.imshow("Endpoints", output)
cv2.waitKey(0)
cv2.destroyAllWindows()