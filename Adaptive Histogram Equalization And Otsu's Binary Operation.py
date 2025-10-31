import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Image Read
img = cv2.imread("D:\\Academic\\Semester\\Digital Image Lab\\balloons_noisy.png")
cv2.imshow("Original Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2. Image Conversion [BGR2GRAY]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale Image", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 3. Image Histogram Conversion
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
plt.title("Grayscale Histogram")
plt.xlabel('Pixel Value')
plt.ylabel("Frequency")
plt.plot(hist)
plt.show()

# 4. Image Histogram Equalization
equalized = cv2.equalizeHist(gray)
hist_eq = cv2.calcHist([equalized], [0], None, [256], [0, 256])
plt.title("Equalized Histogram")
plt.xlabel('Pixel Value')
plt.ylabel("Frequency")
plt.plot(hist_eq)
plt.show()
cv2.imshow("Equalized Image", equalized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 5. Equalized Image already in Grayscale – displaying for confirmation
cv2.imshow("Equalized Grayscale Image", equalized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 6. CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=40)
clahe_img = clahe.apply(equalized)
cv2.imshow("CLAHE Image", clahe_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 7. Threshold and Adaptive Threshold
# Simple binary threshold
_, binary_thresh = cv2.threshold(clahe_img, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("Simple Threshold", binary_thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Adaptive Threshold
adaptive_thresh = cv2.adaptiveThreshold(clahe_img, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
cv2.imshow("Adaptive Threshold", adaptive_thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 8. OTSU Binarization
_, otsu_thresh = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("OTSU Threshold", otsu_thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

