import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def load_templates_from_folder(folder_path):
    templates = {}
    if not os.path.exists(folder_path):
        return templates
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            char_name = os.path.splitext(filename)[0]
            template_path = os.path.join(folder_path, filename)
            template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if template_img is not None:
                _, template_thresh = cv2.threshold(template_img, 128, 255, cv2.THRESH_BINARY_INV)
                templates[char_name] = template_thresh
    return templates

def process_license_plate_debug(image, templates):
    os.makedirs("debug", exist_ok=True)
    h, w, _ = image.shape
    scale = 500 / w
    new_h, new_w = int(h * scale), int(w * scale)
    original_resized = cv2.resize(image, (new_w, new_h))
    cv2.imwrite("debug/1_original_resized.jpg", original_resized)
    cv2.imshow("1. Original Resized Image", original_resized)
    cv2.waitKey(0)
    gray = cv2.cvtColor(original_resized, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("debug/2a_grayscale.jpg", gray)
    cv2.imshow("2a. Grayscale Conversion", gray)
    cv2.waitKey(0)
    filtered_gray = cv2.bilateralFilter(gray, 11, 13, 13)
    cv2.imwrite("debug/2b_bilateral_filter.jpg", filtered_gray)
    cv2.imshow("2b. Bilateral Filter (Noise Reduction)", filtered_gray)
    cv2.waitKey(0)
    sobel_x = cv2.Sobel(filtered_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(filtered_gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_combined = np.uint8(sobel_combined / sobel_combined.max() * 255)
    cv2.imwrite("debug/3_sobel_combined.jpg", sobel_combined)
    cv2.imshow("3. Sobel Combined Edges", sobel_combined)
    cv2.waitKey(0)
    sobel_x_abs = cv2.convertScaleAbs(sobel_x)
    cv2.imwrite("debug/3a_horizontal_edges.jpg", sobel_x_abs)
    cv2.imshow("3a. Horizontal Edges (Sobel X)", sobel_x_abs)
    cv2.waitKey(0)
    gradient = sobel_x_abs
    gradient_enhanced = cv2.dilate(gradient, np.ones((2,2), np.uint8), iterations=1)
    cv2.imwrite("debug/3b_enhanced_edges.jpg", gradient_enhanced)
    cv2.imshow("3b. Enhanced Horizontal Edges", gradient_enhanced)
    cv2.waitKey(0)
    binary = cv2.adaptiveThreshold(gradient_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite("debug/4_binarization.jpg", binary)
    cv2.imshow("4. Binarization (Adaptive Threshold)", binary)
    cv2.waitKey(0)
    img_height, img_width = binary.shape
    kernel_width = max(10, img_width // 40)
    kernel_height = max(3, img_height // 80)
    small_horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    horizontal_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, small_horizontal_kernel)
    cv2.imwrite("debug/5a_horizontal_closed.jpg", horizontal_closed)
    cv2.imshow("5a. Small Horizontal Closing (Preserve Lines)", horizontal_closed)
    cv2.waitKey(0)
    line_contours, _ = cv2.findContours(horizontal_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = cv2.cvtColor(horizontal_closed, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_img, line_contours, -1, (0, 255, 0), 2)
    cv2.imwrite("debug/5b_line_contours.jpg", contour_img)
    cv2.imshow("5b. Individual Line Contours", contour_img)
    cv2.waitKey(0)
    candidate_lines = []
    for i, contour in enumerate(line_contours):
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio > 0.5 and w > 10 and h > 2:
            candidate_lines.append((contour, area, aspect_ratio, w, h))
    candidate_lines.sort(key=lambda x: x[1], reverse=True)
    plate_lines = [candidate[0] for candidate in candidate_lines[:2] if candidate[1] > 200]
    if len(plate_lines) >= 1:
        all_points = np.vstack(plate_lines)
        plate_contour = cv2.convexHull(all_points)
        combined_img = cv2.cvtColor(horizontal_closed, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(combined_img, [plate_contour], -1, (0, 0, 255), 3)
        cv2.imwrite("debug/5c_combined_plate.jpg", combined_img)
        cv2.imshow("5c. Combined Plate Region", combined_img)
        cv2.waitKey(0)
    else:
        cv2.destroyAllWindows()
        return "", []
    x, y, w, h = cv2.boundingRect(plate_contour)
    if w == 0 or h == 0:
        cv2.destroyAllWindows()
        return "", []
    plate_crop = filtered_gray[y:y+h, x:x+w]
    cv2.imwrite("debug/7_cropped_plate.jpg", plate_crop)
    cv2.imshow("7. Cropped Plate", plate_crop)
    cv2.waitKey(0)
    scale_factor = 3
    enlarged_height = int(plate_crop.shape[0] * scale_factor)
    enlarged_width = int(plate_crop.shape[1] * scale_factor)
    plate_enlarged = cv2.resize(plate_crop, (enlarged_width, enlarged_height), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite("debug/7a_enlarged_plate.jpg", plate_enlarged)
    cv2.imshow("7a. Enlarged Plate (Lanczos)", plate_enlarged)
    cv2.waitKey(0)
    canny_edges = cv2.Canny(plate_enlarged, 30, 100)
    cv2.imwrite("debug/7b_canny_edges.jpg", canny_edges)
    cv2.imshow("7b. Canny Edges", canny_edges)
    cv2.waitKey(0)
    kernel = np.ones((3,3), np.uint8)
    thickened_edges = cv2.dilate(canny_edges, kernel, iterations=1)
    cv2.imwrite("debug/7c_thickened_edges.jpg", thickened_edges)
    cv2.imshow("7c. Thickened Edges", thickened_edges)
    cv2.waitKey(0)
    plate_binary = thickened_edges
    cv2.imwrite("debug/8_final_binary.jpg", plate_binary)
    cv2.imshow("8. Final Binary for Segmentation", plate_binary)
    cv2.waitKey(0)
    char_contours, _ = cv2.findContours(plate_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    char_bounding_boxes = []
    min_height = 30
    max_height = 300
    min_width = 9
    max_width = 120
    for cnt in char_contours:
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        if min_height < ch < max_height and min_width < cw < max_width:
            char_bounding_boxes.append((cx, cy, cw, ch))
    char_bounding_boxes.sort(key=lambda b: b[0])
    segmented_chars_display = cv2.cvtColor(plate_binary, cv2.COLOR_GRAY2BGR)
    recognized_plate = ""
    for i, (cx, cy, cw, ch) in enumerate(char_bounding_boxes):
        cv2.rectangle(segmented_chars_display, (cx, cy), (cx + cw, cy + ch), (0, 255, 0), 2)
        recognized_plate += f"C{i+1} "
    cv2.imwrite("debug/9_segmented_characters.jpg", segmented_chars_display)
    cv2.imshow("9. Segmented Characters", segmented_chars_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return recognized_plate.strip(), []

if __name__ == "__main__":
    image_path = r"D:\Academic\Semester\Digital Image Lab\Bangla License Plate\train\images\73_jpg.rf.5fbd74b878b1c8e3fdf142a56334466c.jpg"
    templates_folder_path = "templates"
    templates = {}
    if not os.path.exists(image_path):
        pass
    else:
        vehicle_image = cv2.imread(image_path)
        if vehicle_image is None:
            pass
        else:
            result, _ = process_license_plate_debug(vehicle_image, templates)
            print("\n-----------------------------------------")
            print(f"Segmented Plate Labels: {result}")
            print("-----------------------------------------")

