import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
def nearest_neighbor(image, new_height, new_width):
    height, width, channels = image.shape
    resized = np.zeros((new_height, new_width, channels))
    height_ratio = height / new_height
    width_ratio = width / new_width
    for i in range(new_height):
        for j in range(new_width):
            src_i = min(int(i * height_ratio), height - 1)
            src_j = min(int(j * width_ratio), width - 1)
            resized[i, j] = image[src_i, src_j]
    
    return resized

def bilinear_interpolation(image, new_height, new_width):
    height, width, channels = image.shape
    resized = np.zeros((new_height, new_width, channels))
    height_ratio = (height - 1) / (new_height - 1) if new_height > 1 else 0
    width_ratio = (width - 1) / (new_width - 1) if new_width > 1 else 0
    for i in range(new_height):
        for j in range(new_width):
            src_i = i * height_ratio
            src_j = j * width_ratio
            i0 = int(np.floor(src_i))
            i1 = min(i0 + 1, height - 1)
            j0 = int(np.floor(src_j))
            j1 = min(j0 + 1, width - 1)
            di = src_i - i0
            dj = src_j - j0
            for c in range(channels):
                top = image[i0, j0, c] * (1 - dj) + image[i0, j1, c] * dj
                bottom = image[i1, j0, c] * (1 - dj) + image[i1, j1, c] * dj
                resized[i, j, c] = top * (1 - di) + bottom * di
    
    return resized

def cubic_kernel(x):
    abs_x = np.abs(x)
    if abs_x <= 1:
        return 1.5 * abs_x**3 - 2.5 * abs_x**2 + 1
    elif abs_x < 2:
        return -0.5 * abs_x**3 + 2.5 * abs_x**2 - 4 * abs_x + 2
    else:
        return 0

def bicubic_interpolation(image, new_height, new_width):
    height, width, channels = image.shape
    resized = np.zeros((new_height, new_width, channels))
    height_ratio = (height - 1) / (new_height - 1) if new_height > 1 else 0
    width_ratio = (width - 1) / (new_width - 1) if new_width > 1 else 0
    
    for i in range(new_height):
        for j in range(new_width):
            src_i = i * height_ratio
            src_j = j * width_ratio
            i0 = int(np.floor(src_i))
            j0 = int(np.floor(src_j))
            di = src_i - i0
            dj = src_j - j0
            for c in range(channels):
                value = 0
                for m in range(-1, 3):
                    for n in range(-1, 3):
                        ii = max(0, min(height - 1, i0 + m))
                        jj = max(0, min(width - 1, j0 + n))
                        weight = cubic_kernel(m - di) * cubic_kernel(n - dj)
                        value += image[ii, jj, c] * weight
                resized[i, j, c] = np.clip(value, 0, 1 if image.dtype == np.float32 or image.dtype == np.float64 else 255)
    
    return resized

def load_and_resize_image(image_path, new_height, new_width):
    image = imread(image_path)
    if image.dtype != np.float32 and image.dtype != np.float64:
        image = image.astype(np.float32) / 255.0
    nn_resized = nearest_neighbor(image, new_height, new_width)
    bilinear_resized = bilinear_interpolation(image, new_height, new_width)
    bicubic_resized = bicubic_interpolation(image, new_height, new_width)
    
    return image, nn_resized, bilinear_resized, bicubic_resized

def display_results(original, nn, bilinear, bicubic):
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(original)
    plt.title("Original Image")
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.imshow(nn)
    plt.title("Nearest Neighbor Interpolation")
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.imshow(bilinear)
    plt.title("Bilinear Interpolation")
    plt.axis('off')
    plt.subplot(2, 2, 4)
    plt.imshow(bicubic)
    plt.title("Bicubic Interpolation")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    image_path = "d:\\Academic\\Semester\\Digital Image Lab\\monalisa.jpg"
    new_height, new_width = 300, 400    
    try:
        original, nn, bilinear, bicubic = load_and_resize_image(image_path, new_height, new_width)
        display_results(original, nn, bilinear, bicubic)        
    except FileNotFoundError:
        print(f"Error: The file '{image_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")