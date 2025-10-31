import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
def apply_affine(img, matrix):
    affine_matrix = matrix[:2, :].flatten()
    transformed = img.transform(
        img.size,
        Image.AFFINE,
        affine_matrix,
        resample=Image.BILINEAR
    )
    return transformed
def get_affine_matrix(operation, **kwargs):
    if operation == "scale":
        sx = kwargs.get("sx", 1)
        sy = kwargs.get("sy", 1)
        return np.array([
            [sx, 0, 0],
            [0, sy, 0],
            [0, 0, 1]
        ])
    elif operation == "rotate":
        angle = kwargs.get("angle", 0)
        rad = np.radians(angle)
        return np.array([
            [np.cos(rad), -np.sin(rad), 0],
            [np.sin(rad),  np.cos(rad), 0],
            [0, 0, 1]
        ])
    elif operation == "translate":
        tx = kwargs.get("tx", 0)
        ty = kwargs.get("ty", 0)
        return np.array([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ])
    elif operation == "shear":
        shx = kwargs.get("shx", 0)
        shy = kwargs.get("shy", 0)
        return np.array([
            [1, shx, 0],
            [shy, 1, 0],
            [0, 0, 1]
        ])
    else:
        return np.identity(3)
if __name__ == "__main__":
    img_path = "d:\\Academic\\Semester\\Digital Image Lab\\monalisa.jpg"
    img = Image.open(img_path)
    scale_matrix = get_affine_matrix("scale", sx=1.5, sy=1.2)
    rotate_matrix = get_affine_matrix("rotate", angle=30)
    translate_matrix = get_affine_matrix("translate", tx=40, ty=-30)
    shear_matrix = get_affine_matrix("shear", shx=0.3, shy=0.2)
    scaled_img = apply_affine(img, scale_matrix)
    rotated_img = apply_affine(img, rotate_matrix)
    translated_img = apply_affine(img, translate_matrix)
    sheared_img = apply_affine(img, shear_matrix)
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs[0, 0].imshow(scaled_img)
    axs[0, 0].set_title("Scaled Image")
    axs[0, 0].axis('off')
    axs[0, 1].imshow(rotated_img)
    axs[0, 1].set_title("Rotated Image 45 degree")
    axs[0, 1].axis('off')
    axs[1, 0].imshow(translated_img)
    axs[1, 0].set_title("Translated Image Using Matrix")
    axs[1, 0].axis('off')
    axs[1, 1].imshow(sheared_img)
    axs[1, 1].set_title("Sheared Image")
    axs[1, 1].axis('off')
    plt.tight_layout()
    plt.show()

