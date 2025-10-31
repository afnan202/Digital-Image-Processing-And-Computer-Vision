from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
image = Image.open("D:\\Academic\\Semester\\Digital Image Lab\\input.png").convert('L')
img_array = np.array(image)
output1 = 255 - img_array
min_val = np.min(image)
max_val = np.max(image)
stretched = ((img_array - min_val) * 265 / (max_val - min_val)).astype(np.uint8)
output2 = np.array(stretched, dtype=np.uint8)
gamma1 = 0.5
output3 = 255 * (img_array / 255) ** gamma1
output3 = np.array(output3, dtype=np.uint8)
gamma2 = 2.5
output4 = 255 * (img_array / 255) ** gamma2
output4 = np.array(output4, dtype=np.uint8)
titles = ['(a) Input image', '(b) Output image 1', '(c) Output image 2', 
          '(d) Output image 3', '(e) Output image 4']
images = [img_array, output1, output2, output3, output4]

plt.figure(figsize=(10, 8))
for i in range(5):
    plt.subplot(3, 2, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
