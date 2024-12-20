import numpy as np
import matplotlib.pyplot as plt

def get_random_crop(image, crop_height, crop_width):

    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    crop = image[y: y + crop_height, x: x + crop_width]

    return crop



example_image = np.random.randint(0, 256, (1024, 1024, 3))
random_crop = get_random_crop(example_image, 100, 100)

plt.imshow(example_image)
plt.show()
plt.imshow(random_crop)
plt.show()