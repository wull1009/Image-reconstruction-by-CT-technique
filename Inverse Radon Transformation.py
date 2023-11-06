import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2

#拉东变换
def DiscreteRadonTransform(image, steps):
    print('正在拉东变换...')
    channels = len(image[0])
    res = np.zeros((channels, channels), dtype='float64')
    for s in range(steps):
        rotation = ndimage.rotate(image, -s*180/steps, reshape=False).astype('float64')
        res[:,s] = sum(rotation)
    return res


def IRandonTransform(image, steps):
    print('正在逆变换...')
    channels = len(image[0])
    origin = np.zeros((steps, channels, channels))
    for i in range(steps):
        projectionValue = image[:, i]
        projectionValueExpandDim = np.expand_dims(projectionValue, axis=0)
        projectionValueRepeat = projectionValueExpandDim.repeat(channels, axis=0)
        origin[i] = ndimage.rotate(projectionValueRepeat, i*180/steps, reshape=False).astype(np.float64)
    iradon = np.sum(origin, axis=0)
    return iradon

image = cv2.imread("12777.png", cv2.IMREAD_GRAYSCALE)
radon = DiscreteRadonTransform(image, len(image[0]))

iradon = IRandonTransform(radon, len(image[0]))


plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(iradon, cmap='gray')
plt.show()

plt.imshow(iradon, cmap='gray')
plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.savefig('./iradon.png', dpi=138.7, bbox_inches='tight', pad_inches=0.0)

