from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import imageio
import cv2


def DiscreteRadonTransform(image, steps):
    channels = len(image[0])
    res = np.zeros((channels, channels), dtype='float64')
    for s in range(steps):
        rotation = ndimage.rotate(image, -s*180/steps, reshape=False).astype('float64')
        #print(sum(rotation).shape)
        res[:,s] = sum(rotation)
    return res

#读取原始图片
#image = cv2.imread("whiteLineModify.png", cv2.IMREAD_GRAYSCALE)
#image=imageio.imread('shepplogan.jpg').astype(np.float64)
#image = cv2.imread("whitePoint.png", cv2.IMREAD_GRAYSCALE)
image = cv2.imread("12777.png", cv2.IMREAD_GRAYSCALE)
radon = DiscreteRadonTransform(image, len(image[0]))
print(radon.shape)

#绘制原始图像和对应的sinogram图
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(radon, cmap='gray')
plt.savefig('./radoncompare.png', dpi=138.7, bbox_inches='tight', pad_inches=0.0)
plt.show()


plt.imshow(radon, cmap='gray')
plt.axis('off')
plt.xticks([])  #去掉横坐标值
plt.yticks([])  #去掉纵坐标值
plt.savefig('./radon.png', dpi=138.7, bbox_inches='tight', pad_inches=0.0)
