import numpy as np
from scipy import ndimage
from scipy.signal import convolve
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


#两种滤波器的实现
def RLFilter(N, d):
    filterRL = np.zeros((N,))
    for i in range(N):
        filterRL[i] = - 1.0 / np.power((i - N / 2) * np.pi * d, 2.0)
        if np.mod(i - N / 2, 2) == 0:
            filterRL[i] = 0
    filterRL[int(N/2)] = 1 / (4 * np.power(d, 2.0))
    return filterRL

def SLFilter(N, d):
    filterSL = np.zeros((N,))
    for i in range(N):
        #filterSL[i] = - 2 / (np.power(np.pi, 2.0) * np.power(d, 2.0) * (np.power((4 * (i - N / 2)), 2.0) - 1))
        filterSL[i] = - 2 / (np.pi**2.0 * d**2.0 * (4 * (i - N / 2)**2.0 - 1))
    return filterSL

def IRandonTransform(image, steps):
    print('正在逆变换...')
    channels = len(image[0])
    origin = np.zeros((steps, channels, channels))
    filter = RLFilter(channels, 1)
    #filter = SLFilter(channels, 1)
    for i in range(steps):
        projectionValue = image[:, i]
        projectionValueFiltered = convolve(filter, projectionValue, "same")
        projectionValueExpandDim = np.expand_dims(projectionValueFiltered, axis=0)
        projectionValueRepeat = projectionValueExpandDim.repeat(channels, axis=0)
        origin[i] = ndimage.rotate(projectionValueRepeat, i*180/steps, reshape=False).astype(np.float64)
    iradon = np.sum(origin, axis=0)
    return iradon


image = cv2.imread("12777.png", cv2.IMREAD_GRAYSCALE)
radon = DiscreteRadonTransform(image, len(image[0]))

iradon = IRandonTransform(radon, len(image[0]))


plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(2, 2, 2)
plt.imshow(radon, cmap='gray')
plt.title('Radon Image')

plt.subplot(2, 2, 3)
plt.imshow(iradon, cmap='gray')
plt.title('Filter Image')

plt.tight_layout()
plt.savefig("SLfilter.png", cmap='gray')
#plt.savefig("RLfilter.png", cmap='gray')

plt.show()
