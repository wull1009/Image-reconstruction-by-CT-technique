import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from numba import jit


def projection(image, theta):
    """
    计算投影值
    :param image: 原始图像
    :param theta: 射束旋转角度
    :return: 投影值矩阵
    projectionNum: 射束个数
    thetaNum: 角度个数
    randontansform: 拉东变换结果
    """
    print('正在计算投影...')

    projectionNum = len(image[0])
    thetaNum = len(theta)
    radontansform = np.zeros((projectionNum, thetaNum), dtype='float64')
    for i in range(len(theta)):
        # 进行离散拉登变换
        rotation = ndimage.rotate(image, -theta[i], reshape=False).astype('float64')
        radontansform[:, i] = sum(rotation)
    return radontansform

@jit(nopython=True)
def systeMartrix(theta, size, num, delta):
    """
    计算系统矩阵
    :param theta: 射束旋转角度
    :param size: 图片尺寸
    :param num: 射束条数
    :param delat: 网格边长
    :return: gridNum：穿过网格编号，gridLen：穿过网格长度
    """
    print('正在计算系统矩阵...')

    functionNum = len(theta) * num  # 方程个数，即为系统矩阵的行数
    # 射线最多穿过2*size个方格
    gridNum = np.zeros((functionNum, 2 * size))  # 系统矩阵：编号
    gridLen = np.zeros((functionNum, 2 * size))  # 系统矩阵：长度
    N = np.arange(-(size - 1) / 2, (size - 1) / 2 + 1)  # 射束
    for loop1 in range(len(theta)):
        th = theta[loop1]  # 射束角度
        for loop2 in range(size):
            u = np.zeros((2 * size))  # 编号
            v = np.zeros((2 * size))  # 长度

            # 垂直入射
            if th == 0:
                # 射束未穿过图像时
                if (N[loop2] >= size / 2 * delta) or (N[loop2] <= -size / 2 * delta):
                    continue
                # 入射网格编号
                kin = np.ceil(size / 2 + N[loop2] / delta)
                # 穿过网格编号
                kk = np.arange(kin, (kin + size * size), step=size)
                u[0:size] = kk
                v[0:size] = np.ones(size) * delta

            # 平行入射
            elif th == 90:
                if (N[loop2] >= size / 2 * delta) or (N[loop2] <= -size / 2 * delta):
                    continue
                # 出射网格编号
                kout = size * np.ceil(size / 2 - N[loop2] / delta)
                kk = np.arange(kout - size + 1, kout + 1)
                u[0:size] = kk
                v[0:size] = np.ones(size) * delta

            else:
                # phi为射束与x轴所夹锐角
                if th > 90:
                    phi = th - 90
                elif th < 90:
                    phi = 90 - th
                # 角度值换算为弧度制
                phi = phi * np.pi / 180
                # 截距
                b = N / np.cos(phi)
                # 斜率
                m = np.tan(phi)
                # 入射点纵坐标
                y1 = -(size / 2) * delta * m + b[loop2]
                # 出射点纵坐标
                y2 = (size / 2) * delta * m + b[loop2]

                # 射束未穿过图像
                if (y1 < -size / 2 * delta and y2 < -size / 2 * delta) or (
                        y1 > size / 2 * delta and y2 > size / 2 * delta):
                    continue

                # 穿过a、b边（左侧和上侧）
                if (y1 <= size / 2 * delta and y1 >= -size / 2 * delta and y2 > size / 2 * delta):
                    """
                    (xin,yin): 入射点坐标
                    (xout,yout): 出射点坐标
                    kin,kout: 入射格子标号，出射格子编号
                    d1: 入射格子左下角与入射射束距离
                    """
                    yin = y1
                    yout = size / 2 * delta
                    # xin = -size / 2 * delta
                    xout = (yout - b[loop2]) / m
                    kin = size * np.floor(size / 2 - yin / delta) + 1
                    kout = np.ceil(xout / delta) + size / 2
                    d1 = yin - np.floor(yin / delta) * delta

                # 穿过a、c边（左侧和右侧）
                elif (
                        y1 <= size / 2 * delta and y1 >= -size / 2 * delta and y2 >= -size / 2 * delta and y2 < size / 2 * delta):
                    # xin = -size / 2 * delta
                    # xout = size / 2 * delta
                    yin = y1
                    yout = y2
                    kin = size * np.floor(size / 2 - yin / delta) + 1
                    kout = size * np.floor(size / 2 - yout / delta) + size
                    d1 = yin - np.floor(yin / delta) * delta

                # 穿过d、b边（下侧和上侧）
                elif (y1 < - size / 2 * delta and y2 > size / 2 * delta):
                    yin = - size / 2 * delta
                    yout = size / 2 * delta
                    xin = (yin - b[loop2]) / m
                    xout = (yout - b[loop2]) / m
                    kin = size * (size - 1) + size / 2 + np.ceil(xin / delta)
                    kout = np.ceil(xout / delta) + size / 2
                    d1 = size / 2 * delta + np.floor(xin / delta) * delta * m + b[loop2]

                # 穿过d、c边（下侧和右侧）
                elif (y1 < - size / 2 * delta and y2 >= -size / 2 * delta and y2 < size / 2 * delta):
                    yin = -size / 2 * delta
                    yout = y2
                    xin = (yin - b[loop2]) / m
                    # xout = size / 2 * delta
                    kin = size * (size - 1) + size / 2 + np.ceil(xin / delta)
                    kout = size * np.floor(size / 2 - yout / delta) + size
                    d1 = size / 2 * delta + np.floor(xin / delta) * delta * m + b[loop2]

                else:
                    continue

                # 计算穿过的格子编号和长度
                """
                k: 射线穿过的格子编号
                c: 穿过格子的序号
                d2: 穿过的格子的右侧与该方格右下角顶点的距离
                """
                k = kin  # 入射的格子即为穿过的第一个格子
                c = 0  # c为穿过格子的序号
                d2 = d1 + m * delta  # 与方格右侧交点

                # 当方格数在1~n^2内迭代计算
                while k >= 1 and k <= np.power(size, 2):
                    """
                    根据射线与方格的左右两侧的交点关系，来确定穿过方格的六种情况。
                    在每种情况中，存入穿过的方格编号，穿过方格的射线长度。
                    若该方格是最后一个穿过的方格，则停止迭代；若不是最后一个方格，则计算下一个穿过的方格的编号、左右边与射线的交点。
                    """
                    if d1 >= 0 and d2 > delta:
                        u[c] = k  # 穿过方格的编号
                        v[c] = (delta - d1) * np.sqrt(np.power(m, 2) + 1) / m  # 穿过方格的射线长度
                        if k > size and k != kout:  # 若不是最后一个方格
                            k -= size  # 下一个方格编号
                            d1 -= delta  # 下一个方格左侧交点
                            d2 = delta * m + d1  # 下一个方格右侧交点
                        else:  # 若是最后一个方格则直接跳出循环
                            break

                    elif d1 >= 0 and d2 == delta:
                        u[c] = k
                        v[c] = delta * np.sqrt(np.power(m, 2) + 1)
                        if k > size and k != kout:
                            k -= size + 1
                            d1 = 0
                            d2 = d1 + m * delta
                        else:
                            break

                    elif d1 >= 0 and d2 < delta:
                        u[c] = k
                        v[c] = delta * np.sqrt(np.power(m, 2) + 1)
                        if k != kout:
                            k += 1
                            d1 = d2
                            d2 = d1 + m * delta
                        else:
                            break

                    elif d1 <= 0 and d2 >= 0 and d2 <= delta:
                        u[c] = k
                        v[c] = d2 * np.sqrt(np.power(m, 2) + 1) / m
                        if k != kout:
                            k += 1
                            d1 = d2
                            d2 = d1 + m * delta
                        else:
                            break

                    elif d1 <= 0 and d2 > delta:
                        u[c] = k
                        v[c] = delta * np.sqrt(np.power(m, 2) + 1) / m
                        if k > size and k != kout:
                            k -= size
                            d1 -= delta
                            d2 = d1 + m * delta
                        else:
                            break

                    elif d1 <= 0 and d2 == delta:
                        u[c] = k
                        v[c] = d2 * np.sqrt(np.power(m, 2) + 1) / m
                        if k > size and k != kout:
                            k -= size + 1
                            d1 = 0
                            d2 = delta * m
                        else:
                            break

                    else:
                        print(d1, d2, "数据错误！")

                    c += 1

                # 当射线斜率为负数时，利用对称性进行计算
                if th < 90:
                    u_temp = np.zeros(2 * size)
                    # 排除掉未穿过图像的射束
                    if u.any() == 0:
                        continue
                    # 射束穿过的格子的编号
                    indexMTZero = np.where(u > 0)

                    # 利用对称性求得穿过网格编号
                    for loop in range(len(u[indexMTZero])):
                        """计算穿过的方格编号，利用方格编号与边长的取余的关系，得到对称的射线穿过的方格编号。"""
                        r = np.mod(u[loop], size)
                        if r == 0:
                            u_temp[loop] = u[loop] - size + 1
                        else:
                            u_temp[loop] = u[loop] - 2 * r + size + 1
                    u = u_temp

            # 方格编号
            gridNum[loop1 * num + loop2, :] = u
            # 穿过方格的射线长度
            gridLen[loop1 * num + loop2, :] = v

    return gridNum, gridLen

@jit(nopython=True)
def iteration(theta, size, gridNum, gridLen, F, ite_num):
    """
    按照公式迭代重建
    :param theta: 旋转角度
    :param size: 图像边长
    :param gridNum: 射线穿过方格编号
    :param gridLen: 射线穿过方格长度
    :param F: 重建后图像
    :return: 重建后图像
    """
    print('正在进行迭代...')

    c = 0  # 迭代计数
    while (c < ite_num):
        print('第' + str(c+1) + '次迭代')
        for loop1 in range(len(theta)):  # 在角度theta下
            for loop2 in range(size):  # 第loop2条射线
                u = gridNum[loop1 * size + loop2, :]
                v = gridLen[loop1 * size + loop2, :]
                if u.any() == 0:  # 若射线未穿过图像，则直接计算下一条射线
                    continue
                # 本条射线对应的行向量
                w = np.zeros(sizeSquare, dtype=np.float64)
                # 本条射线穿过的网格编号
                uLargerThanZero = np.where(u > 0)
                # 本条射线穿过的网格长度
                w[u[uLargerThanZero].astype(np.int64) - 1] = v[uLargerThanZero]
                # 计算估计投影值
                PP = w.dot(F)
                # 计算实际投影与估计投影的误差
                error = projectionValue[loop2, loop1] - PP
                # 求修正值
                C = error / sum(np.power(w, 2)) * w.conj()
                # 进行修正
                F = F + lam * C
        F[np.where(F < 0)] = 0
        c = c + 1

    F = F.reshape(size, size).conj()

    return F


if __name__ == '__main__':
    image = cv2.imread("1277(256).png", cv2.IMREAD_GRAYSCALE)
    """
    theta: 射束旋转角度
    num: 射束条数
    size: 图片尺寸
    delta: 网格边长
    ite_num: 迭代次数
    lam: 松弛因子
    """
    theta = np.linspace(0, 180, 60, dtype=np.float64)
    theta1 = np.linspace(0, 180, 256, dtype=np.float64)
    num = np.int64(256)
    size = np.int64(256)
    delta = np.int64(1)
    lam = np.float64(.25)
    sizeSquare = size * size

    # 计算投影值
    projectionValue = projection(image, theta)
    projectionForPh = projection(image, theta1)
    # 计算系统矩阵
    gridNum, gridLen = systeMartrix(theta, size, num, delta)

    # 重建后图像矩阵
    F = np.zeros((size * size,))

    ite_num =10
    # 迭代法重建
    reconImage = iteration(theta, size, gridNum, gridLen, F, ite_num)
    print('共迭代%d次。' % ite_num)
    # plt.subplot(3, 3, ite_num)
    # plt.imshow(reconImage, cmap='gray')
    # plt.title('loop' + str(ite_num))
    #
    #
    # plt.tight_layout()
    # plt.savefig('./loop.png', dpi=300, bbox_inches='tight')
    # plt.show()

    # 绘制原始图像、重建图像、误差图像
    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')

    plt.subplot(2, 2, 2)
    plt.imshow(projectionForPh, cmap='gray')
    plt.title('Radon Image')

    plt.subplot(2, 2, 3)
    plt.imshow(reconImage, cmap='gray')
    plt.title('Reconstruction Image')

    plt.subplot(2, 2, 4)
    plt.imshow(reconImage - image, cmap='gray')
    plt.title('Error Image')

    print(sum(sum(reconImage-image)))

    plt.tight_layout()
    plt.savefig("ART.png", cmap='gray')

    plt.show()

