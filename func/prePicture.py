# 保持比例，指定输出图片的大小，宽和高 img是文件地址 数据要求是numpy
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import copy as cp
import random
import collections
# 封装resize函数 单通道
def resize_img_keep_ratio(img_name,target_size):
    img = cv2.imread(img_name) # 读取图片
    old_size= img.shape[0:2] # 原始图像大小
    ratio = min(float(target_size[i])/(old_size[i]) for i in range(len(old_size))) # 计算原始图像宽高与目标图像大小的比例，并取其中的较小值
    new_size = tuple([int(i*ratio) for i in old_size]) # 根据上边求得的比例计算在保持比例前提下得到的图像大小
    img = cv2.resize(img,(new_size[1], new_size[0])) # 根据上边的大小进行放缩
    pad_w = target_size[1] - new_size[1] # 计算需要填充的像素数目（图像的宽这一维度上）
    pad_h = target_size[0] - new_size[0] # 计算需要填充的像素数目（图像的高这一维度上）
    top,bottom = pad_h//2, pad_h-(pad_h//2)
    left,right = pad_w//2, pad_w -(pad_w//2)
    img_new = cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,None,(0,0,0))
    img_new = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)
    # print(img_new.shape)
    return img_new

#


#图像分块 输出为 m*n个图像 输入
def divide_method2(img, m, n):  # 分割成m行n列
    m=m+1
    n=n+1
    h, w = img.shape[0], img.shape[1]
    grid_h = int(h * 1.0 / (m - 1) + 0.5)  # 每个网格的高
    grid_w = int(w * 1.0 / (n - 1) + 0.5)  # 每个网格的宽
    # 满足整除关系时的高、宽
    h = grid_h * (m - 1)
    w = grid_w * (n - 1)
    # 图像缩放

    # plt.imshow(img_re)
    gx, gy = np.meshgrid(np.linspace(0, w, n), np.linspace(0, h, m))
    gx = gx.astype(np.int)
    gy = gy.astype(np.int)
    divide_image = np.zeros([m - 1, n - 1, grid_h, grid_w])  # 这是一个五维的张量，前面两维表示分块后图像的位置（第m行，第n列），后面三维表示每个分块后的图像信息
    # print(divide_image.shape,img.shape)
    for i in range(m - 1):
        for j in range(n - 1):
            divide_image[i, j, ...] = img[
                                      gy[i][j]:gy[i + 1][j + 1], gx[i][j]:gx[i + 1][j + 1]]
    return divide_image
# 显示图片和上一个函数配套使用
def display_blocks(divide_image):
    m, n = divide_image.shape[0], divide_image.shape[1]
    for i in range(m):
        for j in range(n):
            plt.subplot(m, n, i * n + j + 1)
            plt.imshow(divide_image[i, j, :])
            plt.axis('off')
    plt.show()

#
# img = '1.png' # 待处理的图片地址, 替换成你的地址就好
# target_size=[224, 224] # 目标图像大小
# resized_img = resize_img_keep_ratio(img, target_size)
#
#
# divide_image2 = divide_method2(resized_img, 4, 4)  # 该函数中m+1和n+1表示网格点个数，m和n分别表示分块的块数
# fig3 = plt.figure('分块后的子图像:图像缩放法')
# display_blocks(divide_image2)

if __name__ == "__main__":
    img = '1.png' # 待处理的图片地址, 替换成你的地址就好
    target_size=[256, 256] # 目标图像大小
    # plt.imshow(img, cmap='gray')
    # resized_img = resize_img_keep_ratio(img, target_size)
    img = resize_img_keep_ratio(img, [256, 256])
    img = np.float32(img)
    Dctimg = cv2.dct(img)
    print(np.max(Dctimg),np.min(Dctimg))
    Dctimg=(Dctimg-np.min(Dctimg))/(np.max(Dctimg)-np.min(Dctimg))
    print(np.max(Dctimg),np.min(Dctimg))
    print(Dctimg*255)
    plt.imshow(Dctimg*255,cmap='gray')
    plt.show()
def Prewitt(image):
    # image = cv2.imread(imgpath)
    # lena = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 灰度转化处理
    # grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # kernel
    kernelX = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    kernelY = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
    x = cv2.filter2D(image, cv2.CV_16S, kernelX)
    y = cv2.filter2D(image, cv2.CV_16S, kernelY)

    # 转uint8
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)

    # 加权和
    Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return Prewitt