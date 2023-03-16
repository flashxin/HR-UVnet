# !/usr/bin/python3
# -*- coding:utf-8 -*-
# Author:WeiFeng Liu
# @Time: 2021/12/9 下午6:41

import cv2
import numpy as np
import os
  #  D:\研究生0年级\毕设\dataset\train
import shutil
import random
import numpy as np
import torchvision.datasets as dset
from shutil import copyfile


def tif_to_png(image_path,save_path):
    """
    :param image_path: *.tif image path
    :param save_path: *.png image path
    :return:
    """
    img = cv2.imread(image_path)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#为了转八位图
    # print(img)
    # print(img.dtype)
    array = np.array(img)
    max_val = np.amax(array)
    normalized = (array / max_val)
    array=array*normalized*255
    # im = pil_image.fromarray(normalized)
    filename = image_path.split('/')[-1].split('.')[0]
    # filename = image_path.split('/')[-1]
    # print(filename)
    save_path = save_path + '/' + filename + '.png'
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # print(save_path)




    cv2.imwrite(save_path,array)
def choic_val():
    film=get_file('../../CTset-lite/train')
    seq = random.sample(range(0, len(film)), len(film)//5)
    for cot in range(len(seq)):
        f = film[seq[cot]]
        # print(os.path.basename(os.path.abspath(os.path.dirname(f) + os.path.sep + ".")))
        labels=os.path.basename(os.path.abspath(os.path.dirname(f) + os.path.sep + "."))
        # number = os.path.basename(f)
        # for i in range(len(number)):
        if (labels=="1"):
            father_path = os.path.abspath(os.path.dirname(f) + os.path.sep + ".")  # 读取上一层路径
            grandfather_path = os.path.abspath(os.path.dirname(father_path) + os.path.sep + ".")
            old_grand_path=os.path.abspath(os.path.dirname(grandfather_path) + os.path.sep + ".")
            baseFname = os.path.basename(f)
            baseFname=os.path.join(old_grand_path, 'val\\1', baseFname)
            shutil.copy(f,baseFname)
            os.remove(f)
        else:
            father_path = os.path.abspath(os.path.dirname(f) + os.path.sep + ".")  # 读取上一层路径
            grandfather_path = os.path.abspath(os.path.dirname(father_path) + os.path.sep + ".")
            old_grand_path=os.path.abspath(os.path.dirname(grandfather_path) + os.path.sep + ".")
            baseFname = os.path.basename(f)
            baseFname=os.path.join(old_grand_path, 'val\\0', baseFname)
            shutil.copy(f,baseFname)
            os.remove(f)
def get_file(root_path, all_files=[]):
    '''
    递归函数，遍历该文档目录和子目录下的所有文件，获取其path
    '''
    files = os.listdir(root_path)
    for file in files:
        if not os.path.isdir(root_path + '/' + file):  # not a dir
            all_files.append(root_path + '/' + file)  # os.path.basename(file)
        else:  # is a dir
            get_file((root_path + '/' + file), all_files)
    return all_files
if __name__ == '__main__':
    # root_path = r'../../TrainValidation/'
    # save_path = r'../../CTset-lite/'
    # image_files = os.listdir(root_path)
    # # print(root_path + image_files[0])
    # for image_file in image_files:
    #     tif_to_png(root_path + image_file,save_path)
    #     # print(root_path + image_file)
    #     print("finish")
    choic_val()


