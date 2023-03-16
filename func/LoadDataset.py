import torchvision.models
import numpy
import torch.utils.data as Data #批处理模块
import torch
from torch.autograd.gradcheck import gradcheck
import torchvision.datasets as dset
import copy
from torchvision import transforms
from torch import nn
import wandb
import tqdm
import math
import numpy as np
from sklearn import linear_model
import scipy.misc
from matplotlib import pyplot as plt
from PIL import Image
import  os
import cv2
import func.prePicture as func
data_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize([224,224]),
    transforms.RandomRotation(45, expand=False),
])
class MyDataset(torch.utils.data.Dataset):
    def __init__(self,train=True):
        super(MyDataset, self).__init__()
        if train:
            self.root='../../dataset/train'
        else:
            self.root='../../dataset/val'
        self.files=[]
        self.get_file(self.root,self.files)
        self.mode=train
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        image_name = self.files[idx]
        img=func.resize_img_keep_ratio(image_name,[256,256])
        Pimg=func.Prewitt(img)#256*256
        Pimg = torch.tensor(Pimg)
        Pimg = torch.unsqueeze(Pimg, 0)

        imgblock = func.divide_method2(img,4,4)#每个小块64*64 64个小块
        img=np.float32(img)
        Dctimg = cv2.dct(img)
        Dctimg=torch.Tensor(Dctimg)
        Dctimg = torch.unsqueeze(Dctimg, 0)
        Dctblock=np.zeros([16,64,64])
        count=0
        for i in range(4):
            for j in range(4):
                # Dctblock[count,...]=cv2.dct(imgblock[i,j,...])
                Dctblock[count, ...] = imgblock[i, j, ...]
        # 猜测由于下句BGR-->RGB转换导致预测混乱
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        father_path = os.path.abspath(os.path.dirname(image_name) + os.path.sep + ".")
        label = int(os.path.basename(father_path))
        # img=torch.cat([img,])
        if self.mode==True:
            img = data_transform(img)
        img = torch.tensor(img)
        if self.mode==False:
            img=torch.unsqueeze(img, 0)
        # print(img.shape)
        img=torch.cat([img,Pimg],dim=0)
        # print(img.shape)
        return img,Dctimg,Dctblock,label

    def get_file(self,root_path, all_files=[]):
        '''
        递归函数，遍历该文档目录和子目录下的所有文件，获取其path
        '''
        files = os.listdir(root_path)
        for file in files:
            if not os.path.isdir(root_path + '/' + file):  # not a dir
                all_files.append(root_path + '/' + file)  # os.path.basename(file)
            else:  # is a dir
                self.get_file((root_path + '/' + file), all_files)
        return all_files