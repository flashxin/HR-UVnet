import sys
sys.path.append("..")
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
import  os
import cv2
import func.LoadDataset as D
from net import HRnet as mymodel
import func.BalcnceDataParallel as Balance
if __name__=="__main__":
    os.environ['TORCH_HOME'] = '../'
    os.environ["CUDA_VISIBLE_DEVICES"] = '1,0'
    objname="HR-UVnet"
    ifcuda=True
    ifwandb=True
    EPOCH = 100
    BATCH_SIZE =12 
    LR = 0.01
    max_acc = 0.75
    testFreq=2
    net=mymodel.HRnet(inchannel=2,kinds=4)
    net.train()
    if ifcuda:
        net.cuda()
    if ifwandb:
        wandb.init(project=objname)
        # # 2. Save model inputs and hyperparameters
        wandb.config = {
            "learning_rate": LR,
            "epochs": EPOCH,
            "batch_size": BATCH_SIZE
        }
    train_dataset=D.MyDataset(train=True)
    test_dataset=D.MyDataset(train=False)
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=2,drop_last=True)
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=2,drop_last=True)
    print(len(train_dataset.files))
    print(len(test_dataset.files))
    optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=0.0005)
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.99)
    for epoch in range(EPOCH):
        for step, (img,Dctblock,label) in enumerate(train_loader):
            net.train()
            img = img.type(torch.FloatTensor)
            Dctblock = Dctblock.type(torch.FloatTensor)
            label = label.type(torch.LongTensor)
            if ifcuda:
                img = img.cuda()
                Dctblock=Dctblock.cuda()
                label=label.cuda()
            out1,out2,out3,vitout,output = net(img,Dctblock)  # cnn output
            loss1 = loss_func(output,label)  # cross entropy loss
            loss2 = loss_func(out1,label)  # cross entropy loss
            loss3 = loss_func(out2,label)  # cross entropy loss
            loss4 = loss_func(out3,label)  # cross entropy loss
            loss5 = loss_func(vitout,label)  # cross entropy loss
            loss = loss1+0.25*(loss2+loss3+loss4+loss5)
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            scheduler.step()
            if step % 10 == 0:
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy())
                if ifwandb:
                    wandb.log({"loss": loss})
        if epoch % testFreq == 0:
            all_num = 0
            accuracy = []
            with torch.no_grad():
                net.eval()
                for step, (img,Dctblock,label) in enumerate(test_loader):
                    img = img.type(torch.FloatTensor)
                    Dctblock = Dctblock.type(torch.FloatTensor)
                    label = label.type(torch.LongTensor)
                    if ifcuda:
                        img = img.cuda()
                        Dctblock = Dctblock.cuda()
                        label = label.cuda()
                        #DCT=DCT.cuda()
                    ou1,out2,out3,vitout,test_output = net(img,Dctblock)
                    pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
                    accuracy.append(float((pred_y == label.data.cpu().numpy()).astype(int).sum()) / float(label.size(0)))
                    all_num += 1
                accuracy = torch.tensor(accuracy)
                test_accuracy = float(torch.mean(accuracy))
                print('| test accuracy: %.4f' % test_accuracy)
                if ifwandb:
                    wandb.log({'Epoch': epoch, "Learn-rate": optimizer.state_dict()['param_groups'][0]['lr'],
                               "acc": test_accuracy, "acc_max": float(torch.max(accuracy)),
                               "acc_min": float(torch.min(accuracy)), })
                if (torch.mean(accuracy) > max_acc):
                    max_acc = torch.mean(accuracy)
                    torch.save(net.state_dict(),objname + '_%.4f.pt' % max_acc)
