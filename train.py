# -*- encoding: utf-8 -*-
# here put the import lib
import pandas as pd
import numpy as np
from utils.DataLoade import CustomDataset
from torch.utils.data import DataLoader
from model.FCN import FCN32s, FCN8x
from model.Unet import UNet
from model.DeepLab import DeepLabV3
import torch
import os
from torch import nn, optim
from torch.nn import functional as F
from utils.eval_tool import label_accuracy_score
from utils.data_txt import image2csv
import argparse
from tqdm import tqdm
import time

from torch.utils.tensorboard import SummaryWriter

#   引用u3+模型
from u3plus.UNet_3Plus import UNet_3Plus
from u3plus.UNet_3Plus import UNet_3Plus_DeepSup

parser = argparse.ArgumentParser(description="choose the model")
parser.add_argument('-m','--model', default='Unet' ,type= str, help= "输入模型名字",
                    choices = ['Unet','FCN','Deeplab','Unet3+','Unet3+_Sup'])
parser.add_argument('-g', '--gpu', default=0, type=int, help="输入所需GPU")
args = parser.parse_args()

GPU_ID = args.gpu
INPUT_WIDTH = 320
INPUT_HEIGHT = 320
BATCH_SIZE = 4

NUM_CLASSES = 21
LEARNING_RATE = 1e-3
epoch = 5

if args.model == 'Unet':
    model = 'UNet'
    net = UNet(3, NUM_CLASSES)
elif args.model == "FCN":
    model = 'FCN8x'
    net = FCN8x(NUM_CLASSES)
elif args.model == "Deeplab":
    model = 'DeepLabV3'
    net = DeepLabV3(NUM_CLASSES)
elif args.model == 'Unet3+':
    model = 'Unet3+'
    net = UNet_3Plus()
elif args.model == 'Unet3+_Sup':
    model = 'Unet3+_Sup'
    net = UNet_3Plus_DeepSup()


# -------------------- 生成csv ------------------
# DATA_ROOT =  './data/'
# image = os.path.join(DATA_ROOT,'JPEGImages')
# label = os.path.join(DATA_ROOT,'SegmentationClass')
# slice_data = [0.7,0.1,0.2] #  训练 验证 测试所占百分比
# tocsv = image2csv(DATA_ROOT,image,label,slice_data,INPUT_WIDTH,INPUT_HEIGHT)
# tocsv.generate_csv()
# -------------------------------------------
model_path = './model_result/best_model_{}.mdl'.format(model)
result_path = './result_{}.txt'.format(model)
tensor_path = './tensorboard'

if not os.path.exists(tensor_path):
    os.mkdir(tensor_path)

if os.path.exists(result_path):
    os.remove(result_path)

train_csv_dir = 'val.csv'

val_csv_dir = 'val.csv'
train_data = CustomDataset(train_csv_dir, INPUT_WIDTH, INPUT_HEIGHT)
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

val_data = CustomDataset(val_csv_dir, INPUT_WIDTH, INPUT_HEIGHT)
val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# net = FCN8x(NUM_CLASSES)
# net = UNet(3,NUM_CLASSES)
# net = model(NUM_CLASSES)
use_gpu = torch.cuda.is_available()

# 构建网络
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
if use_gpu:
    torch.cuda.set_device(GPU_ID)
    net.cuda()
    criterion = criterion.cuda()

writer = SummaryWriter(log_dir='./tensorboard/{}'.format(model))
sample_input = torch.randn(BATCH_SIZE, 3, INPUT_WIDTH, INPUT_HEIGHT)
if use_gpu:
    sample_input = sample_input.cuda()
writer.add_graph(net, sample_input)


# 训练验证
def train():
    best_score = 0.0
    start_time = time.time()  # 开始训练的时间

    #net.loadIFExist(model_path)

    for e in range(epoch):
        net.train()
        epoch_start_time = time.time()  # 记录每个epoch的开始时间
        train_loss = 0.0
        label_true = torch.LongTensor()
        label_pred = torch.LongTensor()
        #   train的进度条
        with tqdm(total=len(train_dataloader), desc=f'{e+1}/{epoch} epoch Train_Progress') as pb_train:
            for i, (batchdata, batchlabel) in enumerate(train_dataloader):
                if use_gpu:
                    batchdata, batchlabel = batchdata.cuda(), batchlabel.cuda()

                output = net(batchdata)
                output = F.log_softmax(output, dim=1)
                loss = criterion(output, batchlabel)

                pred = output.argmax(dim=1).squeeze().data.cpu()
                real = batchlabel.data.cpu()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.cpu().item() * batchlabel.size(0)
                label_true = torch.cat((label_true, real), dim=0)
                label_pred = torch.cat((label_pred, pred), dim=0)
                pb_train.update(1)


        train_loss /= len(train_data)
        acc, mean_acc_cls, mean_iu, fwavacc, _, _, _, mean_dice= label_accuracy_score(label_true.numpy(), label_pred.numpy(), NUM_CLASSES)

        print(
            f'epoch: {e + 1}, train_loss: {train_loss:.4f}, acc: {acc:.4f}, acc_cls: {mean_acc_cls:.4f}, mean_iu: {mean_iu:.4f}, fwavacc: {fwavacc:.4f},mean_dice:{mean_dice:.4f}')
        print(f'Time for this epoch: {time.time() - epoch_start_time:.2f} seconds')

        with open(result_path, 'a') as f:
            f.write(
                f'\n epoch: {e + 1}, train_loss: {train_loss:.4f}, acc: {acc:.4f}, acc_cls: {mean_acc_cls:.4f}, mean_iu: {mean_iu:.4f}, fwavacc: {fwavacc:.4f},mean_dice:{mean_dice:.4f}')

        writer.add_scalars('Train_Validation_Loss',{'train_loss':train_loss}, e + 1)
        writer.add_scalars('Train_Validation_Accuracy', {'acc':acc}, e + 1)
        writer.add_scalars('Train_Validation_Mean_accuracy_cls', {'mean_acc_cls':mean_acc_cls}, e + 1)
        writer.add_scalars('Train_Validation_Mean_IU', {'mean_iu':mean_iu}, e + 1)
        writer.add_scalars('Train_Validation_Fwavacc', {'fwavacc':fwavacc}, e + 1)
        writer.add_scalars('Train_Validation_Mean_dice', {'mean_dice':mean_dice}, e + 1)


        net.eval()
        val_loss = 0.0
        val_label_true = torch.LongTensor()
        val_label_pred = torch.LongTensor()
        with tqdm(total=len(val_dataloader), desc=f'{e+1}/{epoch} epoch Val_Progress') as pb_val:
            with torch.no_grad():
                for i, (batchdata, batchlabel) in enumerate(val_dataloader):
                    if use_gpu:
                        batchdata, batchlabel = batchdata.cuda(), batchlabel.cuda()

                    output = net(batchdata)
                    output = F.log_softmax(output, dim=1)
                    loss = criterion(output, batchlabel)

                    pred = output.argmax(dim=1).squeeze().data.cpu()
                    real = batchlabel.data.cpu()

                    val_loss += loss.cpu().item() * batchlabel.size(0)
                    val_label_true = torch.cat((val_label_true, real), dim=0)
                    val_label_pred = torch.cat((val_label_pred, pred), dim=0)

                    pb_val.update(1)

            val_loss /= len(val_data)
            val_acc, val_mean_acc_cls, val_mean_iu, val_fwavacc, _, _, _, val_mean_dice= label_accuracy_score(val_label_true.numpy(),
                                                                                  val_label_pred.numpy(), NUM_CLASSES)

        print(
            f'epoch: {e + 1}, val_loss: {val_loss:.4f}, acc: {val_acc:.4f}, acc_cls: {val_mean_acc_cls:.4f}, mean_iu: {val_mean_iu:.4f}, fwavacc: {val_fwavacc:.4f},mean_dice:{val_mean_dice:.4f}')

        with open(result_path, 'a') as f:
            f.write(
                f'\n epoch: {e + 1}, val_loss: {val_loss:.4f}, acc: {val_acc:.4f}, acc_cls: {val_mean_acc_cls:.4f}, mean_iu: {val_mean_iu:.4f}, fwavacc: {val_fwavacc:.4f},mean_dice:{val_mean_dice:.4f}')

        writer.add_scalars('Train_Validation_Loss', {'val_loss':val_loss}, e + 1)
        writer.add_scalars('Train_Validation_Accuracy', {'val_acc':val_acc}, e + 1)
        writer.add_scalars('Train_Validation_Mean_accuracy_cls',{'val_mean_acc_cls':val_mean_acc_cls}, e + 1)
        writer.add_scalars('Train_Validation_Mean_IU', {'val_mean_iu':val_mean_iu}, e + 1)
        writer.add_scalars('Train_Validation_Fwavacc', {'val_fwavacc':val_fwavacc}, e + 1)
        writer.add_scalars('Train_Validation_Mean_dice', {'val_mean_dice':val_mean_dice}, e + 1)

        score = (val_mean_acc_cls + val_mean_iu) / 2
        if score > best_score:
            best_score = score
            torch.save(net.state_dict(), model_path)

        torch.cuda.empty_cache()
    total_time = time.time() - start_time
    print(f'Total training time: {total_time:.2f} seconds')


if __name__ == "__main__":
    train()
