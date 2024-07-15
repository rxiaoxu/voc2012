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

parser = argparse.ArgumentParser(description="choose the model")
parser.add_argument('-m', '--model', default='FCN', type=str, help="输入模型名字", choices=['Unet', 'FCN', 'Deeplab'])
parser.add_argument('-g', '--gpu', default=0, type=int, help="输入所需GPU")
args = parser.parse_args()

GPU_ID = args.gpu
INPUT_WIDTH = 320
INPUT_HEIGHT = 320
BATCH_SIZE = 16
NUM_CLASSES = 21
LEARNING_RATE = 1e-3
epoch = 120

if args.model == 'Unet':
    model = 'UNet'
    net = UNet(3, NUM_CLASSES)
elif args.model == "FCN":
    model = 'FCN8x'
    net = FCN8x(NUM_CLASSES)
elif args.model == "Deeplab":
    model = 'DeepLabV3'
    net = DeepLabV3(NUM_CLASSES)
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
if os.path.exists(result_path):
    os.remove(result_path)

train_csv_dir = 'train.csv'
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


# 训练验证
def train():
    best_score = 0.0
    start_time = time.time()  # 开始训练的时间

    for e in tqdm(range(epoch), desc='Epoch Progress'):
        net.train()
        epoch_start_time = time.time()  # 记录每个epoch的开始时间
        train_loss = 0.0
        label_true = torch.LongTensor()
        label_pred = torch.LongTensor()
        total_batches = len(train_dataloader)
        pbar = tqdm(total=total_batches, desc='Batch Progress', position=0)
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
            pbar.update(1) # 更新进度条

        # 关闭进度条
        pbar.close()
        train_loss /= len(train_data)
        acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(label_true.numpy(), label_pred.numpy(), NUM_CLASSES)

        print(
            f'\nepoch: {e + 1}, train_loss: {train_loss:.4f}, acc: {acc:.4f}, acc_cls: {acc_cls:.4f}, mean_iu: {mean_iu:.4f}, fwavacc: {fwavacc:.4f}')
        print(f'Time for this epoch: {time.time() - epoch_start_time:.2f} seconds')

        with open(result_path, 'a') as f:
            f.write(
                f'\n epoch: {e + 1}, train_loss: {train_loss:.4f}, acc: {acc:.4f}, acc_cls: {acc_cls:.4f}, mean_iu: {mean_iu:.4f}, fwavacc: {fwavacc:.4f}')

        net.eval()
        val_loss = 0.0
        val_label_true = torch.LongTensor()
        val_label_pred = torch.LongTensor()
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

            val_loss /= len(val_data)
            val_acc, val_acc_cls, val_mean_iu, val_fwavacc = label_accuracy_score(val_label_true.numpy(),
                                                                                  val_label_pred.numpy(), NUM_CLASSES)

        print(
            f'epoch: {e + 1}, val_loss: {val_loss:.4f}, acc: {val_acc:.4f}, acc_cls: {val_acc_cls:.4f}, mean_iu: {val_mean_iu:.4f}, fwavacc: {val_fwavacc:.4f}')

        with open(result_path, 'a') as f:
            f.write(
                f'\n epoch: {e + 1}, val_loss: {val_loss:.4f}, acc: {val_acc:.4f}, acc_cls: {val_acc_cls:.4f}, mean_iu: {val_mean_iu:.4f}, fwavacc: {val_fwavacc:.4f}')

        score = (val_acc_cls + val_mean_iu) / 2
        if score > best_score:
            best_score = score
            torch.save(net.state_dict(), model_path)

    total_time = time.time() - start_time
    print(f'Total training time: {total_time:.2f} seconds')


if __name__ == "__main__":
    train()
