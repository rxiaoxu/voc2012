import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog
import pandas as pd
import numpy as np
from model.FCN import FCN32s, FCN8x
from model.Unet import UNet
import torch
import torchvision.transforms as transforms
import os
from model.DeepLab import DeepLabV3
# from torch import nn,optim
# from torch.nn import functional as F
from utils.eval_tool import label_accuracy_score

# model = 'UNet'
model = 'FCN8x'
# model = 'DeepLabV3'
GPU_ID = 0
INPUT_WIDTH = 320
INPUT_HEIGHT = 320
BATCH_SIZE = 32
NUM_CLASSES = 21
LEARNING_RATE = 1e-3
model_path = './model_result/best_model_{}.mdl'.format(model)
torch.cuda.set_device(GPU_ID)
net = FCN8x(NUM_CLASSES)


# net = DeepLabV3(NUM_CLASSES)
# net = UNet(3,NUM_CLASSES)
# 加载网络进行测试


def evaluate(val_image_path):
    from utils.DataLoade import label2image, RandomCrop, image2label
    import matplotlib.pyplot as plt
    from PIL import Image
    image_name = os.path.basename(val_image_path)
    val_label_path = './data/SegmentationClass/' + image_name
    val_label_path = os.path.splitext(val_label_path)[0] + '.png'
    image_name = os.path.basename(val_label_path)
    print(val_label_path)
    print(val_image_path)
    val_image = Image.open(val_image_path)
    val_label = Image.open(val_label_path).convert('RGB').resize((320, 320))
    tfs = transforms.Compose([
        transforms.Resize((320, 320)),  # 调整图像大小
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])  # 归一化
    ])
    input_image = tfs(val_image).unsqueeze(0)
    val_label = image2label()(val_label)
    input_label = torch.from_numpy(val_label).long().unsqueeze(0)
    # 加载模型参数并移至GPU
    net.load_state_dict(torch.load(model_path, map_location='cuda'))
    net.cuda()

    # 进行推理
    with torch.no_grad():
        out = net(input_image.cuda())
        pred = out.argmax(dim=1).squeeze().cpu().numpy()
        pred = np.expand_dims(pred, axis=0)
        label = input_label.data.numpy()
        val_pre, val_labe = label2image(NUM_CLASSES)(pred, label)

    fig, ax = plt.subplots(1, 3, figsize=(30, 30))
    ax[0].imshow(val_image)
    ax[1].imshow(val_label)
    ax[2].imshow(val_pre.squeeze())
    save_path = './user_results/pic_{}_{}'.format(model, image_name)
    plt.savefig(save_path)
    # plt.show()
    return save_path


class ImageWindow(tk.Toplevel):
    def __init__(self, parent, image_path, width=1000, height=600):
        super().__init__(parent)
        self.parent = parent
        self.title("Image Switcher")
        self.geometry(f"{width}x{height}")

        # Create a label to display the image
        self.image_label = tk.Label(self)
        self.image_label.pack()

        # Create labels above the image
        label = tk.Label(self, text="image\t\t\t\tlabel\t\t\t\tprediction", anchor=tk.W)
        label.pack(padx=0, pady=10)

        # Load the first image
        self.load_image(image_path)

        close_button = tk.Button(self, text="Close", command=self.close_window)
        close_button.pack()

    def load_image(self, image_path):
        crop_box = (0, 300, 1000, 700)  # 定义裁剪框（左，上，右，下）
        self.image = Image.open(image_path)
        resize_image = self.image.resize((1000, 1000))
        cropped_image = resize_image.crop(crop_box)
        self.photo = ImageTk.PhotoImage(cropped_image)
        self.image_label.config(image=self.photo)

    def close_window(self):
        self.destroy()  # Destroy the current window
        self.parent.deiconify()  # Show the parent window


class StartWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Start Window")
        self.geometry("500x500")

        start_button = tk.Button(self, text="Select Picture", command=self.open_image)
        start_button.pack(pady=20)
        close_button = tk.Button(self, text='Close', command=self.close_window)
        close_button.pack(pady=20)

    def open_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.withdraw()
            ImageWindow(parent=self, image_path=evaluate(file_path))

    def close_window(self):
        self.destroy()


if __name__ == "__main__":
    app = StartWindow()
    app.mainloop()
