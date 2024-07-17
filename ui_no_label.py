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
from tkinter import messagebox
from model.DeepLab import DeepLabV3
from history import ImageSwitcher
# from torch import nn,optim
# from torch.nn import functional as F
from utils.eval_tool import label_accuracy_score

GPU_ID = 0
INPUT_WIDTH = 320
INPUT_HEIGHT = 320
BATCH_SIZE = 32
NUM_CLASSES = 21
LEARNING_RATE = 1e-3
torch.cuda.set_device(GPU_ID)


def evaluate(val_image_path, model_e):
    from utils.DataLoade import colormap
    import matplotlib.pyplot as plt
    from PIL import Image
    model_path = './model_result/best_model_{}.mdl'.format(model_e)
    if model_e == 'FCN8x':
        net = FCN8x(NUM_CLASSES)
    elif model_e == 'UNet':
        net = UNet(3, NUM_CLASSES)
    elif model_e == 'DeepLabV3':
        net = DeepLabV3(NUM_CLASSES)
    else:
        net = FCN8x(NUM_CLASSES)
    image_name = os.path.basename(val_image_path)
    image_name = image_name.replace(".jpg", ".png")
    val_image = Image.open(val_image_path)
    tfs = transforms.Compose([
        transforms.Resize((320, 320)),  # 调整图像大小
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])  # 归一化
    ])
    input_image = tfs(val_image).unsqueeze(0)
    # 加载模型参数并移至GPU
    net.load_state_dict(torch.load(model_path, map_location='cuda'))
    net.cuda()

    # 进行推理
    with torch.no_grad():
        out = net(input_image.cuda())
        pred = out.argmax(dim=1).squeeze().cpu().numpy()
        pred = np.expand_dims(pred, axis=0)
        colormap = np.array(colormap).astype('uint8')
        val_pre = colormap[pred]

    fig, ax = plt.subplots(1, 2, figsize=(15, 10))
    ax[0].imshow(val_image)
    ax[1].imshow(val_pre.squeeze())
    ax[0].axis('off')
    ax[1].axis('off')
    save_path = './user_results/history/pic_{}_{}'.format(model_e, image_name)
    plt.savefig(save_path)
    plt.close()  # 关闭当前图形对象
    # plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(val_pre.squeeze())
    ax.axis('off')
    fig.patch.set_facecolor('none')
    pre_path = './user_results/pic_{}_{}'.format(model_e, image_name)
    plt.savefig(pre_path)
    return pre_path


class ImageViewer:
    def __init__(self, parent, image_path, model):
        self.root = tk.Toplevel()  # 使用 Toplevel 创建顶级窗口
        self.root.title("Image Viewer")  # 设置窗口标题
        self.parent = parent
        self.image_path = image_path
        self.model = model
        # 打开并加载图片
        self.image = Image.open(image_path)

        # 创建 PhotoImage 对象
        self.photo = ImageTk.PhotoImage(self.image)

        # 创建 Label 组件来显示图片
        self.label = tk.Label(self.root, image=self.photo)
        self.label.pack()  # 将 Label 放置在窗口中央

        self.predict = tk.Button(self.root, text="predict", command=self.predict)
        self.predict.pack()

        self.close = tk.Button(self.root, text="close", command=self.close_window)
        self.close.pack()

    def close_window(self):
        self.root.destroy()
        self.parent.deiconify()

    def show(self):
        self.root.mainloop()  # 运行主循环，显示窗口

    def predict(self):
        self.root.withdraw()
        image_window = ImageWindow(parent=self.root, image_path=self.image_path,
                                   pre_path=evaluate(self.image_path, self.model))
        image_window.show()


class ImageWindow:
    def __init__(self, parent, image_path, pre_path):
        self.root = tk.Toplevel()  # 使用 Toplevel 创建顶级窗口
        self.parent = parent
        self.root.title("Image Switcher")
        # self.root.geometry(f"{width}x{height}")

        self.image_path1 = image_path
        self.image_path2 = pre_path

        # 加载并显示图片
        self.load_images()

        close_button = tk.Button(self.root, text="Close", command=self.close_window)
        close_button.pack()

    def load_images(self):
        # 打开并加载图片
        image1 = Image.open(self.image_path1)
        image2 = Image.open(self.image_path2)

        # 将图片转换为 tkinter 可用的对象
        self.tk_image1 = ImageTk.PhotoImage(image1)
        self.tk_image2 = ImageTk.PhotoImage(image2)

        # 在窗口中显示图片
        self.label1 = tk.Label(self.root, image=self.tk_image1)
        self.label1.pack(side=tk.LEFT, padx=10, pady=10)  # 左侧显示，设置间距

        self.label2 = tk.Label(self.root, image=self.tk_image2)
        self.label2.pack(side=tk.RIGHT, padx=10, pady=10)  # 右侧显示，设置间距

    def close_window(self):
        self.root.destroy()  # Destroy the current window
        self.parent.deiconify()  # Show the parent window

    def show(self):
        self.root.mainloop()  # 运行主循环，显示窗口


class StartWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Start Window")
        self.root.geometry("300x300")
        self.selected_option = 'FCN8x'
        start_button = tk.Button(self.root, text="Select Picture", command=self.open_image)
        start_button.pack(pady=20)
        model_button = tk.Button(self.root, text='Choose Model', command=self.choose_model)
        model_button.pack(pady=20)
        close_button = tk.Button(self.root, text='Close', command=self.close_window)
        close_button.pack(pady=20)

    def open_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.root.withdraw()
            image_viewer = ImageViewer(image_path=file_path, parent=self.root, model=self.selected_option)
            image_viewer.show()  # 显示 ImageViewer 窗口

    def close_window(self):
        self.root.destroy()

    def show(self):
        self.root.mainloop()

    def choose_model(self):
        def select_option(opt):
            # 更新选择的选项
            self.selected_option = opt
            # 显示选择的选项
            messagebox.showinfo("选择的选项", f"你选择了：{opt}")
            # 关闭选择窗口
            selection_window.destroy()

        # 创建一个新的选择窗口
        selection_window = tk.Toplevel(self.root)
        selection_window.title("选择窗口")

        # 创建选项按钮
        options = ["FCN8x", "UNet", "DeepLabV3"]
        for option in options:
            button = tk.Button(selection_window, text=option, command=lambda opt=option: select_option(opt))
            button.pack(pady=5)


if __name__ == "__main__":
    app = StartWindow()
    app.show()
