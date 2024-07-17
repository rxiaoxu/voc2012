import tkinter as tk
from PIL import Image, ImageTk


class ImageViewer:
    def __init__(self, image_path):
        self.root = tk.Tk()  # 创建主窗口
        self.root.title("Image Viewer")  # 设置窗口标题

        # 打开并加载图片
        self.image = Image.open(image_path)

        # 创建 PhotoImage 对象
        self.photo = ImageTk.PhotoImage(self.image)

        # 创建 Label 组件来显示图片
        self.label = tk.Label(self.root, image=self.photo)
        self.label.pack()  # 将 Label 放置在窗口中央

    def run(self):
        self.root.mainloop()  # 运行主循环，显示窗口


# 示例使用
if __name__ == "__main__":
    image_path = "./data/JPEGImages/2010_004369.jpg"
    image_viewer = ImageViewer(image_path)
    # image_viewer.run()
