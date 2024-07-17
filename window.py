import os
import tkinter as tk
from PIL import Image, ImageTk


class ImageSwitcher(tk.Toplevel):
    def __init__(self, parent, image_folder='./pic_results', width=800, height=450):
        super().__init__(parent)
        self.title("Image Switcher")
        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(image_folder) if
                            f.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        self.image_index = 0

        self.geometry(f"{width}x{height}")

        # Create labels above the image
        label = tk.Label(self, text="image\t\t\t\tlabel\t\t\t\tprediction", anchor=tk.W)
        label.pack(padx=0, pady=10)

        # Create a label to display the image
        self.image_label = tk.Label(self)
        self.image_label.pack()

        # Create a button to switch to the next image
        self.next_button = tk.Button(self, text="Next", command=self.next_image)
        self.next_button.pack(padx=0, pady=30)

        # Load the first image
        self.load_image()

    def load_image(self):
        crop_box = (0, 300, 800, 550)  # 定义裁剪框（左，上，右，下）
        image_path = os.path.join(self.image_folder, self.image_files[self.image_index])
        self.image = Image.open(image_path)
        resize_image = self.image.resize((800, 800))
        cropped_image = resize_image.crop(crop_box)
        self.photo = ImageTk.PhotoImage(cropped_image)
        self.image_label.config(image=self.photo)

    def next_image(self):
        self.image_index = (self.image_index + 1) % len(self.image_files)
        self.load_image()


class StartWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Start Window")
        self.geometry("500x500")

        start_button = tk.Button(self, text="Start", command=self.open_image_switcher)
        start_button.pack(pady=20)

    def open_image_switcher(self):
        self.withdraw()
        ImageSwitcher(self)


if __name__ == "__main__":
    app = StartWindow()
    app.mainloop()
