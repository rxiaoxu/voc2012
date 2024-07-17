import os
import tkinter as tk
from PIL import Image, ImageTk


class ImageSwitcher:
    def __init__(self, parent, image_folder='./user_results/history'):
        self.root = tk.Toplevel()
        self.parent = parent
        self.root.title("Image Switcher")
        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(image_folder) if
                            f.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        self.image_index = 0

        # self.geometry(f"{width}x{height}")

        # Create labels above the image
        label = tk.Label(self.root, text="image\t\t\t\tlabel\t\t\t\tprediction", anchor=tk.W)
        label.pack(padx=0, pady=10)

        # Create a label to display the image
        self.image_label = tk.Label(self.root)
        self.image_label.pack()

        # Create a button to switch to the next image
        self.next_button = tk.Button(self.root, text="Next", command=self.next_image)
        self.next_button.pack(padx=0, pady=30)

        self.close_button = tk.Button(self.root, text="Close", command=self.close)
        self.close_button.pack()
        # Load the first image
        self.load_image()

    def load_image(self):
        image_path = os.path.join(self.image_folder, self.image_files[self.image_index])
        self.image = Image.open(image_path)
        self.photo = ImageTk.PhotoImage(self.image)
        self.image_label.config(image=self.photo)

    def next_image(self):
        self.image_index = (self.image_index + 1) % len(self.image_files)
        self.load_image()

    def show(self):
        self.root.mainloop()

    def close(self):
        self.root.destroy()
        self.parent.deiconify()
