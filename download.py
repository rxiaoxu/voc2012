import torch
import torchvision.models as models
import os

dir = "./pretrained_models"
if not os.path.exists(dir):
    os.mkdir(dir)
directory = "./pretrained_models/resnet"  # 例如："/path/to/save/directory"
if not os.path.exists(directory):
    os.mkdir(directory)
def save_model(model, path):
    torch.save(model.state_dict(), path)


def download_and_save_resnet_models(directory):
    # 确保目录存在
    os.makedirs(directory, exist_ok=True)

    # 下载 ResNet-18
    resnet18 = models.resnet18(pretrained=True)
    save_model(resnet18, os.path.join(directory, 'resnet18.pth'))
    print("ResNet-18 saved.")

    # 下载 ResNet-34
    resnet34 = models.resnet34(pretrained=True)
    save_model(resnet34, os.path.join(directory, 'resnet34.pth'))
    print("ResNet-34 saved.")

    # 下载 ResNet-50
    resnet50 = models.resnet50(pretrained=True)
    save_model(resnet50, os.path.join(directory, 'resnet50.pth'))
    print("ResNet-50 saved.")


if __name__ == "__main__":
    directory = "./pretrained_models/resnet"  # 例如："/path/to/save/directory"
    if not os.path.exists(directory):
        os.mkdir(directory)
    download_and_save_resnet_models(directory)
