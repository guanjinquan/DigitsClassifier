import torch
from torchvision.transforms import Compose, Normalize, Resize, RandomAdjustSharpness, ToTensor
import os
from PIL import Image
from net import Net
import warnings
warnings.filterwarnings("ignore")


def read_image(file_path):
    img = Image.open(file_path)  # 读进来是彩色图片
    img = img.convert("L")       # 转为灰度图片 
    return img


if __name__ == "__main__":
    pretrained_model_path = "./model.pth"
    mydata_path = "./mydata/"
    
    try:
        net = torch.load(pretrained_model_path)
    except:
        print("[ERROR] No model found!")
        exit(1)
    
    transforms = Compose([  # 保证输入时PLI格式的图片
        ToTensor(),
        Normalize(mean = [0.5], std = [0.5]),
        Resize((64, 64), antialias=None),
        RandomAdjustSharpness(sharpness_factor=10, p=1)
    ])
    
    accuracy = 0
    net.eval()
    with torch.no_grad():
        for file_name in os.listdir(mydata_path):
            file_path = mydata_path + file_name
            img = read_image(file_path)     # 读取图片PIL格式
            img = transforms(img)             # 先处理数据
            
            input = img.view(1, img.shape[0], img.shape[1], img.shape[2])  # 设置batch_size=1
            output = net(input)                                              # 再进行推导
            predicted = torch.argmax(torch.softmax(output, dim=1), dim=-1)
            print("File: %s, Predicted: %d" % (file_name, predicted))
            
            if int(file_name[0]) == predicted:  # 默认文件名的第一个字符是数字，代表这张图片的真实值
                accuracy += 1
    
    print("Accuracy: %.4f" % (accuracy / len(os.listdir(mydata_path))))
            
        