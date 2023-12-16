import torch

class Block(torch.nn.Module):
    def __init__(self, in_features, out_features, kernelsize, stride, padding, downsample=True):
        super(Block, self).__init__()
        
        layers = [
            torch.nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernelsize, stride=stride, padding=padding),
            torch.nn.BatchNorm2d(out_features, affine=True),
            torch.nn.ReLU(),
        ]
        
        affine_layers = [
            torch.nn.Conv2d(in_channels=in_features,out_channels=out_features,kernel_size=kernelsize,stride=stride,padding=padding),
        ]
        
        if downsample:
            layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
            affine_layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.model = torch.nn.Sequential(*layers)
        self.affine = torch.nn.Sequential(*affine_layers)
    
    def forward(self, input):
        output = self.model(input)
        residual = self.affine(input)
        return output + residual


class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        
        # 预处理层
        laplace = torch.nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 3, stride = 1, padding = 1).cpu()
        laplace.weight = torch.nn.Parameter(torch.tensor([
            [[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]
        ], dtype=torch.float32), requires_grad=False)  # laplace 卷积，能够提取边缘信息，用于增强图像的边缘信息
        
        erode = torch.nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 3, stride = 1, padding = 1).cpu()
        erode.weight = torch.nn.Parameter(torch.tensor([
            [[[0, 0.8, 0], [0.8, 2, 0.8], [0, 0.8, 0]]]
        ], dtype=torch.float32), requires_grad=False)  # erode 卷积，能够腐蚀图像不清晰的边缘，使得数字更加清晰
        
        dilate = torch.nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
        
        open_operation = torch.nn.Sequential(erode, dilate)  # 开运算，先腐蚀后膨胀，能够去除图像中的小噪点
        
        self.preprocess = torch.nn.ModuleList([open_operation, erode, laplace, erode, laplace])  # 预处理
        
        # 训练模型
        self.model = torch.nn.Sequential(
            #The size of the picture is 64 x 64
            Block(1, 16, 3, 1, 1, downsample=True),
            #The size of the picture is 32 x 32
            Block(16, 32, 3, 1, 1, downsample=True),
            #The size of the picture is 16 x 16
            Block(32, 64, 3, 1, 1, downsample=True),
            #The size of the picture is 8 x 8
            Block(64, 128, 3, 1, 1, downsample=False),
            
            torch.nn.Flatten(),
            torch.nn.Linear(in_features = 8 * 8 * 128, out_features = 256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features = 256, out_features = 10),  # out_feature=10，因为是一个10分类问题
            # torch.nn.Softmax(dim=1)
        )
        
    def forward(self, input):
        for module in self.preprocess:
            input = module(input)
        
        # PIL图片数据范围为(0,255), 经过ToTensor之后变成了(0,1), 这里再进行一次归一化，数据范围变成(-1, 1)
        # 所以为了保持数据范围不变，下面进行数值上下界的truncate
        input = torch.where(input > 1, 1, input)
        input = torch.where(input < -1, -1, input)
        
        # 再进行模型的推理
        output = self.model(input)
        return output