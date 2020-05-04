import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image
# from skimage import io
import matplotlib.pyplot as plt


class GradCAM():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, model):
        self.maps = None
        self.grad = None
        self.model = model.to(self.device)
        self.model.eval()

    def hookLayer(self, layerName):
        def hookFunc(module, gradIn, gradOut):
            self.grad = gradOut[0]
        self.model._modules[layerName].register_backward_hook(hookFunc)

    def getCAM(self, imgPath, layerName, classNum):
        img = self.imgLoad(imgPath)
        img = img.to(self.device)
        self.hookLayer(layerName)

        maps, outputs = self.getMapsOutputs(img, self.model, layerName)
        # maps = maps.data.cpu().numpy()[0]

        oneHot = torch.zeros((1, outputs.size()[-1]), device=self.device)
        oneHot[0][classNum] = 1

        self.model.zero_grad()
        outputs.backward(gradient=oneHot, retain_graph=True)
        # grad = self.grad.data.cpu().numpy()[0]

        # weights = grad.mean(axis=(1, 2), keepdims=True)    # gradients avgpooling, original
        # weights = grad                                       # gradients on every pixel, better look
        # weights = np.maximum(weights, 0)

        cam = self.grad * maps
        cam = cam.data.cpu().numpy()[0]
        cam = np.maximum(cam, 0)
        cam = cam.mean(axis=0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        # cam = np.uint8(cam * 255)
 
        return cam

    @staticmethod
    def getMapsOutputs(img, model, layerName):
        x = img
        maps = None
        for i, (name, layer) in enumerate(model._modules.items()):
            if name == 'fc':
                x = x.reshape(x.size(0), -1)
            x = layer(x)

            if name == layerName:
                maps = x

        return maps, x

    @staticmethod
    def imgLoad(path):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        img = Image.open(path).convert('RGB')
        ten = transform(img)
        return ten.unsqueeze(0)


def imgTrans(imgPath):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)])
    img = Image.open(imgPath).convert('RGB')
    img = transform(img)
    # img = np.asarray(img)
    return img

def plotCAM(imgPath, cam):
    img = imgTrans(imgPath).convert('RGBA')

    cmap = plt.get_cmap('jet')
    cam = np.uint8(255 * cmap(cam))
    cam = Image.fromarray(cam).convert('RGBA')
    
    camOnImg = Image.blend(img, cam, 0.6)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img)  
    axs[1].imshow(camOnImg)

    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set(aspect=1, adjustable='box')

    fig.tight_layout()
    fig.subplots_adjust(wspace =0, hspace=0)
    plt.show()
    # fig.savefig('./outputs' + str(classNum) + layerName + '.jpg')


if __name__ == "__main__":

    img = [['snake.jpg'  , 56],   # 0  snake 
           ['cat_dog.png', 243],  # 1  bull mastiff dog
           ['cat_dog.png', 281],  # 2  tabby cat 
           ['spider.png' , 72],   # 3  spider
           ['dd_tree.jpg', 31],   # 4  tree
           ['239_281.png', 239],  # 5  Bernese mountain dog
           ['239_281.png', 281],  # 6  tabby cat
           ['color.jpg'  , 340]]  # 7  test color

    model = models.resnet50(pretrained=True) 
    
    gradCAM = GradCAM(model)

    index = 1
    imgPath = 'inputs/' + img[index][0]
    classNum = img[index][1]
    layerName = 'layer4'   # conv1, bn1, relu, maxpool, layer1-4

    cam = gradCAM.getCAM(imgPath, layerName, classNum)
    cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)

    plotCAM(imgPath, cam)
