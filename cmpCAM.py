import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
import cv2
import pandas as pd
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt
from GradCAM import GradCAM

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

def getCamOnImg(img, cam):
    img = img.convert('RGBA')
    cmap = plt.get_cmap('jet')
    cam = cv2.resize(cam, img.size, interpolation=cv2.INTER_CUBIC)
    cam = np.uint8(255 * cmap(cam))
    cam = Image.fromarray(cam).convert('RGBA') 
    camOnImg = Image.blend(img, cam, 0.6)
    return camOnImg

class IndexTracker(object):
    def __init__(self, axs, imgs, titles):
        self.idx = 0
        self.axs = axs
        self.imgs = imgs
        self.titles = titles
        self.slices = len(self.imgs)
        
        self.gradCAM = GradCAM(models.resnet50(pretrained=True))
        self.layerName = 'layer2'

        self.update()

    def onscroll(self, event):
        # print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.idx = (self.idx - 1) % self.slices
        else:
            self.idx = (self.idx + 1) % self.slices
        self.update()

    def update(self):
        path, pred, label = self.imgs[self.idx]

        img = imgTrans(path)
        self.im = self.axs[0].imshow(img)

        cam = self.gradCAM.getCAM(path, self.layerName, pred)
        camOnImg = getCamOnImg(img, cam)     
        self.axs[1].imshow(camOnImg)

        cam = self.gradCAM.getCAM(path, self.layerName, label)
        camOnImg = getCamOnImg(img, cam)     
        self.axs[2].imshow(camOnImg)

        self.axs[1].set_title(self.titles[self.idx])

        for ax in self.axs.flat:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set(aspect=1, adjustable='box')

        self.im.axes.figure.canvas.draw()


dataRoot = '../_datasets/weather/fog/1/'
testPath = 'test.csv'
predPath = 'pred.csv'

preds = pd.read_csv(predPath, usecols=[0]).values.T[0]
names = pd.read_csv(testPath, usecols=[0]).values.T[0]
labels = pd.read_csv(testPath, usecols=[1]).values.T[0]
clsloc = pd.read_csv('clsloc.csv').values.T[0]

wrongImgs = []
wrongMsgs = []
for i, label in enumerate(labels):
    if preds[i] != label:
        img = dataRoot + names[i]
        wrongImgs.append([img, preds[i], label])
        msg = "Pred: " + clsloc[preds[i]] + " $\leftarrow$ " + clsloc[label]
        wrongMsgs.append(msg)

err = len(wrongImgs)
acc = 1 - err/len(preds)
print('Total errors:', err, '  Accuracy: {:.3f} %'.format(100 * acc))


fig, axs = plt.subplots(1, 3)
tracker = IndexTracker(axs, wrongImgs, wrongMsgs)

fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
fig.tight_layout()
fig.subplots_adjust(wspace =0, hspace=0)
plt.show()
