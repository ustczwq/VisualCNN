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
        self.layerName = 'layer1'

        self.update()

    def onscroll(self, event):
        # print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.idx = (self.idx - 1) % self.slices
        else:
            self.idx = (self.idx + 1) % self.slices
        self.update()

    def update(self):
        testPath, cleanPath, pred, label = self.imgs[self.idx]

        self.plotAxs(0, testPath, pred, label)
        self.plotAxs(1, cleanPath, pred, label)

        self.axs[0][1].set_title(self.titles[self.idx])
        for ax in self.axs.flat:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set(aspect=1, adjustable='box')

        self.im.axes.figure.canvas.draw()

    def plotAxs(self, axNum, imgPath, pred, label):
        img = imgTrans(imgPath)
        predCAM  = self.gradCAM.getCAM(imgPath, self.layerName, pred)
        labelCAM = self.gradCAM.getCAM(imgPath, self.layerName, label)

        self.im = self.axs[axNum][0].imshow(img)
        self.axs[axNum][1].imshow(getCamOnImg(img, predCAM))
        self.axs[axNum][2].imshow(getCamOnImg(img, labelCAM))


testDir = '../_datasets/weather/fog/1/'
testRes = 'test.csv'
predRes = 'fog1Pred.csv'

cleanDir = '/media/zwq/Data/ILSVRC2012_img_val/'
cleanPredRes = 'cleanPred.csv'
cleanDataCsv = 'clean.csv'

labels = pd.read_csv(predRes, usecols=[1]).values.T[0]
preds  = pd.read_csv(predRes, usecols=[0]).values.T[0]
cleanPreds = pd.read_csv(cleanPredRes, usecols=[0]).values.T[0]
cleanNames = pd.read_csv(cleanDataCsv, usecols=[0]).values.T[0]
testNames = pd.read_csv(testRes, usecols=[0]).values.T[0]

clsloc = pd.read_csv('clsloc.csv').values.T[0]

imgs = []
msgs = []
for i, label in enumerate(labels):
    if cleanPreds[i] == label and preds[i] != label:
        testImg = testDir + testNames[i]
        cleanImg = cleanDir + cleanNames[i]
        imgs.append([testImg, cleanImg, preds[i], label])
        msg = "Pred: " + clsloc[preds[i]] + " $\leftarrow$ " + clsloc[label]
        msgs.append(msg)

acc = np.sum(preds == labels) / len(labels)
print('Test  Accuracy: {:.3f} %'.format(100 * acc))

acc = np.sum(cleanPreds == labels) / len(labels)
print('Clean Accuracy: {:.3f} %'.format(100 * acc))

err = len(imgs) / np.sum(cleanPreds == labels)
print('Corruption Err: {:.3f} %'.format(100 * err))


fig, axs = plt.subplots(2, 3)
tracker = IndexTracker(axs, imgs, msgs)

fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
fig.tight_layout()
fig.subplots_adjust(wspace =0, hspace=0)

plt.show()
