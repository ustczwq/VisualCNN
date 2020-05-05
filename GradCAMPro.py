import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def denormalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(
        1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(
        1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.mul(std).add(mean)


def normalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(
        1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(
        1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.sub(mean).div(std)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return self.do(tensor)

    def do(self, tensor):
        return normalize(tensor, self.mean, self.std)

    def undo(self, tensor):
        return denormalize(tensor, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class GradCAM():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, model_dict):
        self.maps = dict()
        self.grad = dict()
        self.model_type = model_dict['type']
        self.model = model_dict['arch'].to(self.device)
        self.model.eval()

    def hook_layer(self, layer_name):
        def backward_hook(module, grad_in, grad_out):
            self.grad[layer_name] = grad_out[0]

        def forward_hook(module, grad_in, grad_out):
            self.maps[layer_name] = grad_out

        layer = self.find_layer(self.model_type, self.model, layer_name)

        layer.register_forward_hook(forward_hook)
        layer.register_backward_hook(backward_hook)

    def get_cams(self, img_path, class_idx, visual_dict):
        img = self.load_img(img_path).to(self.device)

        for layer_name in visual_dict['layers']:
            self.hook_layer(layer_name)

        output = self.model(img)
        score = output[:, class_idx].squeeze()
        self.model.zero_grad()
        score.backward(retain_graph=True)

        cams_dict = dict()
        for method in visual_dict['methods']:
            cams_dict[method] = []
            for layer_name in visual_dict['layers']:
                cam = self.get_cam(
                    score, self.grad[layer_name], self.maps[layer_name], method)
                cam = cv2.resize(
                    cam, (224, 224), interpolation=cv2.INTER_CUBIC)
                cams_dict[method].append(cam)

        return cams_dict

    @staticmethod
    def get_cam(score, grad, maps, method):
        b, k, u, v = grad.size()

        # gradients x maps
        if method == 'pro':
            cam = F.relu(F.relu(grad) * maps)

        # grad-cam ++
        elif method == '++':
            alpha_num = grad.pow(2)
            alpha_den = grad.pow(2).mul(2) + \
                maps.mul(grad.pow(3)).view(
                    b, k, -1).sum(-1, keepdim=True).view(b, k, 1, 1)
            alpha_den = torch.where(
                alpha_den != 0.0, alpha_den, torch.ones_like(alpha_den))
            alpha = alpha_num.div(alpha_den + 1e-7)
            positive_grad = F.relu(score.exp() * grad)
            weights = (alpha * positive_grad).view(b,
                                                   k, -1).sum(-1).view(b, k, 1, 1)
            cam = F.relu(weights * maps)

        elif method == '++pro':

            grad = F.relu(grad)
            maps = F.relu(maps)

            ones = torch.ones_like(maps)
            cam = grad * maps
            cam = cam.div(ones + cam)

        # gradients avgpooling, original
        else:
            alpha = F.relu(grad.view(b, k, -1).mean(2))
            weights = alpha.view(b, k, 1, 1)
            cam = F.relu(weights * maps)

        cam = cam.data.cpu().numpy()[0]
        cam = cam.mean(axis=0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))

        return cam

    @staticmethod
    def load_img(path):
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

    @staticmethod
    def find_layer(model_type, model_arch, layer_name):

        if model_type == 'vgg':
            return model_arch.features[int(layer_name)]
        else:
            return model_arch._modules[layer_name]


def img_trans(img_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)])
    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    # img = np.asarray(img)
    return img


def plotCAM(img_path, cams_dict, res_name='result.png'):
    img = img_trans(img_path).convert('RGBA')

    keys = cams_dict.keys()

    cams_num = len(cams_dict[list(keys)[0]])

    fig, axs = plt.subplots(len(keys), cams_num + 1)

    for row_idx, key in enumerate(keys):
        axs[row_idx][0].imshow(img)
        for col_idx in range(cams_num):
            cam_on_img = blend_cam(img, cams_dict[key][col_idx])
            axs[row_idx][col_idx + 1].imshow(cam_on_img)

    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set(aspect=1, adjustable='box')

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.0, hspace=0.0)
    fig.savefig('./results/' + res_name)

    plt.show()


def plot_imgs_cams(imgs_cams, res_name='result.png'):
    keys = list(imgs_cams.keys())

    cams_num = len(imgs_cams[keys[0]][1])

    fig, axs = plt.subplots(len(keys), cams_num + 1, figsize=(4, 12))

    for row_idx, key in enumerate(imgs_cams):
        img_path, cams = imgs_cams[key]
        img = img_trans(img_path).convert('RGBA')
        axs[row_idx][0].imshow(img)
        for col_idx, cam in enumerate(cams):
            cam_on_img = blend_cam(img, cam)
            axs[row_idx][col_idx + 1].imshow(cam_on_img)

    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set(aspect=1, adjustable='box')

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.0, hspace=0.0)
    
    fig.savefig('./results/' + res_name)

    plt.show()



def blend_cam(img_pil, cam):
    cmap = plt.get_cmap('jet')
    cam = np.uint8(255 * cmap(cam))
    cam = Image.fromarray(cam).convert('RGBA')

    cam_on_img = Image.blend(img_pil, cam, 0.6)

    return cam_on_img


if __name__ == "__main__":

    imgs = [['snake.jpg', 56],   # 0  snake
            ['cat_dog.png', 243],  # 1  bull mastiff dog
            ['cat_dog.png', 281],  # 2  tabby cat
            ['spider.png', 72],   # 3  spider
            ['dd_tree.jpg', 31],   # 4  tree
            ['239_281.png', 239],  # 5  Bernese mountain dog
            ['239_281.png', 281],  # 6  tabby cat
            ['all.jpeg', 281]]  # 7  multi dogs

    img_idx = 1
    img_dir = 'inputs'
    img_name, class_idx = imgs[img_idx]
    img_path = os.path.join(img_dir, img_name)

    model_dict = dict()

    model_dict['arch'] = models.resnet50(pretrained=True)
    model_dict['type'] = 'resnet'

    # model_dict['arch'] = models.vgg16(pretrained=True)
    # model_dict['type'] = 'vgg'

    gradCAM = GradCAM(model_dict)

    layers = dict()
    layers['resnet'] = ['layer4', 'layer3', 'layer2', 'layer1']
    layers['vgg'] = ['29', '24', '19', '14']
    methods = ['orignal', '++', '++pro']

    visual_dict = dict()
    visual_dict['layers'] = layers[model_dict['type']]
    visual_dict['methods'] = methods

    # cams_dict = gradCAM.get_cams(img_path, class_idx, visual_dict)

    # res_name = "%s_%s_%d.png" % (img_name.split('.')[0], model_dict['type'], class_idx)
    # plotCAM(img_path, cams_dict, res_name)

    imgs = [['snake.jpg', 56],   # 0  snake
            ['cat_dog.png', 243],  # 1  bull mastiff dog
            ['cat_dog.png', 281],  # 2  tabby cat
            ['spider.png', 72],   # 3  spider
            ['dd_tree.jpg', 31],   # 4  tree
            ['239_281.png', 239],  # 5  Bernese mountain dog
            ['239_281.png', 281],  # 6  tabby cat
            ['all.jpeg', 281]]  # 7  multi dogs

    img_dir = 'inputs'
    visual_dict = dict()
    visual_dict['layers'] = ['layer4', 'layer3', 'layer2']
    visual_dict['methods'] = ['++pro']

    cams_dict = dict()
    imgs_cams = dict()

    for i, img in enumerate(imgs):
        img_name, class_idx = img
        img_path = os.path.join(img_dir, img_name)
        cams_dict = gradCAM.get_cams(img_path, class_idx, visual_dict)
        imgs_cams[str(i)] = [img_path, cams_dict['++pro']]

    plot_imgs_cams(imgs_cams)
