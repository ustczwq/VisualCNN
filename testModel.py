import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
import csv
from Model import CamDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)]
)

model = models.resnet50(pretrained=True)
model = model.to(device)
model.eval()

testData = datasets.ImageFolder(
    root='../_datasets/weather/fog/1',
    transform=transform
)

# # clean dataset
# testData = CamDataset(
#     csvPath='clean.csv',
#     rootDir='/media/zwq/Data/ILSVRC2012_img_val/',
#     transform=transform
# )

testLoader = DataLoader(testData, batch_size=256, shuffle=False)

total = 0
correct = 0
results = [['pred', 'label']]

with torch.no_grad():
    for i, (images, labels) in enumerate(testLoader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().cpu().numpy()
        print(correct, total, correct/total)

        res = np.array([pred.cpu().numpy(), labels.cpu().numpy()]).T
        results.extend(res)

print('Test Accuracy: {:.3f} %'.format(100 * correct / total))

with open('fog1Pred.csv', 'w') as f:
    w = csv.writer(f)
    w.writerows(results)
