import os
import csv

def writeCsv(data, path):
    with open(path, 'w') as f:
        w = csv.writer(f)
        w.writerows(data)

dataRoot = '../_datasets/weather/brightness/1'

# test = [['name', 'label']]

# dirs = os.listdir(dataRoot)
# dirs.sort()
# for label, synset in enumerate(dirs):
#     imgs = os.listdir(os.path.join(dataRoot, synset))
#     for img in imgs:
#         name = os.path.join(synset, img)
#         # name = img
#         test.append([name, label])

# writeCsv(test, 'test.csv')


clsloc = [['name', 'label']]
with open('map_clsloc.txt') as f:
    lines = f.readlines()
lines.sort()
for i, line in enumerate(lines):
    name = list(filter(None, line.split(" ")))
    name = name[2].strip("\n")
    clsloc.append([name, i])

writeCsv(clsloc, 'clsloc.csv')