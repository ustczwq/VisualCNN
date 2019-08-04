from skimage import io 
import numpy as np 

img = np.zeros((224, 224, 3), dtype=np.uint8)

for h in range(224):
    for w in range(224):
        # if h < 75:
        #     img[h, w] = [255, 0, 0] if w < 112 else [0, 0, 0]
        # elif h < 150:
        #     img[h, w] = [0, 255, 0] if w < 112 else [225, 255, 255]
        # else:
        #     img[h, w] = [0, 0, 255] if w < 112 else [128, 128, 128]
        img[h, w] = [0, 255, 0]
        

io.imsave('inputs/g.jpg', img)