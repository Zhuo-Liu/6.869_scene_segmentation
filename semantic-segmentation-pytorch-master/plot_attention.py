import numpy as np
from PIL import Image
import torch

attention = np.load('attention.npy')[0]
print(attention.shape)

img = Image.open(r"./1.jpg")
width, height = img.size

newsize = (attention.shape[0]*8//512,512//8)
w, h = newsize
im1 = img.resize(newsize)
im1.show()

car_point = [66,61]
M = w * car_point[1] + car_point[0]
car_attention = attention[:,M:M+1].squeeze()
car_attention = np.resize(car_attention,(h,w))
min = np.amin(car_attention)
max = np.amax(car_attention)

for i in range(car_attention.shape[0]):
    for j in range(car_attention.shape[1]):
        car_attention[i][j] = (car_attention[i][j] - min) / (max - min) * 255

car_attention.astype(np.uint8)

img2 = Image.fromarray(car_attention, 'L')
img2.show()


# a = np.array([[[1,1,1],
#               [2,2,2]]])

# at = torch.from_numpy(a)
# C,H,W = at.size()
# print(C,H,W)
# at_flattened = at.view(C, -1)
# print(at_flattened)

# b_flattened = at_flattened.numpy()
# print(b_flattened)
# #b = b_flattened[0].resize(H,W)
# b = np.resize(b_flattened[0],(H,W))
# print(b)