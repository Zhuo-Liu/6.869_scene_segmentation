import numpy as np
from PIL import Image
import torch
import cv2
from matplotlib import pyplot as plt

def PAM():
    attention = np.load('attention.npy')[0]
    print(attention.shape)

    img = Image.open(r"./test2.jpg")
    # width, height = img.size
    # print(width,height)

    newsize = (attention.shape[0]*8//512,512//8)
    w, h = newsize
    print(newsize)
    #im1 = img.resize(newsize)

    #im1.show()
    # gray_image = Image.ImageOps.grayscale(im1)

    car_point = [50,50]
    M = w * car_point[0] + car_point[1]
    car_attention = attention[:,0].squeeze()
    car_attention = np.resize(car_attention,(h,w))
    min = np.amin(car_attention)
    max = np.amax(car_attention)

    for i in range(car_attention.shape[0]):
        for j in range(car_attention.shape[1]):
            car_attention[i][j] = (car_attention[i][j] - min) / (max - min) * 255
            # if car_attention[i][j] < 100:
            #     car_attention[i][j] = 0

    car_attention = car_attention.astype(np.uint8)

    img2 = Image.fromarray(car_attention, 'L')
    img2 = img2.resize(img.size)
    img2.save('./attention.jpg')

def CAM():
    img = cv2.imread("test2.jpg")[:,:,::-1]
    img = cv2.resize(img, (224, 224))
    activations = np.load('sc_conv.npy')[0]
    weights = postprocess_activations(activations)
    heatmap = apply_heatmap(weights, img)
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    ax = plt.imshow(heatmap)
    return

def apply_heatmap(weights, img):
  #generate heat maps 
  heatmap = cv2.applyColorMap(weights, cv2.COLORMAP_JET)
  heatmap = cv2.addWeighted(heatmap, 0.7, img, 0.3, 0)
  return heatmap

def postprocess_activations(activations, k):

  output = np.abs(activations)
  #output = np.sum(output, axis = -1).squeeze()
  output = output[:,:,k]

  #resize and convert to image 
  output = cv2.resize(output, (224, 224))
  output /= output.max()
  output *= 255
  return 255 - output.astype('uint8')
  
if __name__ == '__main__':
    #PAM()
    #CAM()
    img = cv2.imread("test2.jpg")[:,:,::-1]
    img = cv2.resize(img, (224, 224))
    activations = np.load('sc_conv.npy')[0]
    for k in range(30):
        weights = postprocess_activations(activations, k)
        heatmap = apply_heatmap(weights, img)
        plt.figure(figsize=(12, 10))
        plt.axis('off')
        plt.imshow(heatmap)
        plt.savefig('./att/{}.jpg'.format(k))
        plt.clf()