import cv2

training = open('coco_training.odgt', 'w')
with open('.\cocostuff-10k-v1.1\\train.txt') as f:
    lines = f.read().splitlines()
for i in range(len(lines)):
    im = cv2.imread('.\cocostuff-10k-v1.1\\images\\training\\'+lines[i]+'.jpg')
    h, w, c = im.shape
    s='{"fpath_img": "cocostuff-10k-v1.1/images/training/'+lines[i]+'.jpg", "fpath_segm": "cocostuff-10k-v1.1/annotations/training/'+lines[i]+'.png", "width": '+str(w)+', "height": '+str(h)+'}\n'
    training.write(s)
training.close()

test = open('coco_test.odgt', 'w')
with open('.\cocostuff-10k-v1.1\\test.txt') as f:
    lines = f.read().splitlines()
for i in range(len(lines)):
    im = cv2.imread('.\cocostuff-10k-v1.1\\images\\test\\'+lines[i]+'.jpg')
    h, w, c = im.shape
    s='{"fpath_img": "cocostuff-10k-v1.1/images/test/'+lines[i]+'.jpg", "fpath_segm": "cocostuff-10k-v1.1/annotations/test/'+lines[i]+'.png", "width": '+str(w)+', "height": '+str(h)+'}\n'
    test.write(s)
test.close()