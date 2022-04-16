import scipy.io, shutil
from PIL import Image as im

DIR = 'C:\\Users\yfphy\Downloads\cocostuff-10k-v1.1\\'

with open(DIR+'imageLists\\train.txt') as f:
    lines = f.read().splitlines()

for i in range(len(lines)):
    mat = scipy.io.loadmat(DIR+'annotations\\'+lines[i]+'.mat')
    data = im.fromarray(mat['S'])
    data.save('.\coco_annotations\\training\\'+lines[i]+'.png')
    if (i+1)%100==0:
        print(f'process: {i+1}/{len(lines)}')

for i in range(len(lines)):
    src = DIR+'images\\'+lines[i]+'.jpg'
    dst = '.\coco_images\\training\\'+lines[i]+'.jpg'
    shutil.copyfile(src, dst)

with open(DIR+'imageLists\\test.txt') as f:
    lines = f.read().splitlines()

for i in range(len(lines)):
    mat = scipy.io.loadmat(DIR+'annotations\\'+lines[i]+'.mat')
    data = im.fromarray(mat['S'])
    data.save('.\coco_annotations\\test\\'+lines[i]+'.png')
    if (i+1)%100==0:
        print(f'process: {i+1}/{len(lines)}')

for i in range(len(lines)):
    src = DIR+'images\\'+lines[i]+'.jpg'
    dst = '.\coco_images\\test\\'+lines[i]+'.jpg'
    shutil.copyfile(src, dst)