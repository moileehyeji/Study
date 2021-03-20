import glob
import numpy as np 
from PIL import Image

caltech_dir =  'C:/Study/lotte/data/train/'
categories = []
for i in range(0,1000) :
    i = "%d"%i
    print(i)
    categories.append(i)

nb_classes = len(categories)

image_w = 128
image_h = 128

pixels = image_h * image_w * 3

X = []
y = []

for idx, cat in enumerate(categories):
    
    #one-hot 돌리기.
    label = [0 for i in range(nb_classes)]
    label[idx] = 1

    image_dir = caltech_dir + "/" + cat
    files = glob.glob(image_dir+"/*.jpg")
    print(cat, " 파일 길이 : ", len(files))
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)

        X.append(data)
        y.append(label)

        if i % 700 == 0:
            print(cat, " : ", f)

X = np.array(X)
y = np.array(y)

np.save("C:/Study/lotte/data/npy/4_128_x.npy", arr = X)
np.save("C:/Study/lotte/data/npy/4_128_y.npy", arr = y)

print(X.shape)  #(48000, 128, 128, 3)
print(y.shape)  #(48000, 1000)


img1=[]
for i in range(0,72000):
    filepath='C:/Study/lotte/data/test/%d.jpg'%i
    image2=Image.open(filepath)
    image2 = image2.convert('RGB')
    image2 = image2.resize((128,128))
    image_data2=np.asarray(image2)
    # image_data2 = signal.medfilt2d(np.array(image_data2), kernel_size=3)
    img1.append(image_data2)    

np.save("C:/Study/lotte/data/npy/4_128_test.npy", arr = img1)
np.load("C:/Study/lotte/data/npy/4_128_test.npy", allow_pickle=True)
print(img1.shape)
