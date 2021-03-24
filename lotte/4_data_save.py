import glob
import numpy as np 
from PIL import Image


#==============================================================train
caltech_dir =  'C:/Study/lotte/data/train/'
categories = []
for i in range(0,1000) :
    i = "%d"%i
    categories.append(i)

nb_classes = len(categories)

image_w = 256
image_h = 256

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

np.save("C:/Study/lotte/data/npy/224_project_x.npy", arr=X)
np.save("C:/Study/lotte/data/npy/224_project_y.npy", arr=y)
x = np.load("C:/Study/lotte/data/npy/224_project_x.npy",allow_pickle=True)
y = np.load("C:/Study/lotte/data/npy/224_project_y.npy",allow_pickle=True)

print(x.shape)
print(y.shape)




#==============================================================test_all
img1=[]
for i in range(0,72000):
    filepath='C:/Study/lotte/data/test/%d.jpg'%i
    image2=Image.open(filepath)
    image2 = image2.convert('RGB')
    image2 = image2.resize((224,224))
    image_data2=np.asarray(image2)
    # image_data2 = signal.medfilt2d(np.array(image_data2), kernel_size=3)
    print(i)
    img1.append(image_data2)    

np.save('C:/Study/lotte/data/npy/224_test.npy', arr=img1)
# alphabets = string.ascii_lowercase
# alphabets = list(alphabets)


# x = np.load('../data/csv/Dacon3/train4.npy')
x_pred = np.load('C:/Study/lotte/data/npy/224_test.npy',allow_pickle=True)

print(x_pred.shape) #(72000, 128, 128, 3)