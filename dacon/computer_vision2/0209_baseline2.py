import os
from typing import Tuple, Sequence, Callable
import csv
import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.optim as optim
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary

from torchvision import transforms
from torchvision.models import resnet50

def run():
    torch.multiprocessing.freeze_support()
    print('loop')
# if __name__ == '__main__':
#     run()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 디바이스 설정
    
# 1. 커스텀 데이터셋 만들기
# https://wikidocs.net/57165
class MnistDataset(Dataset):
    # 초기화(initialize) 메서드
    # PathLike :  파일 시스템 경로 프로토콜을 구현하기위한 추상 기본 클래스.
    # https://www.python.org/dev/peps/pep-0519/#protocol
    def __init__(
        self,
        dir: os.PathLike,      
        image_ids: os.PathLike,
        transforms: Sequence[Callable]   # Sequence타입?
    ) -> None:
        self.dir = dir
        self.transforms = transforms

        self.labels = {}
        with open(image_ids, 'r') as f:
            reader = csv.reader(f)  # reader : __next__() 메서드가 호출될 때마다 문자열을 반환하는 객체
            next(reader)
            for row in reader:
                self.labels[int(row[0])] = list(map(int, row[1:]))

        self.image_ids = list(self.labels.keys())

    def __len__(self) -> int:  #  __len__ 은 데이터셋의 크기를 리턴
        return len(self.image_ids)

    def __getitem__(self, index: int) -> Tuple[Tensor]:  # __getitem__ 은  i번째 샘플을 가져오도록 하는 인덱스를 찾는데 사용
        image_id = self.image_ids[index]
        image = Image.open(
            os.path.join(
                self.dir, f'{str(image_id).zfill(5)}.png')).convert('RGB')
        target = np.array(self.labels.get(image_id)).astype(np.float32)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, target


# 2. 이미지 어그멘테이션    : 이미지에 변화
# https://tutorials.pytorch.kr/beginner/data_loading_tutorial.html
transforms_train = transforms.Compose([  # 여러 transform 들을 Compose로 구성
    transforms.RandomHorizontalFlip(p=0.5),# 주어진 확률로 주어진 이미지를 무작위로 수평으로 뒤집습니다
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(), #  numpy 이미지에서 torch 이미지로 변경합니다.
    transforms.Normalize(       # 학습을 위해 데이터 분포를 쉽게 정규화
        [0.485, 0.456, 0.406],  # 세 채널 모두에 대한 평균
        [0.229, 0.224, 0.225]   # 세 채널 모두에 대한 표준 편차
    )
])

transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

path = './dacon/computer2/data'

trainset = MnistDataset('./dacon/computer2/data/dirty_mnist_2nd', './dacon/computer2/data/dirty_mnist_2nd_answer.csv', transforms_train)
testset = MnistDataset('./dacon/computer2/data/test_dirty_mnist_2nd', './dacon/computer2/data/sample_submission.csv', transforms_test)


# num_workers : data 로딩을 위해 몇 개의 서브 프로세스를 사용할 것인지를 결정
train_loader = DataLoader(trainset, batch_size=256, num_workers=0)  # num_workers=8(https://aigong.tistory.com/136)
test_loader = DataLoader(testset, batch_size=32, num_workers=0)     # num_workers=4


# 3. ResNet50 모형
class MnistModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet = resnet50(pretrained=True)
        self.classifier = nn.Linear(1000, 26)

    def forward(self, x):
        x = self.resnet(x)
        x = self.classifier(x)

        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MnistModel().to(device)
print(summary(model, input_size=(1, 3, 256, 256), verbose=0))

# 4. 학습하기
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MultiLabelSoftMarginLoss()

num_epochs = 10
model.train()

for epoch in range(num_epochs):
        
    for i, (images, targets) in enumerate(train_loader):
        optimizer.zero_grad()

        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            outputs = outputs > 0.5
            acc = (outputs == targets).float().mean()
            print(f'{epoch}: {loss.item():.5f}, {acc.item():.5f}')


# 5. 추론하기
submit = pd.read_csv('./dacon/computer2/data/sample_submission.csv')

model.eval()
batch_size = test_loader.batch_size
batch_index = 0
for i, (images, targets) in enumerate(test_loader):
    images = images.to(device)
    targets = targets.to(device)
    outputs = model(images)
    outputs = outputs > 0.5
    batch_index = i * batch_size
    submit.iloc[batch_index:batch_index+batch_size, 1:] = \
        outputs.long().squeeze(0).detach().cpu().numpy()
    
submit.to_csv('./dacon/computer2/csv/0207_baseline.csv', index=False)