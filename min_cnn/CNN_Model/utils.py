import os  # 파일 경로를 다루기 위한 표준 라이브러리
import pandas as pd  # CSV 파일을 다루기 위한 라이브러리
import torch  # PyTorch 프레임워크
from PIL import Image, ImageOps  # 이미지 처리를 위한 라이브러리 (좌우 반전 위해 ImageOps 추가)
from torch.utils.data import Dataset  # PyTorch Dataset 클래스 상속
import numpy as np  # 랜덤 flip을 위해 추가

# 자율주행 조향 데이터셋 클래스 정의
class SteeringDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None,
                 use_random_flip=False, indices=None):
        df = pd.read_csv(csv_path)  # CSV 파일을 pandas DataFrame으로 읽기

        # indices가 주어지면 그 부분만 서브셋으로 사용 (train/valid 분리용)
        if indices is not None:
            df = df.iloc[indices].reset_index(drop=True)

        self.data = df
        self.img_dir = img_dir
        self.transform = transform
        self.use_random_flip = use_random_flip

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(image_path).convert('RGB')
        angle = float(row['steering'])

        # 🔥 학습 단계에서만 사용하는 랜덤 좌우 반전 + 각도 보정
        if self.use_random_flip and np.random.rand() < 0.5:
            image = ImageOps.mirror(image)     # 좌우 반전
            angle = 180.0 - angle              # 조향각도 반전

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(angle, dtype=torch.float32)