import os  # 운영체제 경로 및 파일 처리 모듈
import pandas as pd  # 데이터프레임 처리를 위한 pandas
import numpy as np  # 수치 계산용 numpy
from PIL import Image  # 이미지 처리를 위한 Pillow
import torch  # PyTorch 메인 라이브러리
import torch.nn as nn  # 신경망 레이어와 손실함수 모듈
from torch.utils.data import Dataset, DataLoader, random_split  # 데이터셋 및 분할/로더 유틸리티
import torchvision.transforms as transforms  # 데이터 전처리 변환 모듈
import matplotlib.pyplot as plt  # 시각화를 위한 matplotlib
from tqdm import tqdm  # 진행률 표시 바
from model import SteeringModel, get_unique_train_folder  # 사용자 정의 모델과 학습 폴더 생성기
from utils import SteeringDataset  # 사용자 정의 데이터셋 클래스
from config import *  # 설정 상수 불러오기

# 학습 함수 정의
def train(model, loader, optimizer, criterion, device, clip_grad=None):
    model.train()  # 모델을 학습 모드로 전환
    running_loss = 0.0  # 에폭 동안 손실 누적 변수
    for imgs, angles in tqdm(loader, desc="Training", leave=False):  # 배치 단위 학습 루프
        imgs, angles = imgs.to(device), angles.to(device)  # 데이터를 GPU/CPU로 이동
        optimizer.zero_grad()  # 이전 기울기 초기화
        outputs = model(imgs).squeeze(1) 
        loss = criterion(outputs, angles)
        #loss = criterion(model(imgs).squeeze(), angles)  # 예측과 실제 각도의 손실 계산
        loss.backward()  # 손실 역전파
        # gradient clipping to avoid exploding gradients
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()  # 가중치 업데이트
        running_loss += loss.item() * imgs.size(0)  # 배치 손실 누적
    return running_loss / len(loader.dataset)  # 평균 손실 반환

# 하단 crop 함수 정의
def crop_bottom(img):
    img = img.resize((RESIZE_WIDTH, RESIZE_HEIGHT))  # 먼저 크기 조정
    return img.crop((0, 120, 320, 180))  # 하단 60픽셀 영역 crop

# 평가 함수 정의
def evaluate(model, loader, criterion, device):
    model.eval()  # 모델을 평가 모드로 전환
    total_loss = 0.0  # 손실 누적값
    preds, labels = [], []  # 예측값과 실제값 저장 리스트
    with torch.no_grad():  # 평가 시 그래프 생성 비활성화
        for imgs, angles in tqdm(loader, desc="Evaluating", leave=False):  # 배치별 평가 루프
            imgs, angles = imgs.to(device), angles.to(device)  # 데이터를 디바이스로 이동
            output = model(imgs).squeeze()  # 모델 예측 수행
            loss = criterion(output, angles)  # 손실 계산
            total_loss += loss.item() * imgs.size(0)  # 손실 누적
            preds.extend(output.cpu().numpy())  # 예측 결과 저장
            labels.extend(angles.cpu().numpy())  # 실제 각도 저장
    return total_loss / len(loader.dataset), preds, labels  # 평균 손실, 예측, 실제 반환


if __name__ == '__main__':
    train_transform = transforms.Compose([
        transforms.Lambda(crop_bottom),
        transforms.Resize((60, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    valid_transform = transforms.Compose([
        transforms.Lambda(crop_bottom),
        transforms.Resize((60, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    df = pd.read_csv(LABELS_CSV)
    n_samples = len(df)

    indices = torch.randperm(n_samples).tolist()
    train_size = int(0.8 * n_samples)
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:]

    train_set = SteeringDataset(
        LABELS_CSV,
        DATASET_DIR,
        transform=train_transform,
        use_random_flip=True,
        indices=train_indices
    )

    valid_set = SteeringDataset(
        LABELS_CSV,
        DATASET_DIR,
        transform=valid_transform,
        use_random_flip=False,
        indices=valid_indices
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 학습 디바이스 설정
    model = SteeringModel().to(device)  # 모델 초기화 및 디바이스 할당
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)  # Adam 옵티마이저 (weight decay 추가)
    criterion = nn.MSELoss()  # 손실 함수: 평균제곱오차

    # LR scheduler 설정 (validation loss 기준 ReduceLROnPlateau)
    scheduler = None
    if REDUCE_LR_ON_PLATEAU:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=REDUCE_LR_FACTOR, patience=REDUCE_LR_PATIENCE, min_lr=MIN_LR, verbose=True)

    save_dir = get_unique_train_folder()  # 고유 학습 결과 폴더 생성
    log_path = os.path.join(save_dir, 'log.csv')  # 로그 파일 경로 설정
    with open(log_path, 'w') as f:  # 로그 파일 초기화
        f.write('epoch,train_loss,valid_loss\n')  # 헤더 작성
    best_loss = float('inf')  # 초기 최적 손실값 무한대
    patience = EARLY_STOPPING_PATIENCE     
    no_improve_count = 0      

    # 학습 루프 시작
    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}")  # 현재 에폭 출력
        # train loop (clip grad을 전달하여 폭주 차단)
        train_loss = train(model, train_loader, optimizer, criterion, device, clip_grad=CLIP_GRAD_NORM)
        valid_loss, preds, labels = evaluate(model, valid_loader, criterion, device)  # 검증 손실 계산
        # scheduler: ReduceLROnPlateau 사용 시 validation loss로 스텝
        if scheduler is not None:
            scheduler.step(valid_loss)
        with open(log_path, 'a') as f:  # 로그 기록 추가
            f.write(f"{epoch},{train_loss:.6f},{valid_loss:.6f}\n")
        # 현재 learning rate 출력
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Train Loss: {train_loss:.4f} | valid Loss: {valid_loss:.4f} | lr: {current_lr:.2e}")

        # 현재 체크포인트 저장 후 이전 체크포인트 제거
        ckpt_path = os.path.join(save_dir, f"checkpoint_epoch{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)  # 현재 모델 가중치 저장
        if epoch > 1:  # 이전 체크포인트 존재 시
            prev_ckpt = os.path.join(save_dir, f"checkpoint_epoch{epoch - 1}.pth")
            if os.path.exists(prev_ckpt):
                os.remove(prev_ckpt)  # 이전 체크포인트 삭제

        if valid_loss < best_loss:  # 새로운 최적 모델인 경우
            best_loss = valid_loss              # 최적 손실 갱신
            no_improve_count = 0               # 개선됐으니 카운트 리셋
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))  # 최적 모델 저장
        else:
            no_improve_count += 1              # 개선 안 됨 → 카운트 증가
            print(f"no improvement count: {no_improve_count}/{patience}")

        torch.save(model.state_dict(), os.path.join(save_dir, "last_model.pth"))  # 마지막 모델 저장

        # 🔥 Early Stopping 체크
        if no_improve_count >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # 손실 곡선 플로팅
    log_df = pd.read_csv(log_path, index_col=False)   # CSV 읽기 (epoch이 인덱스로 잡히지 않도록 설정)
    log_df.columns = log_df.columns.str.strip()      # 열 이름의 앞뒤 공백 제거
    log_df = log_df.astype(float)                    # 모든 열을 실수형(float)으로 변환
    
    plt.figure(figsize=(10,5))                       # 플롯 크기 설정 (10x5 인치)
    
    plt.plot(log_df['epoch'].to_numpy(),             # x축: epoch
             log_df['train_loss'].to_numpy(),        # y축: 학습 손실
             label='Training Loss')                  # 라벨: Training Loss
    
    plt.plot(log_df['epoch'].to_numpy(),             # x축: epoch
             log_df['valid_loss'].to_numpy(),         # y축: 검증 손실
             label='Validation Loss', linestyle='--')# 라벨: Validation Loss (점선)
    
    plt.xlabel('Epoch')                              # x축 레이블
    plt.ylabel('Loss')                               # y축 레이블
    plt.title('Training Loss vs Validation Loss')    # 그래프 제목
    plt.legend()                                     # 범례 표시
    plt.grid()                                       # 그리드 표시
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))  # 결과 이미지를 파일로 저장
    plt.close()                                      # 플롯 닫기
