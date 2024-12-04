import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torch.amp import GradScaler, autocast
from sklearn.metrics import precision_score, recall_score
from torchvision.datasets.folder import default_loader
from app.models.tsn_model import TSNDataset  # 모델 데이터셋이 정의된 곳 (가정)

def train_model(train_list_path, classes_file, num_epochs=10, batch_size=16, learning_rate=0.001, model_save_path="tsn_model.pth"):
    """
    TSN 모델 학습 로직을 처리하는 함수
    """
    # 이미지 전처리 (크기 조정, 텐서 변환, 정규화)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # 이미지 크기 조정
        transforms.ToTensor(),  # 이미지를 텐서로 변환
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 이미지 정규화
    ])

    # 학습 데이터셋 준비
    train_dataset = TSNDataset(train_list_path, classes_file, transform=transform)

    # 클래스 수 확인
    n_classes = len(set(train_dataset.labels))
    print(f"Number of classes: {n_classes}")

    # ResNet50 모델 로드 및 마지막 Fully Connected Layer 수정
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, n_classes)

    # 손실 함수 및 옵티마이저 정의
    criterion = torch.nn.CrossEntropyLoss()  # CrossEntropyLoss (다중 클래스 분류)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam 옵티마이저

    # Mixed Precision 학습을 위한 GradScaler 초기화
    scaler = GradScaler()

    # 학습을 위한 장치 설정 (GPU가 있으면 GPU 사용, 없으면 CPU 사용)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 에폭별 학습 진행
    for epoch in range(num_epochs):
        model.train()  # 모델을 학습 모드로 설정
        running_loss = 0.0  # 에폭 당 손실을 기록할 변수
        all_labels = []  # 실제 레이블을 저장할 리스트
        all_predictions = []  # 예측 결과를 저장할 리스트

        # 데이터 로더를 통한 배치 처리
        for batch in DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True):
            if batch is None or any(item is None for item in batch):
                print(f"Skipping invalid batch: {batch}")
                continue

            images, labels = batch
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()  # 기울기 초기화

            # Mixed Precision Forward
            with autocast(device_type="cuda"):  # 자동 혼합 정밀도(AMP) 활성화
                outputs = model(images)  # 모델 예측
                loss = criterion(outputs, labels)  # 손실 계산

            # 손실의 기울기 계산 및 역전파
            scaler.scale(loss).backward()  # 기울기 계산
            scaler.step(optimizer)  # 옵티마이저 업데이트
            scaler.update()  # 스케일러 업데이트

            running_loss += loss.item()  # 총 손실에 현재 손실 더하기

            # 예측값과 실제값 비교
            _, preds = torch.max(outputs, 1)  # 가장 높은 확률의 클래스 예측
            all_labels.extend(labels.cpu().numpy())  # 실제 레이블 추가
            all_predictions.extend(preds.cpu().numpy())  # 예측된 레이블 추가

        # 에폭별 성능 지표 계산
        if len(all_labels) > 0:
            accuracy = sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_labels)  # 정확도 계산
            precision = precision_score(all_labels, all_predictions, average="macro", zero_division=0)  # 정밀도 계산
            recall = recall_score(all_labels, all_predictions, average="macro", zero_division=0)  # 재현율 계산

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_dataset):.4f}, "
                  f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        else:
            print(f"Epoch {epoch + 1}/{num_epochs}, No valid predictions made.")

    # 학습된 모델 저장
    torch.save(model, model_save_path)
    print(f"Model saved to {model_save_path}")

    return model_save_path  # 학습된 모델 경로 반환
