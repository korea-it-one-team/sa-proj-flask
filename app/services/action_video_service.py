# app/services/action_video_service.py

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
import os
import cv2



def extract_frames(video_path, output_folder, frame_rate=8):
    """
    동영상에서 1초마다 8프레임을 추출합니다.
    :param video_path: 동영상 파일 경로
    :param output_folder: 프레임 저장 폴더
    :param frame_rate: 초당 추출할 프레임 수 (기본 8이면 초당 8프레임)
    """
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    # 비디오 정보 얻기
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 전체 프레임 수
    fps = cap.get(cv2.CAP_PROP_FPS)  # 동영상의 실제 FPS

    print(f"Total frames: {total_frames}, FPS: {fps}")

    frame_idx = 0  # 추출할 프레임의 인덱스

    # 초당 8프레임씩 추출하려면 `frame_rate`만큼 건너뛰며 프레임을 추출
    interval = int(fps // frame_rate)  # 간격을 FPS / 8로 설정하여 초당 8프레임 추출

    print(f"Interval between frames: {interval} frames")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:  # 초당 8프레임씩 추출
            frame_path = os.path.join(output_folder, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(frame_path, frame)  # 프레임 저장

        frame_idx += 1

    cap.release()
    print(f"Frames extracted to {output_folder}")


def load_model(model_path, device):
    """
    저장된 모델을 로드합니다.
    :param model_path: 모델 저장 경로
    :param device: 모델이 로드될 디바이스 (CPU/GPU)
    """
    model = torch.load(model_path)  # 모델 전체 로드
    model.to(device)
    model.eval()  # 평가 모드로 전환
    return model


def predict_frame(image_path, model, device, classes, previous_prediction=None, event_duration=5, frame_idx=None, last_event_time=None, interval_threshold=5, continuous_events=None, confidence_threshold=0.5):
    """
    단일 프레임 예측.
    :param image_path: 이미지 파일 경로
    :param model: 학습된 모델
    :param device: 모델이 로드된 디바이스
    :param classes: 클래스 리스트
    :param confidence_threshold: 확률 기준 (기본값 70%)
    """
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # 모델이 요구하는 입력 크기로 수정
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)  # Softmax를 통해 각 클래스의 확률을 계산
        max_prob, predicted = torch.max(probabilities, 1)  # 가장 높은 확률과 해당 클래스 인덱스

    predicted_event = classes[predicted.item()]

    # 확률이 70% 미만인 경우 예측을 무시
    if max_prob.item() < confidence_threshold:
        return None

    # 중복 예측 처리
    if predicted_event == previous_prediction and (frame_idx - last_event_time) < event_duration:
        return None

    if (frame_idx - last_event_time) < interval_threshold and predicted_event not in continuous_events:
        return None

    return predicted_event


def process_video(video_path, model_path, classes_path, output_folder, predictions_file, frame_rate=1, event_duration=5, interval_threshold=5, continuous_events=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델과 클래스 로드
    with open(classes_path, "r") as f:
        class_list = f.read().splitlines()

    model = load_model(model_path, device)  # 모델 로드

    # 프레임 추출
    extract_frames(video_path, output_folder, frame_rate)

    predictions = []
    previous_prediction = None
    last_event_time = -interval_threshold

    for frame_idx, frame in enumerate(sorted(os.listdir(output_folder), key=lambda x: int(x.split('_')[1].split('.')[0]))):
        frame_path = os.path.join(output_folder, frame)

        # 예측 수행
        prediction = predict_frame(frame_path, model, device, class_list, previous_prediction=previous_prediction,
                                   event_duration=event_duration, frame_idx=frame_idx,
                                   last_event_time=last_event_time, interval_threshold=interval_threshold,
                                   continuous_events=continuous_events)

        if prediction is not None:
            predictions.append({
                "frame": frame,
                "prediction": prediction
            })

            previous_prediction = prediction
            last_event_time = frame_idx

        print(f"Frame {frame} => Prediction: {prediction if prediction else 'None'}")

    # 결과 저장
    with open(predictions_file, "w") as f:
        json.dump(predictions, f, indent=4)  # 보기 좋게 결과 저장
    print(f"Predictions saved to {predictions_file}")
