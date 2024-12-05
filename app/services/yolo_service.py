from ultralytics import YOLO

# YOLO 모델 로드
yolo_model = YOLO('yolov8s.pt')

def detect_objects_with_yolo(frame):
    """
    YOLO를 사용해 객체 감지 후 SAMURAI 프롬프트 형식으로 변환.
    """
    results = yolo_model(frame)  # YOLO 추론
    prompts = {}  # SAMURAI 프롬프트 초기화
    obj_id = 0  # 객체 ID 시작값

    for result in results:
        for box in result.boxes:
            # 바운딩 박스 좌표 가져오기
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = box.cls.item()  # 객체 클래스 (0: person, 32: basketball)

            # SAMURAI 프롬프트로 변환
            if label == 0:  # 사람
                prompts[obj_id] = ((x1, y1, x2, y2), 0)
                obj_id += 1
            elif label == 32:  # 농구공
                prompts[obj_id] = ((x1, y1, x2, y2), 1)
                obj_id += 1

    return prompts