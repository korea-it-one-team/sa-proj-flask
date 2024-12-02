import cv2 #openCV
import numpy as np

# 필드 영역을 저해상도로 감지하고, 객체는 원본 해상도에서 탐지
def detect_green_field_low_res(frame):
    # 해상도 축소 (필드 감지만 저해상도로 처리)
    resized_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    hsv_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)

    # 초록색 범위 설정 (HSV 기준)
    green_lower = np.array([30, 40, 40], dtype=np.uint8)
    green_upper = np.array([90, 255, 255], dtype=np.uint8)

    # 초록색 바닥 필터링
    mask = cv2.inRange(hsv_frame, green_lower, green_upper)

    # 컨투어 찾기 (필드 영역 탐색)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 가장 큰 컨투어를 필드로 간주
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # 좌표를 원래 해상도로 다시 변환
        return largest_contour * 2  # 해상도 축소 비율을 고려하여 원래 크기로 복원

    return None  # 필드를 감지하지 못한 경우