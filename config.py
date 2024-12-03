import os
import cv2  # OpenCV import

def setup_environment():
    # OpenCV 테스트
    try:
        print(f"OpenCV 버전: {cv2.__version__}")

        # Motion JPEG (MJPG) 코덱 설정
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        print(f"FourCC 값: {fourcc}")

        test_path = "test_output.avi"
        out = cv2.VideoWriter(test_path, fourcc, 20.0, (640, 480))

        if out.isOpened():
            print(f"VideoWriter initialized successfully. Test file: {test_path}")
            out.release()
            if os.path.exists(test_path):
                os.remove(test_path)
                print("Test file removed successfully.")
        else:
            print("Error: Failed to initialize VideoWriter. Check codec or path.")
    except Exception as e:
        print(f"OpenCV 초기화 실패: {e}")