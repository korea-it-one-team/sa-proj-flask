from app.services.color_service import load_team_colors
from app.services.color_service import identify_uniform_color_per_person
from app.services.field_service import detect_green_field_low_res
from app.services.model_loader import model
from app.services.state_manager import processing_status, team_colors
import os
import uuid
from flask import send_file, jsonify
import logging
import cv2 #openCV
import numpy as np
import threading
import subprocess

# 재커밋용 주석
# 동영상 처리 시작 및 JSON 불러오기
def process_video(request):
    # 요청에서 동영상과 팀 정보 받음
    file = request.files['video']
    home_team = request.form.get('home_team')
    away_team = request.form.get('away_team')

    if not file or file.filename == '':
        return jsonify({"error": "No video file provided"}), 400

    if not home_team or not away_team:
        return jsonify({"error": "Home team or away team is missing"}), 400

    logging.info(f"Received home_team: {home_team}, away_team: {away_team}")

    # 파일 확장자 검증
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        return jsonify({"error": "Unsupported file format"}), 400

    # JSON에서 팀 색상 정보 불러오기
    if not load_team_colors(home_team, away_team):
        return jsonify({"error": "Failed to load team colors from JSON."}), 400

    # 동영상 파일 저장
    video_bytes = np.frombuffer(file.read(), np.uint8)
    video_filename = f"temp_video_{uuid.uuid4().hex}.mp4"
    video_path = os.path.join("uploads", video_filename)

    os.makedirs("uploads", exist_ok=True)  # 폴더 생성
    with open(video_path, 'wb') as f:
        f.write(video_bytes)

    # 동영상 처리 상태 초기화
    processing_status['status'] = "processing"
    processing_status['progress'] = 0

    # 동영상 처리 비동기 스레드에서 실행
    threading.Thread(target=update_status, args=(video_path,)).start()

    return jsonify({"message": "동영상 처리를 시작했습니다."})

# 다운로드 처리
def download_video():
    video_path = os.path.join("C:/work_oneteam/sa_proj_flask/app/static/video", "processed_video_h264.mp4")

    # 동영상 파일을 다운로드하여 클라이언트로 전송
    return send_file(video_path, as_attachment=True)

# 비디오 처리 상태 반환
def video_status():
    return jsonify(processing_status)

# 비디오 상태 업데이트
def update_status(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error("Error: Could not open video file.")
        processing_status['status'] = "error"
        return

    # 팀 색상 초기화 확인
    if team_colors["home"] is None or team_colors["away"] is None:
        logging.error("Error: Team colors not initialized correctly.")
        processing_status['status'] = "error"
        return

    save_path = os.path.join("C:/work_oneteam/sa_proj_flask/app/static/video", "processed_video.mp4")

    # 디렉터리 존재 확인 및 생성
    video_dir = os.path.dirname(save_path)
    if not os.path.exists(video_dir):
        os.makedirs(video_dir, exist_ok=True)  # 디렉터리 생성

    # 해상도 확인
    width = int(cap.get(3))
    height = int(cap.get(4))
    if width <= 0 or height <= 0:
        logging.error("Invalid video resolution: width or height is zero.")
        processing_status['status'] = "error"
        return

    # VideoWriter 초기화
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' 사용
    out = cv2.VideoWriter(save_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    # VideoWriter 열림 상태 확인
    if not out.isOpened():
        logging.error(f"Error: cv2.VideoWriter failed to open. Check codec or path: {save_path}")
        processing_status['status'] = "error"
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 필드 영역 감지
        field_contour = detect_green_field_low_res(frame)
        if field_contour is not None:
            cv2.drawContours(frame, [field_contour], -1, (0, 255, 255), 2)  # 필드 영역을 노란색으로 표시

        # 객체 탐지
        results = model(frame)
        for result in results:
            for box in result.boxes:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])

                # 공과 선수의 필드 내 포함 여부 확인
                if box.cls == 32 and cv2.pointPolygonTest(field_contour, ((bx1 + bx2) // 2, (by1 + by2) // 2), False) >= 0:
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)  # 공을 초록색 사각형으로 표시

                elif box.cls == 0 and cv2.pointPolygonTest(field_contour, ((bx1 + bx2) // 2, (by1 + by2) // 2), False) >= 0:
                    player_crop = frame[by1:by2, bx1:bx2]
                    team = identify_uniform_color_per_person(player_crop, team_colors["home"], team_colors["away"])

                    # 팀 판별 결과에 따른 박스 색상 결정
                    if team == "home_team":
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)  # 홈팀은 빨간색
                    elif team == "away_team":
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 0, 0), 2)  # 원정팀은 파란색
                    else:
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 0), 2)  # 인식되지 않은 선수는 검은색

                    logging.info(f"Player color detected: {team}")

        # 처리된 프레임 저장 및 진행률 업데이트
        try:
            out.write(frame)
            logging.info(f"Frame {processed_frames + 1} saved")
        except Exception as frame_err:
            logging.error(f"Error saving frame {processed_frames + 1}: {frame_err}")
            processing_status['status'] = "error"
            return

        processed_frames += 1
        processing_status['progress'] = int((processed_frames / total_frames) * 100)

    logging.info("Video processing completed, now saving...")

    # 비디오 리소스 해제
    cap.release()
    out.release()

    # 비디오 저장 상태 확인 및 최종 처리
    try:
        verify_and_finalize_save(save_path, video_path)
    except Exception as e:
        logging.error(f"Error during video save finalization: {e}")
        processing_status['status'] = "error"

def verify_and_finalize_save(save_path, video_path):
    """
    비디오 저장 상태를 확인하고 최종 처리를 수행하는 함수.
    """
    # 저장 상태 확인
    logging.info("Video processing completed, now verifying save...")
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        logging.info(f"Processed video saved successfully: {save_path}")
        processing_status['status'] = "completed"
    else:
        logging.error(f"Error: Processed video file not found or empty at {save_path}")
        processing_status['status'] = "error"
        return

    # 디렉터리 상태 로그 출력
    video_dir = os.path.dirname(save_path)
    try:
        directory_contents = os.listdir(video_dir)
        logging.info(f"Directory contents: {directory_contents}")
    except Exception as dir_err:
        logging.error(f"Error reading directory contents: {dir_err}")

    # 파일 크기 확인
    try:
        file_size = os.path.getsize(save_path)
        logging.info(f"Processed video file size: {file_size} bytes.")

        if file_size > 100 * 1024:  # 100KB 이상
            logging.info("File size check passed.")
        else:
            logging.warning("File size too small, but continuing with integrity check.")
    except Exception as size_err:
        logging.error(f"Error checking file size: {size_err}")
        processing_status['status'] = "error"
        return

    # H.264로 변환 (선택적으로 사용)
    h264_output_path = save_path.replace(".mp4", "_h264.mp4")
    try:
        convert_to_h264(save_path, h264_output_path)
        logging.info(f"Converted to H.264 successfully: {h264_output_path}")
    except Exception as e:
        logging.error(f"Error converting to H.264: {e}")
        processing_status['status'] = "error"
        return

    # 원본 비디오 파일 삭제
    if os.path.exists(video_path):
        try:
            os.remove(video_path)
            logging.info(f"원본 비디오 파일 {video_path} 삭제 완료.")
        except Exception as delete_err:
            logging.error(f"Error deleting original video file: {delete_err}")

def convert_to_h264(input_path, output_path):
    """FFmpeg를 사용하여 비디오를 H.264로 변환"""
    ffmpeg_path = r'C:\work_oneteam\sa_proj_flask\libs\ffmpeg\bin\ffmpeg.exe'  # FFmpeg 경로
    ffmpeg_cmd = [
        ffmpeg_path,
        '-i', input_path,
        '-c:v', 'libx264',  # H.264 코덱
        '-preset', 'fast',  # 인코딩 속도 설정
        '-crf', '23',       # 품질 설정 (낮을수록 고품질)
        output_path
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        logging.info(f"변환 성공: {output_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg 변환 실패: {e}")
        raise e