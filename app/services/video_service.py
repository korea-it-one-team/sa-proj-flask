import torch
from zmq.utils.garbage import gc

from app.services.color_service import load_team_colors
from app.services.color_service import identify_uniform_color_per_person
from app.services.field_service import detect_green_field_low_res
from app.services.model_loader import model
from app.services.state_manager import processing_status, team_colors, set_initial_objects, initial_objects
from app.services.yolo_service import detect_objects_with_yolo
import os
import uuid
from flask import send_file, jsonify
import logging
import cv2 #openCV
import numpy as np
import threading
import subprocess

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
    threading.Thread(target=update_status, args=(video_path, model)).start()

    return jsonify({"message": "동영상 처리를 시작했습니다."})

# 비디오 처리 상태 반환
def video_status():
    return jsonify(processing_status)

# 동영상 처리 로직
def update_status(video_path, model):
    try:
        logging.info("Starting video processing...")
        # save_path 디렉터리 존재 확인 및 생성
        save_path = os.path.join("C:/work_oneteam/sa_proj_flask/app/static/video", "processed_video.mp4")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 1. 초기화
        logging.info("Initializing video processing...")
        cap, out, state = initialize_video_processing(video_path, model, save_path)

        # 2. 프레임 처리
        logging.info("Processing video frames...")
        process_video_frames(cap, out, state)

        # 3. 처리 후 저장
        logging.info("Finalizing video processing...")
        finalize_video_processing(cap, out, save_path, video_path)
    except Exception as e:
        logging.error(f"Error during video processing: {e}")
        processing_status['status'] = "error"
        raise e

def initialize_video_processing(video_path, model, save_path):
    # cap open
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        processing_status['status'] = "error"
        raise ValueError("Could not open video file.")

    # 팀 색상 초기화 확인
    if team_colors["home"] is None or team_colors["away"] is None:
        logging.error("Error: Team colors not initialized correctly.")
        processing_status['status'] = "error"
        return

    # 해상도 확인
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        processing_status['status'] = "error"
        raise ValueError("Invalid video resolution: width or height is zero.")

    # VideoWriter 초기화
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, 20.0, (width, height))

    # VideoWriter 열림 상태 확인
    if not out.isOpened():
        raise ValueError(f"Could not initialize VideoWriter for path: {save_path}")

    # 첫 프레임에서 프롬프트 생성
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Failed to read the first frame from the video.")

    # YOLO로 프롬프트 생성
    prompts = detect_objects_with_yolo(frame)

    video_capture = cv2.VideoCapture(video_path)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_capture.release()

    print(f"Frame dimensions: {frame_width}x{frame_height}")
    if frame_width > 640 or frame_height > 480:
        print("Warning: Frame size is large. Consider resizing.")

    # SAMURAI 상태 초기화
    state = model.init_state(video_path, offload_video_to_cpu=True, async_loading_frames=False)
    set_initial_objects(state, prompts)

    print("Allocated memory:", torch.cuda.memory_allocated() / 1024 ** 3, "GB")
    print("Reserved memory:", torch.cuda.memory_reserved() / 1024 ** 3, "GB")
    return cap, out, state

def process_video_frames(cap, out, state):
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 크기 축소
        frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

        # 필드 감지
        field_contour = detect_green_field_low_res(frame)
        if field_contour is not None:
            cv2.drawContours(frame, [field_contour], -1, (0, 255, 255), 2)

        try:
            _, object_ids, masks = model.propagate_in_video(state)
            for obj_id, mask in zip(object_ids, masks):
                # 마스크 및 바운딩 박스 계산
                mask = mask[0].cpu().numpy()
                mask = mask > 0.0
                non_zero_indices = np.argwhere(mask)
                if len(non_zero_indices) > 0:
                    y_min, x_min = non_zero_indices.min(axis=0).tolist()
                    y_max, x_max = non_zero_indices.max(axis=0).tolist()
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

                    center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2
                    if cv2.pointPolygonTest(field_contour, (center_x, center_y), False) >= 0:
                        if is_ball(obj_id):
                            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
                        else:
                            player_crop = frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
                            team = identify_uniform_color_per_person(player_crop, team_colors["home"], team_colors["away"])
                            # 팀별 박스 색상
                            if team == "home_team":
                                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 255), 2)  # 빨간색
                            elif team == "away_team":
                                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 2)  # 파란색
                            else:
                                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 0), 2)  # 검은색
        except Exception as e:
            logging.error(f"Error during SAMURAI tracking: {e}")
            processing_status['status'] = "error"
            return

        # 처리된 프레임 저장 및 진행률 업데이트
        try:
            out.write(frame)
            logging.info(f"Frame {processed_frames + 1} saved")
        except Exception as frame_err:
            logging.error(f"Error saving frame {processed_frames + 1}: {frame_err}")
            processing_status['status'] = "error"
            return

        # 주기적으로 GPU 캐시 정리
        if processed_frames % 10 == 0:  # 10 프레임마다 실행
            torch.cuda.empty_cache()
            gc.collect()

        processed_frames += 1
        processing_status['progress'] = int((processed_frames / total_frames) * 100)

def finalize_video_processing(cap, out, save_path, video_path):
    cap.release()
    out.release()
    verify_and_finalize_save(save_path, video_path)

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

# 다운로드 처리
def download_video():
    # SAMURAI 처리된 동영상 경로
    processed_video_dir = "C:/work_oneteam/sa_proj_flask/app/static/video"
    processed_video_filename = "processed_video_h264.mp4"
    video_path = os.path.join(processed_video_dir, processed_video_filename)

    # 동영상 파일을 다운로드하여 클라이언트로 전송
    if not os.path.exists(video_path):
        return jsonify({"error": "Processed video file not found."}), 404

    return send_file(video_path, as_attachment=True)

def is_ball(obj_id):
    """
    객체 ID를 기반으로 해당 객체가 공인지 여부를 반환합니다.
    """
    # initial_objects에서 ID 조회, 기본값은 "unknown"
    object_type = initial_objects.get(obj_id, "unknown")
    return object_type == "basketball"