import requests
import torch

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
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 한글 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"  # 시스템에서 한글 폰트 경로 확인 후 설정
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams["font.family"] = font_prop.get_name()

# 동영상 처리 시작 및 JSON 불러오기
def process_video(request):
    # 요청에서 동영상과 팀 정보 받음
    file = request.files['video']
    home_team = request.form.get('home_team')
    away_team = request.form.get('away_team')
    article_id = request.form.get('article_id')  # 추가된 부분

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

    try:
        article_id = int(article_id)  # 정수 변환
    except ValueError:
        raise ValueError("Invalid article_id format. Expected an integer.")

    # 동영상 처리 상태 초기화
    processing_status[article_id] = {"status": "processing", "progress": 0}

    # 동영상 처리 비동기 스레드에서 실행
    threading.Thread(target=update_status, args=(video_path, article_id,)).start()

    return jsonify({"message": f"동영상 처리가 {article_id}에서 시작되었습니다."})

# 다운로드 처리
def download_file(article_id, file_type):
    """
    요청받은 article_id와 file_type에 해당하는 파일을 클라이언트로 전송.
    file_type은 'video' 또는 'summary'로 지정.
    """
    if file_type == "video":
        file_path = os.path.join(
            "C:/work_oneteam/sa_proj_flask/app/static/video",
            f"processed_video_{article_id}_h264.mp4"
        )
    elif file_type == "summary":
        file_path = os.path.join(
            "C:/work_oneteam/sa_proj_flask/app/static/video",
            f"processed_video_{article_id}_summary.png"
        )
    else:
        return jsonify({"error": "Invalid file_type parameter"}), 400

    # 파일 존재 여부 확인
    if not os.path.exists(file_path):
        return jsonify({"error": f"File not found for article_id {article_id}"}), 404

    # 파일 다운로드
    return send_file(file_path, as_attachment=True)

# 비디오 처리 상태 반환
def video_status(request):
    article_id = request.args.get("article_id")  # 요청에서 article_id 가져오기

    if not article_id:
        return jsonify({"error": "article_id is required"}), 400

    try:
        article_id = int(article_id)  # article_id를 정수로 변환
    except ValueError:
        return jsonify({"error": "Invalid article_id format"}), 400

    if article_id not in processing_status:
        return jsonify({"error": f"No status found for article_id {article_id}"}), 404

    return jsonify(processing_status.get(article_id))  # 요청된 article_id의 상태 반환

# 비디오 상태 업데이트
def update_status(video_path, article_id):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Error: Could not open video file for article {article_id}.")
        processing_status[article_id]["status"] = "error"
        return

    # 팀 색상 초기화 확인
    if team_colors["home"] is None or team_colors["away"] is None:
        logging.error("Error: Team colors not initialized correctly.")
        processing_status[article_id]["status"] = "error"
        return

    save_path = os.path.join(
        "C:/work_oneteam/sa_proj_flask/app/static/video",
        f"processed_video_{article_id}.mp4"  # 파일 이름에 article_id 포함
    )

    # 디렉터리 존재 확인 및 생성
    video_dir = os.path.dirname(save_path)
    if not os.path.exists(video_dir):
        os.makedirs(video_dir, exist_ok=True)  # 디렉터리 생성

    # 해상도 확인
    width = int(cap.get(3))
    height = int(cap.get(4))
    if width <= 0 or height <= 0:
        logging.error("Invalid video resolution: width or height is zero.")
        processing_status[article_id]["status"] = "error"
        return

    # VideoWriter 초기화
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' 사용
    out = cv2.VideoWriter(save_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    # VideoWriter 열림 상태 확인
    if not out.isOpened():
        logging.error(f"Error: cv2.VideoWriter failed to open. Check codec or path: {save_path}")
        processing_status[article_id]["status"] = "error"
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = 0

    # 데이터 저장용 딕셔너리 초기화
    frame_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 필드 영역 감지
        field_contour = detect_green_field_low_res(frame)
        if field_contour is not None:
            cv2.drawContours(frame, [field_contour], -1, (0, 255, 255), 2)  # 필드 영역을 노란색으로 표시

        # 프레임별 데이터 저장용 리스트
        current_frame_data = {
            "frame": processed_frames + 1,
            "home_team": [],
            "away_team": [],
            "unidentified": [],
            "ball": []
        }

        # 객체 탐지
        results = model(frame)
        for result in results:
            for box in result.boxes:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])

                # 공과 선수의 필드 내 포함 여부 확인
                if box.cls == 32 and cv2.pointPolygonTest(field_contour, ((bx1 + bx2) // 2, (by1 + by2) // 2), False) >= 0:
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)  # 공을 초록색 사각형으로 표시
                    current_frame_data["ball"].append((bx1, by1, bx2, by2))  # 공 데이터 추가
                    logging.info(f"Ball coordinates: {current_frame_data['ball']}")

                elif box.cls == 0 and cv2.pointPolygonTest(field_contour, ((bx1 + bx2) // 2, (by1 + by2) // 2), False) >= 0:
                    player_crop = frame[by1:by2, bx1:bx2]
                    team = identify_uniform_color_per_person(player_crop, team_colors["home"], team_colors["away"])

                    # 팀 판별 결과에 따른 박스 색상 결정
                    if team == "home_team":
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)  # 홈팀은 빨간색
                        current_frame_data["home_team"].append((bx1, by1, bx2, by2))
                        logging.info(f"Home team coordinates: {current_frame_data['home_team']}")
                    elif team == "away_team":
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 0, 0), 2)  # 원정팀은 파란색
                        current_frame_data["away_team"].append((bx1, by1, bx2, by2))
                        logging.info(f"Away team coordinates: {current_frame_data['away_team']}")
                    else:
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 0), 2)  # 인식되지 않은 선수는 검은색
                        current_frame_data["unidentified"].append((bx1, by1, bx2, by2))

                    logging.info(f"Player color detected: {team}")

        # 처리된 프레임 저장 및 진행률 업데이트
        try:
            out.write(frame)
            # current_frame_data를 frame_data 리스트에 추가
            frame_data.append(current_frame_data)
            logging.info(f"Frame {processed_frames + 1} saved")
        except Exception as frame_err:
            logging.error(f"Error saving frame {processed_frames + 1}: {frame_err}")
            processing_status[article_id]["status"] = "error"
            return

        processed_frames += 1
        processing_status[article_id]["progress"] = int((processed_frames / total_frames) * 100)

        if processed_frames % 10 == 0:
            torch.cuda.empty_cache()
            logging.info("cuda cache emptied...")

    # 비디오 리소스 해제
    cap.release()
    out.release()

    # 비디오 저장 상태 확인 및 최종 처리
    try:
        logging.info(f"Frame data: {frame_data[:5]}")  # 첫 5 프레임 데이터 확인
        visualize_summary_data(frame_data, video_resolution=(height, width), save_path=save_path)
        verify_and_finalize_save(save_path, video_path, article_id)
    except Exception as e:
        logging.error(f"Error during video save finalization: {e}")
        processing_status[article_id]["status"] = "error"

def visualize_summary_data(frame_data, video_resolution, save_path):
    logging.info(f"video_resolution : {video_resolution}")

    home_movements = []
    away_movements = []
    ball_movements = []

    home_heatmap = np.zeros(video_resolution, dtype=np.float32)
    away_heatmap = np.zeros(video_resolution, dtype=np.float32)
    ball_heatmap = np.zeros(video_resolution, dtype=np.float32)

    ball_ownership = []  # 공 소유 추적 정보 저장

    for frame_idx, frame in enumerate(frame_data):
        for bbox in frame["home_team"]:
            home_movements.append(((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2))
            home_heatmap[bbox[1]:bbox[3], bbox[0]:bbox[2]] += 1

        for bbox in frame["away_team"]:
            away_movements.append(((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2))
            away_heatmap[bbox[1]:bbox[3], bbox[0]:bbox[2]] += 1

        for bbox in frame["ball"]:
            ball_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            ball_movements.append(ball_center)
            ball_heatmap[bbox[1]:bbox[3], bbox[0]:bbox[2]] += 1

            # 공 소유 추적: 가장 가까운 팀을 추적
            closest_team = "none"
            closest_distance = float("inf")
            for team, team_bboxes in [("home", frame["home_team"]), ("away", frame["away_team"])]:
                for player_bbox in team_bboxes:
                    player_center = ((player_bbox[0] + player_bbox[2]) // 2, (player_bbox[1] + player_bbox[3]) // 2)
                    distance = np.linalg.norm(np.array(ball_center) - np.array(player_center))
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_team = team
            ball_ownership.append((frame_idx + 1, closest_team))

    # Heatmap Normalization
    home_heatmap /= home_heatmap.max() if home_heatmap.max() > 0 else 1
    away_heatmap /= away_heatmap.max() if away_heatmap.max() > 0 else 1
    ball_heatmap /= ball_heatmap.max() if ball_heatmap.max() > 0 else 1

    # Plotting
    fig, ax = plt.subplots(2, 2, figsize=(16, 12))

    # 축구 필드 배경 이미지 로드
    try:
        field_img = plt.imread("soccer_field.png")  # 축구 필드 배경 이미지
    except FileNotFoundError:
        field_img = np.zeros(video_resolution)  # 배경 이미지가 없을 경우 기본 검은 화면

    # 1. 이동 궤적 시각화
    ax[0, 0].imshow(field_img, extent=[0, video_resolution[1], 0, video_resolution[0]])
    ax[0, 0].scatter(
        [pos[0] for pos in home_movements],
        [pos[1] for pos in home_movements],
        c="red", label="홈팀", alpha=0.6, s=10
    )
    ax[0, 0].scatter(
        [pos[0] for pos in away_movements],
        [pos[1] for pos in away_movements],
        c="blue", label="원정팀", alpha=0.6, s=10
    )
    ax[0, 0].scatter(
        [pos[0] for pos in ball_movements],
        [pos[1] for pos in ball_movements],
        c="green", label="공", alpha=0.8, s=20
    )
    ax[0, 0].set_title("이동 궤적 시각화")
    ax[0, 0].legend()
    ax[0, 0].text(0.5, -0.1, "각 팀과 공의 이동 경로를 시각화한 결과", fontsize=10, ha='center', transform=ax[0, 0].transAxes)

    # 2. 홈팀 Heatmap
    ax[0, 1].imshow(home_heatmap, cmap="Reds", interpolation="nearest")
    ax[0, 1].set_title("홈팀 활동 히트맵")
    ax[0, 1].text(0.5, -0.1, "홈팀 선수들의 활동 밀집도를 보여줍니다.", fontsize=10, ha='center', transform=ax[0, 1].transAxes)

    # 3. 어웨이팀 Heatmap
    ax[1, 0].imshow(away_heatmap, cmap="Blues", interpolation="nearest")
    ax[1, 0].set_title("원정팀 활동 히트맵")
    ax[1, 0].text(0.5, -0.1, "원정팀 선수들의 활동 밀집도를 보여줍니다.", fontsize=10, ha='center', transform=ax[1, 0].transAxes)

    # 4. 공 Heatmap
    ax[1, 1].imshow(ball_heatmap, cmap="Greens", interpolation="nearest")
    ax[1, 1].set_title("공 활동 히트맵")
    ax[1, 1].text(0.5, -0.1, "공의 이동 및 머문 위치를 시각화한 결과", fontsize=10, ha='center', transform=ax[1, 1].transAxes)

    # 공 소유자 정보 추가
    ball_ownership_summary = "\n".join([f"프레임 {frame}: {owner}" for frame, owner in ball_ownership[:5]])
    fig.text(0.5, 0.02, f"공 소유 요약 (최초 5 프레임):\n{ball_ownership_summary}", fontsize=12, ha='center')

    # 저장
    output_path = save_path.replace(".mp4", "_summary.png")  # .mp4 대신 .png로 저장
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # 공 소유자 정보를 포함하도록 여백 조정
    plt.savefig(output_path)
    plt.close(fig)

    logging.info(f"Summary image saved at: {output_path}")

def verify_and_finalize_save(save_path, video_path, article_id):
    """
    비디오 저장 상태를 확인하고 최종 처리를 수행하는 함수.
    """
    # 저장 상태 확인
    logging.info("Video processing completed, now verifying save...")
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        logging.info(f"Processed video saved successfully: {save_path}")
        processing_status[article_id]["status"] = "converting"
    else:
        logging.error(f"Error: Processed video file not found or empty at {save_path}")
        processing_status[article_id]["status"] = "error"
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
        processing_status[article_id]["status"] = "error"
        return

    # H.264로 변환 (선택적으로 사용)
    h264_output_path = save_path.replace(".mp4", "_h264.mp4")
    try:
        convert_to_h264(save_path, h264_output_path)
        logging.info(f"Converted to H.264 successfully: {h264_output_path}")
        processing_status[article_id]["status"] = "completed"
        notify_springboot(article_id, "completed")
    except Exception as e:
        logging.error(f"Error converting to H.264: {e}")
        processing_status[article_id]["status"] = "error"
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

def notify_springboot(article_id, status):
    """
    SpringBoot에 처리 상태를 알림.
    """
    springboot_url = "http://localhost:8088/openCV/analysisCompleted"
    payload = {"article_id": str(article_id), "status": status}  # article_id를 str로 변환

    try:
        response = requests.post(springboot_url, json=payload)
        if response.status_code == 200:
            logging.info(f"Successfully notified SpringBoot for article {article_id}: {status}")
        else:
            logging.error(f"Failed to notify SpringBoot. Response: {response.status_code}, {response.text}")
    except Exception as e:
        logging.error(f"Error notifying SpringBoot: {e}")