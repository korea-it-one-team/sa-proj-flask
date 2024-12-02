from app.services.color_service import load_team_colors
from app.services.color_service import identify_uniform_color_per_person
from app.services.field_service import detect_green_field_low_res
from app.services.model_loader import model
from app.services.state_manager import processing_status, team_colors

import os
import time

from flask import Flask, request, send_file, jsonify
import logging
import cv2 #openCV
import numpy as np
import threading

# 동영상 처리 시작 및 JSON 불러오기
def process_video():
    # 요청에서 동영상과 팀 정보 받음
    file = request.files['video']
    home_team = request.form.get('home_team')
    away_team = request.form.get('away_team')
    logging.info(f"Received home_team: {home_team}, away_team: {away_team}")

    # JSON에서 팀 색상 정보 불러오기
    if not load_team_colors(home_team, away_team):
        return jsonify({"error": "Failed to load team colors from JSON."}), 400

    # 동영상 파일 저장
    video_bytes = np.frombuffer(file.read(), np.uint8)
    video_path = "temp_video.mp4"
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
    # Java 프로젝트의 static 폴더에 저장된 동영상 경로로 수정
    video_path = os.path.join("C:/work_oneteam/sa_proj_flask/app/static/video", "processed_video.mp4")

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
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(save_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

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
        out.write(frame)
        logging.info(f"Frame {processed_frames + 1} saved")
        processed_frames += 1
        processing_status['progress'] = int((processed_frames / total_frames) * 100)

    logging.info("Video processing completed, now saving...")
    cap.release()
    out.release()

    # 상태를 "saving"으로 변경
    processing_status['status'] = "saving"

    # 동영상 파일이 정상적으로 저장되었는지 확인하는 추가 검증
    time.sleep(1)  # 파일 시스템에서 저장 완료까지의 잠깐의 딜레이

    # 파일 크기 확인 (예시: 최소 파일 크기를 1MB 이상으로 설정)
    if os.path.exists(save_path) and os.path.getsize(save_path) > 1 * 1024 * 1024:  # 1MB 이상
        print("File size check passed.")
    else:
        print("File size too small, video may be corrupted.")
        processing_status['status'] = "error"
        return

    # 동영상 무결성 확인: OpenCV로 다시 열어 프레임을 확인
    check_cap = cv2.VideoCapture(save_path)
    if check_cap.isOpened():
        ret, _ = check_cap.read()
        check_cap.release()
        if ret:
            print("Video integrity check passed.")
            processing_status['status'] = "completed"
        else:
            print("Error: Unable to read video frames, video may be corrupted.")
            processing_status['status'] = "error"
    else:
        print("Error: Unable to open video file, video may be corrupted.")
        processing_status['status'] = "error"

    # 원본 비디오 파일 삭제
    if os.path.exists(video_path):
        os.remove(video_path)
        print(f"원본 비디오 파일 {video_path} 삭제 완료.")