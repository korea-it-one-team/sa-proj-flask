from flask import Blueprint, request, jsonify, send_file
from app.services.action_video_service import process_video

bp = Blueprint("video_action", __name__)

@bp.route('/video_action', methods=['POST'])
def study_video_route():
    video_file = request.files.get('video')
    model_file = request.files.get('model')
    classes_file = request.files.get('classes')

    # 파일이 하나라도 누락되면 에러 응답
    if not video_file or not model_file or not classes_file:
        return jsonify({"error": "Missing required files"}), 400

    # 파일을 서버에 저장
    video_path = "uploads/video.mp4"
    model_path = "uploads/model.pth"
    classes_path = "uploads/classes.txt"
    video_file.save(video_path)
    model_file.save(model_path)
    classes_file.save(classes_path)

    output_folder = "uploads/frames"
    predictions_file = "uploads/predictions.json"

    # 서비스 함수 호출
    frame_rate = 8
    continuous_events = ["HIGH PASS", "PASS", "CROSS", "SHOT", "DRIBBLE"]
    process_video(video_path, model_path, classes_path, output_folder, predictions_file, frame_rate, event_duration=5, interval_threshold=5, continuous_events=continuous_events)

    # 예측 결과를 반환
    return jsonify({"message": "Video processed and predictions saved", "predictions_file": predictions_file})


