from flask import Blueprint, request, jsonify, send_file
from app.services.study_service import train_model

bp = Blueprint("study", __name__)

@bp.route('/video_study', methods=['POST'])
def study_video_route():
    """
    TSN 모델 학습 요청을 처리하는 라우트
    """
    # 클라이언트로부터 전달받은 파라미터 값들 (기본값 설정)
    train_list_path = request.form.get("train_list", "C:/work_oneteam/one-team-SA-proj/TSN_preprocessed/train_list.txt")
    classes_file = request.form.get("classes_file", "C:/work_oneteam/one-team-SA-proj/TSN_preprocessed/classes.txt")
    num_epochs = int(request.form.get("num_epochs", 10))
    batch_size = int(request.form.get("batch_size", 16))
    learning_rate = float(request.form.get("learning_rate", 0.001))
    model_save_path = request.form.get("model_save_path", "tsn_model.pth")

    try:
        # 모델 학습 함수 호출
        model_path = train_model(train_list_path, classes_file, num_epochs, batch_size, learning_rate, model_save_path)
        # 학습 완료 후 모델 경로를 반환
        return jsonify({"message": "Model training complete", "model_path": model_path}), 200
    except Exception as e:
        # 예외 발생 시 에러 메시지 반환
        return jsonify({"error": str(e)}), 500
