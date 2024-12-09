from flask import Blueprint, request, jsonify, send_file
from app.services.video_service import process_video, download_file, video_status

bp = Blueprint("video", __name__)

@bp.route('/process_video', methods=['POST'])
def process_video_route():
    return process_video(request)

@bp.route('/download_video', methods=['GET'])
def download_video_route():
    """
    Flask에서 SpringBoot로 요청받은 article_id에 따라 동영상을 다운로드합니다.
    """
    article_id = request.args.get("article_id")  # SpringBoot 요청에서 article_id 가져오기
    file_type = request.args.get("file_type")

    if not article_id:
        return jsonify({"error": "article_id is required"}), 400

    try:
        article_id = int(article_id)  # article_id를 int로 변환
    except ValueError:
        return jsonify({"error": "article_id must be an integer"}), 400

    return download_file(article_id, file_type)

@bp.route('/video-status', methods=['GET'])
def video_status_route():
    return video_status(request)