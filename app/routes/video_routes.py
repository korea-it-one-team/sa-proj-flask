from flask import Blueprint, request, jsonify, send_file
from app.services.video_service import process_video, download_video, video_status

bp = Blueprint("video", __name__)

@bp.route('/process_video', methods=['POST'])
def process_video_route():
    return process_video(request)

@bp.route('/download_video', methods=['GET'])
def download_video_route():
    return download_video()

@bp.route('/video-status', methods=['GET'])
def video_status_route():
    return video_status()