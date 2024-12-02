from flask import Blueprint, request
import os

bp = Blueprint("health", __name__)  # url_prefix 제거

@bp.route('/health', methods=['GET'])
def health_check():
    return "Flask 서버가 실행 중입니다.", 200

@bp.route('/shutdown', methods=['GET'])
def shutdown():
    shutdown_server = request.environ.get('werkzeug.server.shutdown')
    if shutdown_server is None:
        print("Werkzeug 서버가 아니므로 강제 종료를 시도합니다.")
        os._exit(0)
    else:
        shutdown_server()
        return 'Flask 서버가 종료되었습니다.'