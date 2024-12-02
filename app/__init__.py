from flask import Flask
from flask_cors import CORS  # CORS를 추가
from app.routes.health_routes import bp as health_bp
from app.routes.image_routes import bp as image_bp
from app.routes.video_routes import bp as video_bp
from app.utils.logger import setup_logging
from config import setup_environment

def create_app():
    # 환경 설정 초기화
    setup_environment()

    # Flask 애플리케이션 생성
    app = Flask(__name__)

    # 로깅 설정
    setup_logging()

    # CORS 설정 추가
    CORS(app, resources={r"/*": {"origins": "http://localhost:8088"}})  # Spring Boot의 출처만 허용

    # 블루프린트 등록
    app.register_blueprint(image_bp, url_prefix="/image")
    app.register_blueprint(video_bp, url_prefix="/video")
    app.register_blueprint(health_bp, url_prefix="/health")

    return app