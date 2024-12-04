from flask import Flask
from flask_cors import CORS
from app.routes.health_routes import bp as health_bp
from app.routes.image_routes import bp as image_bp
from app.routes.video_routes import bp as video_bp
from app.routes.action_video_routes import bp as video_action_bp
from app.routes.study_routes import bp as study_bp
from app.utils.logger import setup_logging

def create_app():

    # Flask 애플리케이션 생성
    app = Flask(__name__)

    # 로깅 설정
    setup_logging()

    # CORS 설정 추가
    CORS(app, resources={r"/*": {"origins": "http://localhost:8088"}})  # Spring Boot의 출처만 허용

    # 블루프린트 등록
    app.register_blueprint(image_bp, url_prefix="/image")
    app.register_blueprint(video_bp, url_prefix="/video")
    app.register_blueprint(video_action_bp, url_prefix="/video_action")
    app.register_blueprint(study_bp, url_prefix="/study")
    app.register_blueprint(health_bp)  # url_prefix 제거

    return app