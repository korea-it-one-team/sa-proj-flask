import os
import torch
from flask import Flask
from flask_cors import CORS
from app.routes.health_routes import bp as health_bp
from app.routes.image_routes import bp as image_bp
from app.routes.video_routes import bp as video_bp
from app.utils.logger import setup_logging

def create_app():

    # Flask 애플리케이션 생성
    app = Flask(__name__)

    os.environ["HYDRA_FULL_ERROR"] = "1"
    print("HYDRA_FULL_ERROR:", os.getenv("HYDRA_FULL_ERROR"))

    # 로깅 설정
    setup_logging()

    # CORS 설정 추가
    CORS(app, resources={r"/*": {"origins": "http://localhost:8088"}})  # Spring Boot의 출처만 허용

    # GPU가 사용 가능한지 확인
    print(f"Is GPU available? {torch.cuda.is_available()}")

    # 현재 사용 중인 GPU 이름 확인
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # GPU 메모리 사용량 출력
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Reserved memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

    # 블루프린트 등록
    app.register_blueprint(image_bp, url_prefix="/image")
    app.register_blueprint(video_bp, url_prefix="/video")
    app.register_blueprint(health_bp)  # url_prefix 제거

    return app