from app import create_app

# 환경 초기화는 Flask 앱 생성 전에 수행
from config import setup_environment

setup_environment()

# Flask 애플리케이션 생성
app = create_app()

# 자동 재시작 방지
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)