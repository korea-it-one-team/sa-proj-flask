import logging

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        filename="flask_app.log",
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("로깅 설정이 완료되었습니다.")