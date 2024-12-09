import logging
import matplotlib

def setup_logging():
    """로깅 설정"""
    # 파일 핸들러 생성 (기존 로그 파일을 덮어씁니다)
    file_handler = logging.FileHandler("flask_app.log", mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    # 로거 가져오기
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # 기존 핸들러 제거 (중복 방지)
    if logger.hasHandlers():
        logger.handlers.clear()

    # 새 핸들러 추가
    logger.addHandler(file_handler)

    # 콘솔 핸들러 추가 (선택 사항)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Matplotlib의 디버그 로그 숨기기
    matplotlib_logger = logging.getLogger('matplotlib')
    matplotlib_logger.setLevel(logging.WARNING)  # WARNING 이상의 메시지만 출력

    logging.info("로깅 설정이 완료되었습니다.")