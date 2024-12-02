import os

class Config:
    # DLL 파일 경로 설정
    OPENH264_LIBRARY = r'C:\work_oneteam\one-team-SA-proj\libs\openh264-1.8.0-win64.dll'

def setup_environment():
    # 환경 변수 설정
    os.environ['OPENH264_LIBRARY'] = Config.OPENH264_LIBRARY