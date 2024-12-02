import os
import json

def read_json_file(file_path):
    """JSON 파일 읽기"""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

def write_json_file(file_path, data):
    """JSON 파일 쓰기"""
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def ensure_directory_exists(dir_path):
    """디렉토리 존재 확인 및 생성"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)