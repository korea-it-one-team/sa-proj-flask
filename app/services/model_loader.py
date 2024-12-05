import os
from sam2.build_sam import build_sam2_video_predictor

# 경로 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "samurai", "sam2", "checkpoints")
CONFIG_DIR = os.path.join(BASE_DIR, "samurai", "sam2", "sam2", "configs", "sam2.1")

# 체크포인트 및 설정 파일 경로
CHECKPOINT = os.path.join(CHECKPOINT_DIR, "sam2.1_hiera_base_plus.pt")
CONFIG_FILE = os.path.join(CONFIG_DIR, "sam2.1_hiera_b+.yaml")

def initialize_model(device="cuda:0"):
    """
    SAMURAI 모델을 초기화합니다.
    """
    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(f"Checkpoint file not found: {CHECKPOINT}")
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"Config file not found: {CONFIG_FILE}")

    print(f"Initializing model with checkpoint: {CHECKPOINT}")
    model = build_sam2_video_predictor(
        CONFIG_FILE,
        CHECKPOINT,
        device=device,
        async_loading_frames=False,
    )
    print("Model initialized successfully!")
    return model

# 전역 모델 초기화
try:
    model = initialize_model()
except Exception as e:
    print(f"Error initializing SAMURAI model: {e}")
    model = None
