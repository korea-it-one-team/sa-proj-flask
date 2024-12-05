# 글로벌 상태 변수
processing_status = {
    "status": "idle",  # 처리 상태: idle, processing, completed
    "progress": 0      # 진행률: 0 ~ 100
}

# 전역 변수로 팀 색상 저장
team_colors = {"home": None, "away": None}

# 초기 객체 정보 저장
initial_objects = {}  # {ID: "basketball" or "person"}

def set_initial_objects(state, prompts):
    """
    초기 프레임에서 객체의 ID와 유형을 설정합니다.
    """
    for obj_id, (bbox, label) in prompts.items():
        # 'label'이 0: person, 1: basketball (예제)
        if label == 1:
            initial_objects[obj_id] = "basketball"
        else:
            initial_objects[obj_id] = "person"