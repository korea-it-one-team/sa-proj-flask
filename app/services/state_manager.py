import torch
import logging

# 글로벌 상태 변수
processing_status = {
    "status": "idle",  # 처리 상태: idle, processing, completed
    "progress": 0      # 진행률: 0 ~ 100
}

# 전역 변수로 팀 색상 저장
team_colors = {"home": None, "away": None}

# 초기 객체 정보 저장
initial_objects = {}  # {ID: "basketball" or "person"}

def set_initial_objects(inference_state, prompts, model):
    """
    Initialize objects in the inference state based on YOLO prompts.
    """
    logging.info(f"Received prompts for initialization: {prompts}")

    obj_ids = []

    for obj_id, (bbox, class_id) in prompts.items():
        logging.info(f"Processing obj_id={obj_id} with bbox={bbox}")
        # Add initial box to the state using add_new_points_or_box
        frame_idx, object_ids, masks = model.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,  # 첫 번째 프레임
            obj_id=obj_id,  # 객체 ID
            box=bbox,  # YOLO에서 가져온 bbox
            clear_old_points=True,
            normalize_coords=True,
        )
        logging.info(f"Added object {obj_id} with box {bbox} to state: frame_idx={frame_idx}, object_ids={object_ids}")
        obj_ids.append(obj_id)

    inference_state["obj_ids"] = obj_ids

    logging.info(f"Final object IDs in inference_state: {inference_state['obj_ids']}")
