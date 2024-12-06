import torch
import logging

from app.services.model_loader import model

# 글로벌 상태 변수
processing_status = {
    "status": "idle",  # 처리 상태: idle, processing, completed
    "progress": 0      # 진행률: 0 ~ 100
}

# 전역 변수로 팀 색상 저장
team_colors = {"home": None, "away": None}

# 초기 객체 정보 저장
initial_objects = {}  # {ID: "basketball" or "person"}

def set_initial_objects(inference_state, prompts):
    """
    Initialize objects in the inference state based on YOLO prompts.
    """
    logging.info(f"Received prompts for initialization: {prompts}")

    obj_ids = []

    for obj_id, (bbox, class_id) in prompts.items():
        x_min, y_min, x_max, y_max = bbox
        center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2

        # SAMURAI에 필요한 형태로 변환
        point = torch.tensor([center_x, center_y], dtype=torch.float32).to(inference_state["device"])
        mask = None  # 필요하면 마스크 생성 로직 추가

        obj_ids.append(obj_id)

        # SAMURAI 내부 상태에 객체 등록
        inference_state["obj_id_to_idx"][obj_id] = len(inference_state["obj_id_to_idx"])
        inference_state["obj_idx_to_id"][len(inference_state["obj_id_to_idx"]) - 1] = obj_id

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

    inference_state["obj_ids"] = obj_ids

    logging.info(f"Object IDs after set_initial_objects: {inference_state['obj_ids']}")
    logging.info(f"Initialized cond_frame_outputs: {inference_state['output_dict']['cond_frame_outputs']}")