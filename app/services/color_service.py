import os
from collections import Counter

from app.services.state_manager import team_colors

import logging
import cv2 #openCV
import numpy as np
import json

# JSON에서 팀 색상 로드
# JSON 파일에서 팀 색상 로드 함수
def load_team_colors(home_team_id, away_team_id):
    try:
        # 절대 경로로 JSON 파일을 지정
        json_path = os.path.join("app", "static", "data", "team_colors.json")
        with open(json_path, "r") as file:
            color_data = json.load(file)

        # 팀 ID를 기준으로 검색
        home_team_name = next((team for team, data in color_data.items() if str(data.get("teamID")) == home_team_id), None)
        away_team_name = next((team for team, data in color_data.items() if str(data.get("teamID")) == away_team_id), None)

        if not home_team_name or not away_team_name:
            logging.info(f"Error: Could not find teams for IDs - Home: {home_team_id}, Away: {away_team_id}")
            return False

        # 홈팀 색상 정보 가져오기
        home_colors = color_data[home_team_name].get("home", {})
        team_colors["home"] = extract_colors(home_colors)

        # 원정팀 색상 정보 가져오기
        away_colors = color_data[away_team_name].get("away", {})
        team_colors["away"] = extract_colors(away_colors)

        # 결과 출력 확인
        logging.info(f"Loaded colors - Home: {team_colors['home']}, Away: {team_colors['away']}")
        return True if team_colors["home"] and team_colors["away"] else False

    except FileNotFoundError:
        logging.info("Error: team_colors.json file not found.")
        return False

# 색상 추출 함수 (RGB 값 그대로 사용)
def extract_colors(color_dict):
    colors = []
    for i in range(1, 5):
        color_key = f"primary_color" if i == 1 else f"secondary_color" if i == 2 else f"third_color" if i == 3 else f"fourth_color"
        rgb_color = color_dict.get(color_key)
        if rgb_color:
            colors.append(np.array(rgb_color))  # RGB 값 추가
            logging.info(f"Extracted color for {color_key}: {rgb_color}")
    return colors if colors else None

# 유니폼 색상 추출 및 팀 판별 함수
def identify_uniform_color_per_person(player_crop, team_colors_home, team_colors_away, similarity_threshold=30, majority_threshold=20):
    # 홈팀과 원정팀 색상 정보
    home_primary, home_secondary = team_colors_home[0], team_colors_home[1]
    away_primary, away_secondary = team_colors_away[0], team_colors_away[1]

    # 1. 피부색 필터링
    skin_lower = np.array([45, 34, 30], dtype=np.uint8)
    skin_upper = np.array([255, 224, 210], dtype=np.uint8)
    mask_skin = cv2.inRange(player_crop, skin_lower, skin_upper)
    body_filtered = cv2.bitwise_and(player_crop, player_crop, mask=cv2.bitwise_not(mask_skin))

    # 2. 유사한 색상 기준으로 팀 후보 수집
    team_votes = []
    # processed_pixel_count = 0  # 처리한 픽셀 수 제한

    for y in range(body_filtered.shape[0]):
        for x in range(body_filtered.shape[1]):
            # if processed_pixel_count >= max_pixels:
            #     break

            pixel_color = body_filtered[y, x]
            if np.any(pixel_color):
                # 홈팀과 원정팀 색상과의 유사성 비교
                if np.linalg.norm(pixel_color - home_primary) < similarity_threshold:
                    team_votes.append("home_team")
                elif np.linalg.norm(pixel_color - away_primary) < similarity_threshold:
                    team_votes.append("away_team")
                elif np.linalg.norm(pixel_color - home_secondary) < similarity_threshold:
                    team_votes.append("home_team")
                elif np.linalg.norm(pixel_color - away_secondary) < similarity_threshold:
                    team_votes.append("away_team")
                # else:
                #     team_votes.append("cannot_detected")  # 유사한 색상이 없으면 추가

                # 3. 중간에 결과 반환 조건
                count = Counter(team_votes)
                if count["home_team"] >= majority_threshold:
                    return "home_team"
                elif count["away_team"] >= majority_threshold:
                    return "away_team"

                # processed_pixel_count += 1  # 처리한 픽셀 수 증가

    # 4. 최종 투표 결과 확인
    if team_votes:
        team_count = Counter(team_votes)
        most_common_team, count = team_count.most_common(1)[0]
        logging.info(f"Team votes: {team_count}, Assigned to: {most_common_team}")
        return most_common_team
    else:
        logging.warning("Uniform color could not be detected. Assigning as 'cannot_detected'")
        return "cannot_detected"