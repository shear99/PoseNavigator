import pygame
import json
import math
from pygame.locals import *

# === 상수 및 전역 변수 ===
VALID_INDICES = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
TARGET_INDICES = [7, 8, 9, 10,13, 14, 15, 16]
EXCLUDED_INDICES = [0, 1, 2, 3, 4]
ANGLE_NODES = [(11, 12), (5, 6)]  # 좌우 엉덩이, 좌우 어깨 등
MAX_SCREEN_WIDTH = 1920
MAX_SCREEN_HEIGHT = 1080
CURRENT_POSE_FILE = 'data1/stand4.json'
TARGET_POSE_FILE = 'data1/stand4_predicted.json'
BACKGROUND_IMAGE_FILE = 'data1/stand4.jpg'

# === 선 두께와 반지름 설정 값을 정의 (current와 target 분리, 투명도 포함)
LINE_RADIUS_SETTINGS = {
    "current": {
        "body": {"line": 100, "radius": 60, "alpha": 255},  # 몸통
        "face": {"line": 150, "radius": 110, "alpha": 255},  # 얼굴
        "other": {"line": 80, "radius": 60, "alpha": 255},  # 나머지
    },
    "target": {
        "body": {"line": 100, "radius": 60, "alpha": 128},  # 몸통
        "face": {"line": 150, "radius": 110, "alpha": 128},  # 얼굴
        "other": {"line": 80, "radius": 60, "alpha": 128},  # 나머지
    },
    "heartbeat": {
        "line_thickness": 2,
        "max_radius": 20,
        "min_radius": 5,
        "alpha": 200,
    }
}

# === 함수 정의 ===

# 선 두께와 반지름 계산 함수
def calculate_line_and_radius(screen_width, min_line_width, min_radius, settings):
    calculated_settings = {}
    for pose_type in ['current', 'target']:
        calculated_settings[pose_type] = {}
        for part in settings[pose_type]:
            calculated_settings[pose_type][part] = {
                "line_width": max(int(screen_width / settings[pose_type][part]["line"]), min_line_width),
                "radius": max(int(screen_width / settings[pose_type][part]["radius"]), min_radius),
                "alpha": settings[pose_type][part]["alpha"]
            }
    # Heartbeat 설정 추가
    calculated_settings["heartbeat"] = {
        "line_thickness": settings["heartbeat"]["line_thickness"],
        "max_radius": settings["heartbeat"]["max_radius"],
        "min_radius": settings["heartbeat"]["min_radius"],
        "alpha": settings["heartbeat"]["alpha"]
    }
    return calculated_settings

# JSON 파일에서 포즈 데이터 로드
def load_pose_from_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    pose_info = data['pose_estimation_info'][0]
    keypoints = pose_info['keypoints']
    image_width = data['image_width']
    image_height = data['image_height']
    image_filename = data['image_filename']  # 이미지 파일명 추가
    pose = {
        'keypoints': keypoints,
        'image_width': image_width,
        'image_height': image_height,
        'bbox': pose_info['bbox'],  # 바운딩 박스 추가
        'image_filename': image_filename
    }
    return pose

# 화면 크기 계산 (비율 유지)
def calculate_screen_size(image_width, image_height, max_width, max_height):
    ratio = min(max_width / image_width, max_height / image_height)
    screen_width = int(image_width * ratio)
    screen_height = int(image_height * ratio)
    return screen_width, screen_height

# 키포인트 데이터 처리
def process_keypoints(pose):
    keypoints = []
    for kp in pose['keypoints']:
        index = kp['index']
        x = kp['coordinates']['x']
        y = kp['coordinates']['y']
        keypoints.append({'index': index, 'x': x, 'y': y})
    return keypoints

# 이미지 좌표계를 화면 좌표계로 변환
def transform_keypoints(keypoints, img_width, img_height, screen_width, screen_height):
    transformed = []
    for kp in keypoints:
        x = kp['x'] / img_width * screen_width
        y = kp['y'] / img_height * screen_height
        transformed.append({'index': kp['index'], 'x': x, 'y': y})
    return transformed

# 투명도를 적용하여 선을 그리는 함수
def draw_line_with_alpha(surface, color, start_pos, end_pos, width, alpha):
    temp_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    color_with_alpha = color + (alpha,)
    pygame.draw.line(temp_surface, color_with_alpha, start_pos, end_pos, width)
    surface.blit(temp_surface, (0, 0))

# 투명도를 적용하여 원을 그리는 함수
def draw_circle_with_alpha(surface, color, position, radius, alpha):

    temp_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    color_with_alpha = color + (alpha,)
    pygame.draw.circle(temp_surface, color_with_alpha, position, radius)
    surface.blit(temp_surface, (0, 0))

def draw_heartbeat_effect(screen, current_kps, target_kps, frame_count, settings, center_value):
    max_distance = center_value/5  # 최대 거리 (정규화 기준)
    min_distance = center_value/100   # 최소 거리 (정규화 기준)

    for target_kp in target_kps:
        # TARGET_INDICES에 포함되지 않는 keypoint는 무시
        if target_kp['index'] not in TARGET_INDICES:
            continue

        # 동일 인덱스의 current_keypoints와 거리 계산
        current_kp = next((kp for kp in current_kps if kp['index'] == target_kp['index']), None)
        if not current_kp:
            continue

        # 두 keypoint 간 거리 계산
        distance = math.hypot(current_kp['x'] - target_kp['x'], current_kp['y'] - target_kp['y'])

        # 거리가 너무 가까운 경우 heartbeat 효과 제거
        if distance <= min_distance:
            continue

        # 거리 정규화 (0.0 ~ 1.0)
        normalized_distance = min(max((distance - min_distance) / (max_distance - min_distance), 0.0), 1.0)

        # 민감도를 반영한 선형 변환
        normalized_distance =  min(max((distance - min_distance) / (max_distance - min_distance), 0.0), 1.0)

        # 가까운 거리에 민감하게 반응하도록 비선형 변환
        sensitivity_factor = 3  # 민감도 계수 (값이 클수록 가까운 거리에 민감)
        nonlinear_distance = pow(normalized_distance, 1 / sensitivity_factor)  # 비선형 변환

        # 색상 변화 (멀수록 빨간색, 가까울수록 초록색)
        red = int(255 * nonlinear_distance)
        green = int(255 * (1 - nonlinear_distance))
        color = (red, green, 0)

        # 비트 크기와 주기 변화
        max_radius = settings["max_radius"] + 10 * nonlinear_distance
        min_radius = settings["min_radius"] + 5 * nonlinear_distance
        frequency = 0.1 + 0.2 * nonlinear_distance  # 주기: 멀수록 빠르게

        # 하트비트 크기 계산
        heartbeat_radius = max_radius + min_radius * math.sin(frame_count * frequency)
        heartbeat_radius = max(heartbeat_radius, settings["min_radius"])

        # 하트비트 효과 그리기
        draw_circle_with_alpha(screen, color, (int(target_kp['x']), int(target_kp['y'])), int(heartbeat_radius), settings["alpha"])

# 키포인트와 스켈레톤을 화면에 그리고 좌표값 표시
def draw_pose_with_skeleton_and_coordinates(screen, keypoints, skeleton, sizes, pose_type):
    # 파트별 인덱스
    face_indices = [[0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]  # 얼굴
    body_indices = [[5, 6], [11, 12], [5, 11], [6, 12]]  # 몸통

    for bone in skeleton:
        idx1, idx2 = bone

         # EXCLUDED_INDICES에 포함된 인덱스는 무시
        if idx1 in EXCLUDED_INDICES or idx2 in EXCLUDED_INDICES:
            continue

        kp1 = next((item for item in keypoints if item["index"] == idx1), None)
        kp2 = next((item for item in keypoints if item["index"] == idx2), None)
        if kp1 and kp2:
            # 선 두께와 투명도 선택
            if bone in face_indices:
                line_width = sizes[pose_type]["face"]["line_width"]
                alpha = sizes[pose_type]["face"]["alpha"]
            elif bone in body_indices:
                line_width = sizes[pose_type]["body"]["line_width"]
                alpha = sizes[pose_type]["body"]["alpha"]
            else:
                line_width = sizes[pose_type]["other"]["line_width"]
                alpha = sizes[pose_type]["other"]["alpha"]

            # 투명도 적용하여 선 그리기
            draw_line_with_alpha(screen, (0, 255, 0), 
                                 (int(kp1['x']), int(kp1['y'])), 
                                 (int(kp2['x']), int(kp2['y'])), int(line_width), alpha)

    for kp in keypoints:
        # EXCLUDED_INDICES에 포함된 keypoints는 무시
        if kp['index'] in EXCLUDED_INDICES or (pose_type == "target" and kp['index'] in [11, 12]):
            continue
        
        # 반지름과 투명도 선택
        if any(kp['index'] in pair for pair in face_indices):
            radius = sizes[pose_type]["face"]["radius"]
            alpha = sizes[pose_type]["face"]["alpha"]
        elif kp['index'] in [5, 6, 11, 12]:
            radius = sizes[pose_type]["body"]["radius"]
            alpha = sizes[pose_type]["body"]["alpha"]
        else:
            radius = sizes[pose_type]["other"]["radius"]
            alpha = sizes[pose_type]["other"]["alpha"]

        # 투명도 적용하여 원 그리기
        draw_circle_with_alpha(screen, (0, 255, 0), (int(kp['x']), int(kp['y'])), int(radius), alpha)
        # 좌표 출력 부분은 주석 처리
        # coord_text = f"({int(kp['x'])}, {int(kp['y'])})"
        # text_surface = font.render(coord_text, True, (255, 255, 255))
        # screen.blit(text_surface, (int(kp['x']) + 5, int(kp['y']) - 15))

# 마우스 위치에 가까운 키포인트 선택
def get_selected_keypoint(mouse_pos, keypoints):
    for kp in keypoints:
        distance = math.hypot(mouse_pos[0] - kp['x'], mouse_pos[1] - kp['y'])
        if distance < 10:
            return kp['index']
    return None

# === 메인 함수 ===
def main():
    pygame.init()

    current_pose_data = load_pose_from_json(CURRENT_POSE_FILE)
    target_pose_data = load_pose_from_json(TARGET_POSE_FILE)

    # SCREEN_WIDTH와 SCREEN_HEIGHT를 current_pose_data의 이미지 크기로 설정
    SCREEN_WIDTH, SCREEN_HEIGHT = calculate_screen_size(
        current_pose_data['image_width'], current_pose_data['image_height'], MAX_SCREEN_WIDTH, MAX_SCREEN_HEIGHT
    )

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Pose Manipulation with Effects')
    clock = pygame.time.Clock()

    # 배경 이미지를 화면 크기에 맞게 스케일링
    background_image = pygame.image.load(BACKGROUND_IMAGE_FILE)
    background_image = pygame.transform.scale(background_image, (SCREEN_WIDTH, SCREEN_HEIGHT))

    current_keypoints = process_keypoints(current_pose_data)
    target_keypoints = process_keypoints(target_pose_data)

    # 키포인트를 화면 좌표계로 변환
    current_keypoints = transform_keypoints(
        current_keypoints, 
        current_pose_data['image_width'], 
        current_pose_data['image_height'], 
        SCREEN_WIDTH, 
        SCREEN_HEIGHT
    )
    target_keypoints = transform_keypoints(
        target_keypoints, 
        target_pose_data['image_width'], 
        target_pose_data['image_height'], 
        SCREEN_WIDTH, 
        SCREEN_HEIGHT
    )

    # 별도 스켈레톤 리스트 정의
    current_skeleton = [
        [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
        [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
        [7, 9], [8, 10]
    ]
    target_skeleton = [
        [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]
    ]

    # 최소 선 두께와 반지름 설정
    min_line_width = 2
    min_radius = 4

    # 선 두께와 반지름 계산
    sizes = calculate_line_and_radius(SCREEN_WIDTH, min_line_width, min_radius, LINE_RADIUS_SETTINGS)

    running = True
    selected_kp_index = None
    frame_count = 0

    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

            elif event.type == MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                selected_kp_index = get_selected_keypoint(mouse_pos, current_keypoints)

            elif event.type == MOUSEBUTTONUP:
                selected_kp_index = None

            elif event.type == MOUSEMOTION and selected_kp_index is not None:
                mouse_pos = pygame.mouse.get_pos()
                for kp in current_keypoints:
                    if kp['index'] == selected_kp_index:
                        kp['x'], kp['y'] = mouse_pos

        # 배경 이미지 그리기
        screen.blit(background_image, (0, 0))

        # 목표 포즈의 하트비트 효과
        draw_heartbeat_effect(screen, current_keypoints, target_keypoints, frame_count, sizes["heartbeat"], SCREEN_WIDTH)

        # 현재 포즈 스켈레톤 그리기
        draw_pose_with_skeleton_and_coordinates(screen, current_keypoints, current_skeleton, sizes, "current")

        # 목표 포즈 스켈레톤 그리기
        draw_pose_with_skeleton_and_coordinates(screen, target_keypoints, target_skeleton, sizes, "target")

        # 화면 업데이트
        pygame.display.flip()
        clock.tick(60)
        frame_count += 1

    pygame.quit()


if __name__ == '__main__':
    main()
