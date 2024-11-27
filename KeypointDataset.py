import os
import json
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import numpy as np

class KeypointDataset(Dataset):
    def __init__(self, json_dir=None, json_data=None):
        self.data_list = []
        if json_data is not None:
            self.load_data_from_json(json_data)
        elif json_dir is not None:
            self.load_data_from_dir(json_dir)
        else:
            raise ValueError("Either json_dir or json_data must be provided.")

    def load_data_from_json(self, json_data):
        poses = json_data.get('pose_estimation_info', [])
        for pose in poses:
            keypoints = pose.get('keypoints', [])
            bbox = pose['bbox']
            kp_array = self.process_keypoints(keypoints, bbox)
            data = self.create_data_object(kp_array)
            self.data_list.append(data)

    def load_data_from_dir(self, json_dir):
        # json_dir에서 JSON 파일들을 읽어와서 self.data_list에 추가하는 로직
        pass

    def process_keypoints(self, keypoints, bbox):
        # 바운딩 박스에서 x_min, y_min, x_max, y_max 추출
        x_min = bbox['top_left']['x']
        y_min = bbox['top_left']['y']
        x_max = bbox['bottom_right']['x']
        y_max = bbox['bottom_right']['y']

        width = x_max - x_min
        height = y_max - y_min

        kp_coords = []
        for kp in keypoints:
            x = kp['coordinates']['x']
            y = kp['coordinates']['y']
            # Normalize keypoints to [0,1] within the bounding box
            x_norm = (x - x_min) / width
            y_norm = (y - y_min) / height
            kp_coords.append([x_norm, y_norm])
        kp_array = np.array(kp_coords)
        return kp_array

    def create_skeleton_for_features(self):
        # 각도 계산에 사용할 스켈레톤 연결 및 가중치 정의
        skeleton_for_features = [
            [0,1], [0,2], [0,3], [0,4], [1,2],              # 얼굴
            [2,4], [4,6], [1,3], [3,5],                     # 눈 - 귀, 귀 - 어깨
            [5,6],  [6,12], [5,11],                         # 어깨 - 골반 
            [11,12], [12,14], [11,13],                      # 골반 - 무릎  
            [14,16], [13,15], [15,16],                      # 무릎 - 발 
            [6,8], [8,10], [5,7], [7,9],                    # 팔꿈치 - 어깨
            [6,11], [5,12], [11,14], [12,13], [13,14],
            [6,10], [5,9],
        ]
        # 이미 리스트를 텐서로 변환
        skeleton_for_features = torch.tensor(skeleton_for_features, dtype=torch.long)

        # 각 연결의 가중치 (예시로 설정)
        edge_weights_for_features = torch.tensor([
            0.1, 0.1, 0.1, 0.1, 0.1,
            0.1, 0.3, 0.1, 0.3, 
            0.6, 0.5, 0.4,
            0.6, 0.8, 0.7,
            0.8, 0.8, 0.8,
            0.8, 0.8, 0.8, 0.8,
            0.3, 0.3, 0.3, 0.3, 0.3, 
            0.3, 0.3
        ], dtype=torch.float)

        return skeleton_for_features, edge_weights_for_features

    def create_data_object(self, kp_array):
        x = torch.zeros((17, 2), dtype=torch.float)
        num_keypoints = kp_array.shape[0]
        x[:num_keypoints] = torch.tensor(kp_array, dtype=torch.float)
        
        # Skeleton for angle computation and edge weights
        skeleton_for_features, edge_weights_for_features = self.create_skeleton_for_features()
        data = self.create_data_object_with_features(x, skeleton_for_features, edge_weights_for_features)
        return data

    def create_data_object_with_features(self, x, skeleton_for_features, edge_weights_for_features):
        num_nodes = x.size(0)

        # 각 노드별 연결 정보 생성
        adjacency_list = [[] for _ in range(num_nodes)]
        for idx, edge in enumerate(skeleton_for_features):
            i, j = edge
            weight = edge_weights_for_features[idx]
            adjacency_list[i].append((j, weight))
            adjacency_list[j].append((i, weight))  # 무방향 그래프 가정

        # 각 노드별로 단위 벡터 계산 및 가중치 적용
        features = []
        max_connections = 4
        for i in range(num_nodes):
            connected_nodes = adjacency_list[i]
            unit_vectors = []
            for (j, weight) in connected_nodes:
                if x[j].sum() == 0 or x[i].sum() == 0:
                    unit_vec = torch.zeros(2)
                else:
                    vec = x[j] - x[i]
                    norm = torch.norm(vec)
                    if norm != 0:
                        unit_vec = (vec / norm) * weight
                    else:
                        unit_vec = torch.zeros_like(vec)
                unit_vectors.append(unit_vec)
            if len(unit_vectors) < max_connections:
                unit_vectors += [torch.zeros(2) for _ in range(max_connections - len(unit_vectors))]
            else:
                unit_vectors = unit_vectors[:max_connections]
            unit_vectors_flat = torch.cat(unit_vectors)
            node_feature = torch.cat([x[i], unit_vectors_flat])
            features.append(node_feature)
        x = torch.stack(features)

        # edge_index 생성 시 경고 발생하지 않도록 수정
        edge_index = skeleton_for_features.t().contiguous()
        data = Data(x=x, edge_index=edge_index)
        return data

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
