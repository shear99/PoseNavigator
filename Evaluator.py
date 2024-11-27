import torch
import torch.nn.functional as F
import numpy as np
from GAT import GAT
from KeypointDataset import KeypointDataset

class Evaluator:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # in_channels는 모델에 따라 조정 필요
        in_channels = 10

        # 모델 초기화 (hidden_channels를 훈련 시와 동일하게 설정)
        self.model = GAT(
            in_channels=in_channels, 
            hidden_channels=64, 
            out_channels=in_channels, 
            num_heads=8).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
    
    def evaluate(self, data):
        # data는 torch_geometric.data.Data 객체입니다.
        data = data.to(self.device)
        with torch.no_grad():
            out = self.model(data.x, data.edge_index, data.edge_weight)
            # 좌표 부분만을 사용하여 유사도를 계산합니다.
            # 누락된 키포인트를 고려하여 마스크를 적용합니다.
            valid_mask = (data.x[:, 0] != 0) | (data.x[:, 1] != 0)
            similarity = F.cosine_similarity(out[valid_mask, :2], data.x[valid_mask, :2], dim=1).mean().item()
        return similarity, out.cpu().numpy()  # 예측된 특징 벡터를 반환합니다.
    
    def evaluate_keypoints(self, json_data):
        # json_data는 딕셔너리 형태입니다.
        # KeypointDataset 인스턴스 생성
        dataset = KeypointDataset(json_data=json_data)
        sample_data = dataset[0]
        # 평가 수행
        similarity_score, predicted_features = self.evaluate(sample_data)
        # 예측된 키포인트 좌표 추출
        predicted_keypoints = predicted_features[:, :2]
        
        # 원본 이미지 크기 및 바운딩 박스 정보 추출
        image_width = json_data['image_width']
        image_height = json_data['image_height']
        bbox = json_data['pose_estimation_info'][0]['bbox']
        x_min = bbox['top_left']['x']
        y_min = bbox['top_left']['y']
        x_max = bbox['bottom_right']['x']
        y_max = bbox['bottom_right']['y']
        width_bbox = x_max - x_min
        height_bbox = y_max - y_min

        # 예측된 키포인트 좌표를 원본 이미지 좌표계로 변환
        predicted_keypoints_rescaled = predicted_keypoints.copy()
        predicted_keypoints_rescaled[:, 0] = predicted_keypoints_rescaled[:, 0] * width_bbox + x_min
        predicted_keypoints_rescaled[:, 1] = predicted_keypoints_rescaled[:, 1] * height_bbox + y_min

        # 결과를 동일한 형식의 딕셔너리로 구성
        keypoints_list = []
        for idx, (x, y) in enumerate(predicted_keypoints_rescaled):
            keypoint = {
                "index": idx,
                "coordinates": {
                    "x": float(x),
                    "y": float(y)
                },
                "score": None,  # score는 없으므로 None으로 설정
                "type": None    # type도 없으므로 None으로 설정
            }
            keypoints_list.append(keypoint)
        
        # 반환할 딕셔너리 구성
        output_data = {
            "image_height": image_height,
            "image_width": image_width,
            "model_name": json_data.get("model_name", ""),
            "device_used": json_data.get("device_used", ""),
            "arguments": json_data.get("arguments", {}),
            "pose_estimation_info": [
                {
                    "bbox": bbox,
                    "keypoints": keypoints_list
                }
            ]
        }
        
        return output_data

