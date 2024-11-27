# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import json
import time
from collections import deque
from queue import Queue
from threading import Event, Lock, Thread
from Evaluator import Evaluator

import cv2
import numpy as np

from mmpose.apis import (get_track_id, inference_top_down_pose_model,
                         init_pose_model, vis_pose_result)
from mmpose.core import apply_bugeye_effect, apply_sunglasses_effect
from mmpose.utils import StopWatch

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

try:
    import psutil
    psutil_proc = psutil.Process()
except (ImportError, ModuleNotFoundError):
    psutil_proc = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam-id', type=str, default='http://192.168.0.29:5000/video')
    parser.add_argument(
        '--det-config',
        type=str,
        default='ViTPose/demo/mmdetection_cfg/'
        'ssdlite_mobilenetv2_scratch_600e_coco.py',
        help='Config file for detection')
    parser.add_argument(
        '--det-checkpoint',
        type=str,
        default='ViTPose/configs/checkpoints/ssdlite_mobilenetv2_scratch_600e_coco_20210629_110627-974d9307.pth',
        help='Checkpoint file for detection')
    parser.add_argument(
        '--enable-human-pose',
        type=int,
        default=1,
        help='Enable human pose estimation')
    parser.add_argument(
        '--enable-animal-pose',
        type=int,
        default=0,
        help='Enable animal pose estimation')
    parser.add_argument(
        '--human-pose-config',
        type=str,
        # default='ViTPose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/'
        # 'coco-wholebody/vipnas_res50_coco_wholebody_256x192_dark.py',
        default='ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res50_coco_256x192.py',
        help='Config file for human pose')
    parser.add_argument(
        '--human-pose-checkpoint',
        type=str,
        # default='ViTPose/configs/checkpoints/vipnas_res50_wholebody_256x192_dark-67c0ce35_20211112.pth',
        default= 'ViTPose/configs/checkpoints/res50_coco_256x192-ec54d7f3_20200709.pth',
        help='Checkpoint file for human pose')
    parser.add_argument(
        '--human-det-ids',
        type=int,
        default=[1],
        nargs='+',
        help='Object category label of human in detection results.'
        'Default is [1(person)], following COCO definition.')
    parser.add_argument(
        '--animal-pose-config',
        type=str,
        default='ViTPose/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/'
        'animalpose/hrnet_w32_animalpose_256x256.py',
        help='Config file for animal pose')
    parser.add_argument(
        '--animal-pose-checkpoint',
        type=str,
        default='https://download.openmmlab.com/mmpose/animal/hrnet/'
        'hrnet_w32_animalpose_256x256-1aa7f075_20210426.pth',
        help='Checkpoint file for animal pose')
    parser.add_argument(
        '--animal-det-ids',
        type=int,
        default=[16, 17, 18, 19, 20],
        nargs='+',
        help='Object category label of animals in detection results'
        'Default is [16(cat), 17(dog), 18(horse), 19(sheep), 20(cow)], '
        'following COCO definition.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.5,
        help='bbox score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--vis-mode',
        type=int,
        default=2,
        help='0-none. 1-detection only. 2-detection and pose.')
    parser.add_argument(
        '--sunglasses', action='store_true', help='Apply `sunglasses` effect.')
    parser.add_argument(
        '--bugeye', action='store_true', help='Apply `bug-eye` effect.')

    parser.add_argument(
        '--out-video-file',
        type=str,
        default=None,
        help='Record the video into a file. This may reduce the frame rate')

    parser.add_argument(
        '--out-video-fps',
        type=int,
        default=20,
        help='Set the FPS of the output video file.')

    parser.add_argument(
        '--buffer-size',
        type=int,
        default=-1,
        help='Frame buffer size. If set -1, the buffer size will be '
        'automatically inferred from the display delay time. Default: -1')

    parser.add_argument(
        '--inference-fps',
        type=int,
        default=10,
        help='Maximum inference FPS. This is to limit the resource consuming '
        'especially when the detection and pose model are lightweight and '
        'very fast. Default: 10.')

    parser.add_argument(
        '--display-delay',
        type=int,
        default=0,
        help='Delay the output video in milliseconds. This can be used to '
        'align the output video and inference results. The delay can be '
        'disabled by setting a non-positive delay time. Default: 0')

    parser.add_argument(
        '--synchronous-mode',
        action='store_true',
        help='Enable synchronous mode that video I/O and inference will be '
        'temporally aligned. Note that this will reduce the display FPS.')

    return parser.parse_args()


def process_mmdet_results(mmdet_results, class_names=None, cat_ids=1):
    """Process mmdet results to mmpose input format.

    Args:
        mmdet_results: raw output of mmdet model
        class_names: class names of mmdet model
        cat_ids (int or List[int]): category id list that will be preserved
    Returns:
        List[Dict]: detection results for mmpose input
    """
    if isinstance(mmdet_results, tuple):
        mmdet_results = mmdet_results[0]

    if not isinstance(cat_ids, (list, tuple)):
        cat_ids = [cat_ids]

    # only keep bboxes of interested classes
    bbox_results = [mmdet_results[i - 1] for i in cat_ids]
    bboxes = np.vstack(bbox_results)

    # get textual labels of classes
    labels = np.concatenate([
        np.full(bbox.shape[0], i - 1, dtype=np.int32)
        for i, bbox in zip(cat_ids, bbox_results)
    ])
    if class_names is None:
        labels = [f'class: {i}' for i in labels]
    else:
        labels = [class_names[i] for i in labels]

    det_results = []
    for bbox, label in zip(bboxes, labels):
        det_result = dict(bbox=bbox, label=label)
        det_results.append(det_result)
    return det_results


def read_camera():
    # Initialize video reader
    print('Thread "input" started')
    cam_id = args.cam_id

    # Check if cam_id is a digit (e.g., "0" for default webcam)
    if cam_id.isdigit():
        cam_id = int(cam_id)
    vid_cap = cv2.VideoCapture(cam_id)  # Default webcam or video file
    if not vid_cap.isOpened():
        print(f'Cannot open camera (ID={cam_id})')
        exit()

    print(f"Camera (ID={cam_id}) successfully opened.")

    while not event_exit.is_set():
        # Capture a frame from the video source
        ret_val, frame = vid_cap.read()
        if ret_val:
            ts_input = time.time()

            event_inference_done.clear()
            with input_queue_mutex:
                input_queue.append((ts_input, frame))

            if args.synchronous_mode:
                event_inference_done.wait()

            frame_buffer.put((ts_input, frame))
        else:
            # Signal input ending
            frame_buffer.put((None, None))
            break

    vid_cap.release()

def save_frame_to_image(img, args):
    # 현재 시간을 기반으로 파일 이름 생성
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    img_filename = f"frame_{timestamp}.jpg"
    img_path = os.path.join('output', img_filename)
    os.makedirs('output', exist_ok=True)

    # 이미지를 파일로 저장
    cv2.imwrite(img_path, img)
    print(f"Frame saved to {img_path}")

def save_keypoints_to_json(img, pose_results_list, args, ts_input):

    # Generate a timestamped filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    json_filename = f"pose_{timestamp}.json"
    json_path = os.path.join('output', json_filename)
    os.makedirs('output', exist_ok=True)

    height, width, _ = img.shape

    # Initialize JSON data
    keypoints_data = {
        "image_height": height,
        "image_width": width,
        "model_name": args.human_pose_config,
        "device_used": args.device,
        "arguments": vars(args),
        "pose_estimation_info": []
    }

    # Iterate over each model's pose results
    for pose_results in pose_results_list:
        # Iterate over each detected person
        for result in pose_results:
            bbox = result['bbox']
            x_min, y_min, x_max, y_max, score = bbox
            # Calculate corner coordinates
            bbox_coordinates = {
                "top_left": {"x": float(x_min), "y": float(y_min)},
                "top_right": {"x": float(x_max), "y": float(y_min)},
                "bottom_left": {"x": float(x_min), "y": float(y_max)},
                "bottom_right": {"x": float(x_max), "y": float(y_max)},
                "score": float(bbox[4])
            }

            person_data = {
                "bbox": bbox_coordinates,
                "keypoints": []
            }

            keypoints = result['keypoints']
            for idx, (x, y, score_kp) in enumerate(keypoints):
                # Apply keypoint threshold
                if score_kp < args.kpt_thr:
                    continue

                # Determine keypoint type
                if idx in [0, 1, 2, 3, 4]:
                    type_value = 1
                elif idx in [5, 6, 11, 12]:
                    type_value = 2
                elif idx in [13, 14, 15, 16]:
                    type_value = 3
                else:
                    type_value = None

                person_data["keypoints"].append({
                    "index": idx,
                    "coordinates": {"x": int(x), "y": int(y)},
                    "score": float(score_kp),
                    "type": type_value
                })
            keypoints_data["pose_estimation_info"].append(person_data)

    # JSON 데이터 저장
    timestamp = f"{ts_input:.6f}"  # ts_input 추가된 부분
    with open(json_path, "w") as json_file:
        json.dump(keypoints_data, json_file, indent=4)
    print(f"Keypoints saved to {json_path}")
    
    return keypoints_data

def inference_detection():
    print('Thread "det" started')
    stop_watch = StopWatch(window=10)
    min_interval = 1.0 / args.inference_fps
    _ts_last = None  # timestamp when last inference was done

    while True:
        while len(input_queue) < 1:
            time.sleep(0.001)
        with input_queue_mutex:
            ts_input, frame = input_queue.popleft()
        # inference detection
        with stop_watch.timeit('Det'):
            mmdet_results = inference_detector(det_model, frame)

        t_info = stop_watch.report_strings()
        with det_result_queue_mutex:
            det_result_queue.append((ts_input, frame, t_info, mmdet_results))

        # limit the inference FPS
        _ts = time.time()
        if _ts_last is not None and _ts - _ts_last < min_interval:
            time.sleep(min_interval - _ts + _ts_last)
        _ts_last = time.time()


def inference_pose():
    print('Thread "pose" started')
    stop_watch = StopWatch(window=10)

    while True:
        while len(det_result_queue) < 1:
            time.sleep(0.001)
        with det_result_queue_mutex:
            ts_input, frame, t_info, mmdet_results = det_result_queue.popleft()

        pose_results_list = []
        for model_info, pose_history in zip(pose_model_list,
                                            pose_history_list):
            model_name = model_info['name']
            pose_model = model_info['model']
            cat_ids = model_info['cat_ids']
            pose_results_last = pose_history['pose_results_last']
            next_id = pose_history['next_id']

            with stop_watch.timeit(model_name):
                # process mmdet results
                det_results = process_mmdet_results(
                    mmdet_results,
                    class_names=det_model.CLASSES,
                    cat_ids=cat_ids)

                # inference pose model
                dataset_name = pose_model.cfg.data['test']['type']
                pose_results, _ = inference_top_down_pose_model(
                    pose_model,
                    frame,
                    det_results,
                    bbox_thr=args.det_score_thr,
                    format='xyxy',
                    dataset=dataset_name)

                pose_results, next_id = get_track_id(
                    pose_results,
                    pose_results_last,
                    next_id,
                    use_oks=False,
                    tracking_thr=0.3,
                    use_one_euro=True,
                    fps=None)

                pose_results_list.append(pose_results)

                # update pose history
                pose_history['pose_results_last'] = pose_results
                pose_history['next_id'] = next_id

        t_info += stop_watch.report_strings()
        with pose_result_queue_mutex:
            pose_result_queue.append((ts_input, t_info, pose_results_list))

        event_inference_done.set()


def display(evaluator):
    print('Thread "display" started')
    stop_watch = StopWatch(window=10)

    # initialize result status
    ts_inference = None  # timestamp of the latest inference result
    fps_inference = 0.  # infenrece FPS
    t_delay_inference = 0.  # inference result time delay
    pose_results_list = None  # latest inference result
    t_info = []  # upstream time information (list[str])

    # initialize visualization and output
    sunglasses_img = None  # resource image for sunglasses effect
    text_color = (228, 183, 61)  # text color to show time/system information
    vid_out = None  # video writer

    show_evaluator_predictions = False  # Evaluator 예측 결과 표시 여부를 제어하는 토글 변수
    predicted_data = None  # Evaluator의 예측 결과를 저장할 변수

    # show instructions
    print('Keyboard shortcuts: ')
    print('"v": Toggle the visualization of bounding boxes and poses.')
    print('"s": Toggle the sunglasses effect.')
    print('"b": Toggle the bug-eye effect.')
    print('"Q", "q" or Esc: Exit.')

    while True:
        with stop_watch.timeit('_FPS_'):
            # acquire a frame from buffer
            ts_input, frame = frame_buffer.get()
            # input ending signal
            if ts_input is None:
                break

            img = frame.copy()  # 원본 프레임을 복사하여 사용

            # get pose estimation results
            if len(pose_result_queue) > 0:
                with pose_result_queue_mutex:
                    _result = pose_result_queue.popleft()
                    _ts_input, t_info, pose_results_list = _result

                _ts = time.time()
                if ts_inference is not None:
                    fps_inference = 1.0 / (_ts - ts_inference)
                ts_inference = _ts
                t_delay_inference = (_ts - _ts_input) * 1000

            # visualize detection and pose results
            if pose_results_list is not None:
                for model_info, pose_results in zip(pose_model_list,pose_results_list):
                    pose_model = model_info['model']
                    bbox_color = model_info['bbox_color']
                    dataset_name = pose_model.cfg.data['test']['type']

                    # show pose results
                    if args.vis_mode == 1:
                        img = vis_pose_result(
                            pose_model,
                            img,
                            pose_results,
                            radius=4,
                            thickness=2,
                            dataset=dataset_name,
                            kpt_score_thr=1e7,
                            bbox_color=bbox_color)
                    elif args.vis_mode == 2:
                        img = vis_pose_result(
                            pose_model,
                            img,
                            pose_results,
                            radius=4,
                            thickness=2,
                            dataset=dataset_name,
                            kpt_score_thr=args.kpt_thr,
                            bbox_color=bbox_color)

            # delay control
            if args.display_delay > 0:
                t_sleep = args.display_delay * 0.001 - (time.time() - ts_input)
                if t_sleep > 0:
                    time.sleep(t_sleep)
            t_delay = (time.time() - ts_input) * 1000

            # show time information
            t_info_display = stop_watch.report_strings()  # display fps
            t_info_display.append(f'Inference FPS: {fps_inference:>5.1f}')
            t_info_display.append(f'Delay: {t_delay:>3.0f}')
            t_info_display.append(f'Inference Delay: {t_delay_inference:>3.0f}')
            t_info_str = ' | '.join(t_info_display + t_info)
            cv2.putText(img, t_info_str, (20, 20), cv2.FONT_HERSHEY_DUPLEX,0.3, text_color, 1)

            # collect system information
            sys_info = [
                f'RES: {img.shape[1]}x{img.shape[0]}',
                f'Buffer: {frame_buffer.qsize()}/{frame_buffer.maxsize}'
            ]
            if psutil_proc is not None:
                sys_info += [
                    f'CPU: {psutil_proc.cpu_percent():.1f}%',
                    f'MEM: {psutil_proc.memory_percent():.1f}%'
                ]
            sys_info_str = ' | '.join(sys_info)
            cv2.putText(img, sys_info_str, (20, 40), cv2.FONT_HERSHEY_DUPLEX,
                        0.3, text_color, 1)

            # save the output video frame
            if args.out_video_file is not None:
                if vid_out is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    fps = args.out_video_fps
                    frame_size = (img.shape[1], img.shape[0])
                    vid_out = cv2.VideoWriter(args.out_video_file, fourcc, fps,
                                              frame_size)

                vid_out.write(img)

            # Evaluator 예측 결과를 이미지에 그리기
            if show_evaluator_predictions and predicted_data is not None:
                img = draw_predicted_keypoints(img, predicted_data)

            # display
            cv2.imshow('mmpose webcam demo', img)
            keyboard_input = cv2.waitKey(1) & 0xFF

            if keyboard_input in (27, ord('q'), ord('Q')):
                break
            elif keyboard_input == ord('t'):
                show_evaluator_predictions = not show_evaluator_predictions  # 토글 변수 변경
            elif keyboard_input == ord('v'):
                args.vis_mode = (args.vis_mode + 1) % 3
            elif keyboard_input == ord('r'):
                show_evaluator_predictions = not show_evaluator_predictions  # 토글 변수 변경
                if show_evaluator_predictions:
                    # Evaluator의 예측 결과를 계산하고 저장
                    if pose_results_list is not None:
                        keypoints_data = save_keypoints_to_json(img, pose_results_list, args, ts_input)
                        save_frame_to_image(img, args)
                        predicted_data = evaluator.evaluate_keypoints(keypoints_data)
                        print(f"Predicted Data: {predicted_data}")
                else:
                    predicted_data = None  # 토글이 꺼지면 예측 결과 초기화

                # # 필요하다면 predicted_data를 화면에 시각화 가능
                # img = draw_predicted_keypoints(img, predicted_data)


                # # 이미지 표시
                # cv2.imshow('MMPose Webcam Demo', img)

    cv2.destroyAllWindows()
    if vid_out is not None:
        vid_out.release()
    event_exit.set()

def draw_predicted_keypoints(img, predicted_data):
    # predicted_data에서 필요한 정보 추출
    pose_info_list = predicted_data.get('pose_estimation_info', [])
    for pose_info in pose_info_list:
        bbox = pose_info['bbox']
        keypoints = pose_info['keypoints']
        
        # 바운딩 박스 좌표 추출 (필요하다면 사용)
        x_min = bbox['top_left']['x']
        y_min = bbox['top_left']['y']
        x_max = bbox['bottom_right']['x']
        y_max = bbox['bottom_right']['y']
        
        # 키포인트 좌표 추출
        keypoints_coords = []
        for kp in keypoints:
            x = kp['coordinates']['x']
            y = kp['coordinates']['y']
            keypoints_coords.append((int(x), int(y)))

            # 키포인트 그리기
            cv2.circle(img, (int(x), int(y)), radius=5, color=(0, 255, 0), thickness=-1)
        
        # 스켈레톤 연결 정보
        skeleton = [
            [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
            [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
            [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
            [1, 3], [2, 4], [3, 5], [4, 6]
        ]
        
        # 스켈레톤 그리기
        for edge in skeleton:
            start_idx, end_idx = edge
            if start_idx < len(keypoints_coords) and end_idx < len(keypoints_coords):
                x_start, y_start = keypoints_coords[start_idx]
                x_end, y_end = keypoints_coords[end_idx]
                cv2.line(img, (x_start, y_start), (x_end, y_end), color=(0, 255, 0), thickness=2)
    
    return img

def main():
    global args
    global frame_buffer
    global input_queue, input_queue_mutex
    global det_result_queue, det_result_queue_mutex
    global pose_result_queue, pose_result_queue_mutex
    global det_model, pose_model_list, pose_history_list
    global event_exit, event_inference_done

    args = parse_args()

    assert has_mmdet, 'Please install mmdet to run the demo.'
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    # build detection model
    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())

    # build pose models
    pose_model_list = []
    if args.enable_human_pose:
        pose_model = init_pose_model(
            args.human_pose_config,
            args.human_pose_checkpoint,
            device=args.device.lower())
        model_info = {
            'name': 'HumanPose',
            'model': pose_model,
            'cat_ids': args.human_det_ids,
            'bbox_color': (148, 139, 255),
        }
        pose_model_list.append(model_info)
    if args.enable_animal_pose:
        pose_model = init_pose_model(
            args.animal_pose_config,
            args.animal_pose_checkpoint,
            device=args.device.lower())
        model_info = {
            'name': 'AnimalPose',
            'model': pose_model,
            'cat_ids': args.animal_det_ids,
            'bbox_color': 'cyan',
        }
        pose_model_list.append(model_info)

    # store pose history for pose tracking
    pose_history_list = []
    for _ in range(len(pose_model_list)):
        pose_history_list.append({'pose_results_last': [], 'next_id': 0})

    # frame buffer
    if args.buffer_size > 0:
        buffer_size = args.buffer_size
    else:
        # infer buffer size from the display delay time
        # assume that the maximum video fps is 30
        buffer_size = round(30 * (1 + max(args.display_delay, 0) / 1000.))
    frame_buffer = Queue(maxsize=buffer_size)

    # queue of input frames
    # element: (timestamp, frame)
    input_queue = deque(maxlen=1)
    input_queue_mutex = Lock()

    # queue of detection results
    # element: tuple(timestamp, frame, time_info, det_results)
    det_result_queue = deque(maxlen=1)
    det_result_queue_mutex = Lock()

    # queue of detection/pose results
    # element: (timestamp, time_info, pose_results_list)
    pose_result_queue = deque(maxlen=1)
    pose_result_queue_mutex = Lock()

    # Evaluator 초기화
    evaluator = Evaluator(model_path='gat_model.pth')

    try:
        event_exit = Event()
        event_inference_done = Event()
        t_input = Thread(target=read_camera, args=())
        t_det = Thread(target=inference_detection, args=(), daemon=True)
        t_pose = Thread(target=inference_pose, args=(), daemon=True)

        t_input.start()
        t_det.start()
        t_pose.start()

        # run display in the main thread
        display(evaluator)
        # join the input thread (non-daemon)
        t_input.join()

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
