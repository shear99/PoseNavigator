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
                         init_pose_model)
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

TARGET_INDICES = [7, 8, 9, 10, 13, 14, 15, 16]

LINE_SET_NORMAL = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
    [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
    [7, 9], [8, 10]
]

LINE_SET_REDUCED = [
    [11, 12], [5, 6]  
]

SKELETON_COLOR = (0, 0, 0)     
SKELETON_THICKNESS = 2
KEYPOINT_COLOR = (255, 255, 255) 
KEYPOINT_RADIUS = 4
KPT_THR = 0.3

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam-id', type=str, default='http://192.168.0.30:5000/video')
    parser.add_argument(
        '--det-config',
        type=str,
        default='ViTPose/demo/mmdetection_cfg/ssdlite_mobilenetv2_scratch_600e_coco.py',
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
        default='ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res50_coco_256x192.py',
        help='Config file for human pose')
    parser.add_argument(
        '--human-pose-checkpoint',
        type=str,
        default='ViTPose/configs/checkpoints/res50_coco_256x192-ec54d7f3_20200709.pth',
        help='Checkpoint file for human pose')
    parser.add_argument(
        '--human-det-ids',
        type=int,
        default=[1],
        nargs='+',
        help='Object category label of human in detection results.')
    parser.add_argument(
        '--animal-pose-config',
        type=str,
        default='ViTPose/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/animalpose/hrnet_w32_animalpose_256x256.py',
        help='Config file for animal pose')
    parser.add_argument(
        '--animal-pose-checkpoint',
        type=str,
        default='https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_animalpose_256x256-1aa7f075_20210426.pth',
        help='Checkpoint file for animal pose')
    parser.add_argument(
        '--animal-det-ids',
        type=int,
        default=[16, 17, 18, 19, 20],
        nargs='+',
        help='Object category label of animals in detection results')
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
        help='Record the video into a file.')
    parser.add_argument(
        '--out-video-fps',
        type=int,
        default=20,
        help='Set the FPS of the output video file.')
    parser.add_argument(
        '--buffer-size',
        type=int,
        default=-1,
        help='Frame buffer size.')
    parser.add_argument(
        '--inference-fps',
        type=int,
        default=10,
        help='Maximum inference FPS.')
    parser.add_argument(
        '--display-delay',
        type=int,
        default=0,
        help='Delay the output video in milliseconds.')
    parser.add_argument(
        '--synchronous-mode',
        action='store_true',
        help='Enable synchronous mode.')
    return parser.parse_args()

def process_mmdet_results(mmdet_results, class_names=None, cat_ids=1):
    if isinstance(mmdet_results, tuple):
        mmdet_results = mmdet_results[0]

    if not isinstance(cat_ids, (list, tuple)):
        cat_ids = [cat_ids]

    bbox_results = [mmdet_results[i - 1] for i in cat_ids]
    bboxes = np.vstack(bbox_results)

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
    print('Thread "input" started')
    cam_id = args.cam_id

    if cam_id.isdigit():
        cam_id = int(cam_id)
    vid_cap = cv2.VideoCapture(cam_id)
    if not vid_cap.isOpened():
        print(f'Cannot open camera (ID={cam_id})')
        exit()

    print(f"Camera (ID={cam_id}) successfully opened.")

    while not event_exit.is_set():
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
            frame_buffer.put((None, None))
            break

    vid_cap.release()

def save_frame_to_image(img, args):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    img_filename = f"frame_{timestamp}.jpg"
    img_path = os.path.join('output', img_filename)
    os.makedirs('output', exist_ok=True)
    cv2.imwrite(img_path, img)
    print(f"Frame saved to {img_path}")

def save_keypoints_to_json(img, pose_results_list, args, ts_input):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    json_filename = f"pose_{timestamp}.json"
    json_path = os.path.join('output', json_filename)
    os.makedirs('output', exist_ok=True)

    height, width, _ = img.shape

    keypoints_data = {
        "image_height": height,
        "image_width": width,
        "model_name": args.human_pose_config,
        "device_used": args.device,
        "arguments": vars(args),
        "pose_estimation_info": []
    }

    for pose_results in pose_results_list:
        for result in pose_results:
            bbox = result['bbox']
            x_min, y_min, x_max, y_max, score = bbox
            bbox_coordinates = {
                "top_left": {"x": float(x_min), "y": float(y_min)},
                "top_right": {"x": float(x_max), "y": float(y_min)},
                "bottom_left": {"x": float(x_min), "y": float(y_max)},
                "bottom_right": {"x": float(x_max), "y": float(y_max)},
                "score": float(score)
            }

            person_data = {
                "bbox": bbox_coordinates,
                "keypoints": []
            }

            keypoints = result['keypoints']
            for idx, (x, y, score_kp) in enumerate(keypoints):
                if score_kp < args.kpt_thr:
                    continue
                person_data["keypoints"].append({
                    "index": idx,
                    "coordinates": {"x": int(x), "y": int(y)},
                    "score": float(score_kp),
                    "type": None
                })
            keypoints_data["pose_estimation_info"].append(person_data)

    timestamp = f"{ts_input:.6f}"
    with open(json_path, "w") as json_file:
        json.dump(keypoints_data, json_file, indent=4)
    print(f"Keypoints saved to {json_path}")

    return keypoints_data

def inference_detection():
    print('Thread "det" started')
    stop_watch = StopWatch(window=10)
    min_interval = 1.0 / args.inference_fps
    _ts_last = None

    while True:
        while len(input_queue) < 1:
            time.sleep(0.001)
        with input_queue_mutex:
            ts_input, frame = input_queue.popleft()

        with stop_watch.timeit('Det'):
            mmdet_results = inference_detector(det_model, frame)

        t_info = stop_watch.report_strings()
        with det_result_queue_mutex:
            det_result_queue.append((ts_input, frame, t_info, mmdet_results))

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
        for model_info, pose_history in zip(pose_model_list, pose_history_list):
            model_name = model_info['name']
            pose_model = model_info['model']
            cat_ids = model_info['cat_ids']
            pose_results_last = pose_history['pose_results_last']
            next_id = pose_history['next_id']

            with stop_watch.timeit(model_name):
                det_results = process_mmdet_results(
                    mmdet_results,
                    class_names=det_model.CLASSES,
                    cat_ids=cat_ids)

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

                pose_history['pose_results_last'] = pose_results
                pose_history['next_id'] = next_id

        t_info += stop_watch.report_strings()
        with pose_result_queue_mutex:
            pose_result_queue.append((ts_input, t_info, pose_results_list))

        event_inference_done.set()

def draw_custom_skeleton(img, pose_results, skeleton_lines, kpt_thr=KPT_THR):
    for result in pose_results:
        keypoints = result['keypoints']
        for start_idx, end_idx in skeleton_lines:
            if start_idx in [0,1,2,3,4] or end_idx in [0,1,2,3,4]:
                continue
            x1, y1, score1 = keypoints[start_idx]
            x2, y2, score2 = keypoints[end_idx]
            if score1 > kpt_thr and score2 > kpt_thr:
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), SKELETON_COLOR, thickness=SKELETON_THICKNESS)
    return img

def draw_custom_keypoints(img, pose_results, kpt_thr=KPT_THR):
    for result in pose_results:
        keypoints = result['keypoints']
        for i, (x, y, score) in enumerate(keypoints):
            if i in [0,1,2,3,4]:  
                continue
            if score > kpt_thr:
                cv2.circle(img, (int(x), int(y)), KEYPOINT_RADIUS, KEYPOINT_COLOR, -1)
    return img

def draw_predicted_keypoints(img, predicted_data, radius=5, color=(0, 255, 0), alpha=0.8):
    pose_info_list = predicted_data.get('pose_estimation_info', [])
    overlay = img.copy()
    
    for pose_info in pose_info_list:
        keypoints = pose_info['keypoints']
        for kp in keypoints:
            index = kp['index']
            if index in [0, 1, 2, 3, 4]:
                continue
            x = kp['coordinates']['x']
            y = kp['coordinates']['y']
            cv2.circle(overlay, (int(x), int(y)), radius, color, -1)


    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return img


def draw_heartbeat_effect(img, current_data, target_data, frame_count):
    HEARTBEAT_MAX_RADIUS = 6
    HEARTBEAT_MIN_RADIUS = 5
    HEARTBEAT_COLOR_SENSITIVITY = 1
    HEARTBEAT_FREQUENCY_BASE = 0.1
    HEARTBEAT_FREQUENCY_SCALE = 0.2
    HEARTBEAT_ALPHA = 0.8

    if len(current_data.get('pose_estimation_info', [])) == 0 or len(target_data.get('pose_estimation_info', [])) == 0:
        return img

    current_pose = current_data['pose_estimation_info'][0]['keypoints']
    target_pose = target_data['pose_estimation_info'][0]['keypoints']

    height, width, _ = img.shape
    center_value = width
    max_distance = center_value / 5
    min_distance = center_value / 100

    current_kp_dict = {kp['index']: kp for kp in current_pose}
    target_kp_dict = {kp['index']: kp for kp in target_pose}

    overlay = img.copy()

    for tidx in [7, 8, 9, 10, 13, 14, 15, 16]:
        if tidx not in target_kp_dict:
            continue
        t_kp = target_kp_dict[tidx]
        c_kp = current_kp_dict.get(tidx, None)
        if c_kp is None:
            continue
        if tidx in [0,1,2,3,4]:
            continue

        cx, cy = c_kp['coordinates']['x'], c_kp['coordinates']['y']
        tx, ty = t_kp['coordinates']['x'], t_kp['coordinates']['y']
        distance = np.hypot(cx - tx, cy - ty)

        if distance <= min_distance:
            continue

        normalized_distance = min(max((distance - min_distance) / (max_distance - min_distance), 0.0), 1.0)
        nonlinear_distance = normalized_distance ** (1 / HEARTBEAT_COLOR_SENSITIVITY)

        red = int(255 * nonlinear_distance)
        green = int(255 * (1 - nonlinear_distance))
        color = (0, green, red)

        max_radius = HEARTBEAT_MAX_RADIUS + 10 * nonlinear_distance
        min_radius = HEARTBEAT_MIN_RADIUS + 5 * nonlinear_distance
        frequency = HEARTBEAT_FREQUENCY_BASE + HEARTBEAT_FREQUENCY_SCALE * nonlinear_distance

        heartbeat_radius = max_radius + min_radius * np.sin(frame_count * frequency)
        if heartbeat_radius < min_radius:
            heartbeat_radius = min_radius

        cv2.circle(overlay, (int(tx), int(ty)), int(heartbeat_radius), color, -1)

    img = cv2.addWeighted(overlay, HEARTBEAT_ALPHA, img, 1 - HEARTBEAT_ALPHA, 0)

    return img

def display(evaluator):
    print('Thread "display" started')
    stop_watch = StopWatch(window=10)

    ts_inference = None
    fps_inference = 0.
    t_delay_inference = 0.
    pose_results_list = None
    t_info = []
    vid_out = None

    show_evaluator_predictions = False
    predicted_data = None

    frame_count = 0

    use_reduced_skeleton = False

    print('Keyboard shortcuts: ')
    print('"v": Toggle the visualization of bounding boxes and poses.')
    print('"s": Toggle the sunglasses effect.')
    print('"b": Toggle the bug-eye effect.')
    print('"Q", "q" or Esc: Exit.')
    print('"r": Evaluate keypoints and toggle predicted pose display (toggle skeleton lines)')

    while True:
        with stop_watch.timeit('_FPS_'):
            ts_input, frame = frame_buffer.get()
            if ts_input is None:
                break

            img = frame.copy()

            if len(pose_result_queue) > 0:
                with pose_result_queue_mutex:
                    _result = pose_result_queue.popleft()
                    _ts_input, t_info, pose_results_list = _result

                _ts = time.time()
                if ts_inference is not None:
                    fps_inference = 1.0 / (_ts - ts_inference)
                ts_inference = _ts
                t_delay_inference = (_ts - _ts_input) * 1000

                if pose_results_list is not None and args.vis_mode in (1,2):
                    skeleton_lines = LINE_SET_REDUCED if use_reduced_skeleton else LINE_SET_NORMAL
                    for pose_results in pose_results_list:
                        img = draw_custom_skeleton(img, pose_results, skeleton_lines, kpt_thr=args.kpt_thr)
                        img = draw_custom_keypoints(img, pose_results, kpt_thr=args.kpt_thr)

            if args.display_delay > 0:
                t_sleep = args.display_delay * 0.001 - (time.time() - ts_input)
                if t_sleep > 0:
                    time.sleep(t_sleep)
            t_delay = (time.time() - ts_input) * 1000

            t_info_display = stop_watch.report_strings()
            t_info_display.append(f'Inference FPS: {fps_inference:>5.1f}')
            t_info_display.append(f'Delay: {t_delay:>3.0f}')
            t_info_display.append(f'Inference Delay: {t_delay_inference:>3.0f}')
            t_info_str = ' | '.join(t_info_display + t_info)
            cv2.putText(img, t_info_str, (20, 20), cv2.FONT_HERSHEY_DUPLEX,0.3, (228, 183, 61), 1)

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
                        0.3, (228, 183, 61), 1)

            if args.out_video_file is not None:
                if vid_out is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    fps = args.out_video_fps
                    frame_size = (img.shape[1], img.shape[0])
                    vid_out = cv2.VideoWriter(args.out_video_file, fourcc, fps,
                                              frame_size)
                vid_out.write(img)

            if show_evaluator_predictions and predicted_data is not None:
                img = draw_predicted_keypoints(
                        img,
                        predicted_data,
                        radius=5,          
                        color=(0, 255, 0), 
                        alpha=0.7          
                    )

                if pose_results_list is not None and len(pose_results_list) > 0 and len(pose_results_list[0]) > 0:
                    current_pose_json = {
                        "image_width": img.shape[1],
                        "image_height": img.shape[0],
                        "pose_estimation_info": []
                    }
                    pose_item = pose_results_list[0][0]
                    bbox = pose_item['bbox']
                    bbox_coordinates = {
                        "top_left": {"x": float(bbox[0]), "y": float(bbox[1])},
                        "top_right": {"x": float(bbox[2]), "y": float(bbox[1])},
                        "bottom_left": {"x": float(bbox[0]), "y": float(bbox[3])},
                        "bottom_right": {"x": float(bbox[2]), "y": float(bbox[3])},
                        "score": float(bbox[4])
                    }
                    keypoints_list = []
                    for idx, (x, y, s) in enumerate(pose_item['keypoints']):
                        keypoints_list.append({
                            "index": idx,
                            "coordinates": {"x": int(x), "y": int(y)},
                            "score": float(s),
                            "type": None
                        })
                    current_pose_json["pose_estimation_info"].append({
                        "bbox": bbox_coordinates,
                        "keypoints": keypoints_list
                    })

                    img = draw_heartbeat_effect(img, current_pose_json, predicted_data, frame_count)

            cv2.imshow('mmpose webcam demo', img)
            keyboard_input = cv2.waitKey(1) & 0xFF

            if keyboard_input in (27, ord('q'), ord('Q')):
                break
            elif keyboard_input == ord('v'):
                args.vis_mode = (args.vis_mode + 1) % 3
            elif keyboard_input == ord('r'):
                show_evaluator_predictions = not show_evaluator_predictions
                use_reduced_skeleton = show_evaluator_predictions 
                if show_evaluator_predictions:
                    if pose_results_list is not None:
                        keypoints_data = save_keypoints_to_json(img, pose_results_list, args, ts_inference)
                        save_frame_to_image(img, args)
                        predicted_data = evaluator.evaluate_keypoints(keypoints_data)
                        print(f"Predicted Data: {predicted_data}")
                else:
                    predicted_data = None

            frame_count += 1

    cv2.destroyAllWindows()
    if vid_out is not None:
        vid_out.release()
    event_exit.set()

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

    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())

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

    pose_history_list = []
    for _ in range(len(pose_model_list)):
        pose_history_list.append({'pose_results_last': [], 'next_id': 0})

    if args.buffer_size > 0:
        buffer_size = args.buffer_size
    else:
        buffer_size = round(30 * (1 + max(args.display_delay, 0) / 1000.))
    frame_buffer = Queue(maxsize=buffer_size)

    input_queue = deque(maxlen=1)
    input_queue_mutex = Lock()

    det_result_queue = deque(maxlen=1)
    det_result_queue_mutex = Lock()

    pose_result_queue = deque(maxlen=1)
    pose_result_queue_mutex = Lock()

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

        display(evaluator)
        t_input.join()

    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
