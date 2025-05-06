import os
import random
import tempfile
import time
import json
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from deepface import DeepFace
import analysis_utils as analysis


detection_threshold = 0.65
frame_interval = 18
face_confidence_threshold = 0.8 
MODEL_TRACK_CONFIDENCE = 0.70

def process_video(input_video_path, faces_output_dir=None, analysis_dir=None, crime_category="unknown", video_out_path=None):
    video_name = os.path.splitext(os.path.basename(input_video_path))[0]
    
    if faces_output_dir is None:
        faces_output_dir = f'detected_faces_{video_name}'
    
    if analysis_dir is None:
        analysis_dir = f'analysis_{video_name}'
        
    if video_out_path is None:
        video_out_path = f'{video_name}_out.mp4'

    Path(faces_output_dir).mkdir(parents=True, exist_ok=True)
    Path(analysis_dir).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps  # in seconds
    video_resolution = f"{frame_width}x{frame_height}"

    ret, frame = cap.read()
    
    if not ret:
        print(f"Error: Couldn't read video {input_video_path}")
        return None


    model = YOLO("yolo11n.pt")

    track_history = defaultdict(lambda: [])

    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

    faces_per_id = {}
    person_data = defaultdict(lambda: analysis.create_default_person_data(video_name))

    video_stats = analysis.create_video_stats(
        video_name=video_name,
        video_resolution=video_resolution,
        fps=fps,
        total_frames=total_frames,
        video_duration=video_duration,
        crime_category=crime_category
    )

    frame_count = 0

    pbar = tqdm(total=total_frames, desc=f"Processing {video_name}")

    start_time = time.time()
    max_processing_time = 180  # 3 minutes in seconds

    while ret:
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time > max_processing_time:
            print(f"Processing time limit (3 minutes) reached for {video_name}. Stopping early.")
            break
            
        for _ in range(frame_interval - 1):
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            pbar.update(1)
        
        if not ret:
            break
        
        frame_count += 1
        
        if frame_count % 50 == 0:
            lighting = analysis.estimate_lighting(frame)
            if lighting != "unknown":
                video_stats['lighting_condition'] = lighting
        
        results = model.track(frame, verbose=False, persist=True, classes=0, iou=0.4, conf=MODEL_TRACK_CONFIDENCE,)
        
        for result in results:
            if result.boxes and hasattr(result.boxes, 'id') and result.boxes.id is not None:
                video_stats['valid_frames'] += 1
                
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2 format
                track_ids = result.boxes.id.int().cpu().tolist()
                confidences = result.boxes.conf.cpu().numpy()
                
                for i, (box, track_id, confidence) in enumerate(zip(boxes, track_ids, confidences)):
                    x1, y1, x2, y2 = box
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    if confidence > detection_threshold:
                        if person_data[track_id]['person_id'] is None:
                            person_data[track_id]['person_id'] = track_id
                        
                        width = x2 - x1
                        height = y2 - y1
                        
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        track_history[track_id].append((float(center_x), float(center_y)))
                        if len(track_history[track_id]) > 30:  # retain 30 tracks for 30 frames
                            track_history[track_id].pop(0)
                        
                        if person_data[track_id]['first_seen_frame'] is None:
                            person_data[track_id]['first_seen_frame'] = frame_count
                        person_data[track_id]['last_seen_frame'] = frame_count
                        person_data[track_id]['frames_detected'] += 1
                        person_data[track_id]['frame_ids'].append(frame_count)
                        person_data[track_id]['confidence_values'].append(float(confidence))
                        person_data[track_id]['bounding_boxes'].append([int(x1), int(y1), int(x2), int(y2)])
                        person_data[track_id]['positions'].append([float(center_x), float(center_y)])
                        
                        if len(person_data[track_id]['faces_detected']) < 5:
                            person_region = frame[y1:y2, x1:x2]
                            
                            if person_region.size > 0:  # Check if region is valid
                                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                                    temp_frame_path = temp_file.name
                                    cv2.imwrite(temp_frame_path, person_region)
                                
                                try:
                                    resultado = DeepFace.extract_faces(
                                        img_path=temp_frame_path,
                                        detector_backend='retinaface',
                                        enforce_detection=False,
                                        expand_percentage=30,
                                        align=True
                                    )
                                    
                                    if resultado and len(resultado) > 0:
                                        face = resultado[0]  # Get only the first face
                                        face_img = (face['face'] * 255).astype(np.uint8)
                                        face_confidence = face.get('confidence', 1.0)
                                        facial_area = face.get('facial_area', [0, 0, 0, 0])
                                        
                                        process_face(
                                            person_data=person_data,
                                            track_id=track_id,
                                            face_img=face_img,
                                            face_confidence=face_confidence,
                                            faces_output_dir=faces_output_dir,
                                            frame_count=frame_count,
                                            threshold=face_confidence_threshold,
                                            facial_area=facial_area,
                                            x1=x1,
                                            y1=y1
                                        )
                                    
                                except Exception as e:
                                    print(f"Error processing face for ID {track_id}: {str(e)}")
                                
                                try:
                                    os.unlink(temp_frame_path)
                                except:
                                    pass
            
        ret, frame = cap.read()
        pbar.update(1)

    processing_time = time.time() - start_time
    pbar.close()
    cap.release()
    cv2.destroyAllWindows()

    person_data = consolidate_demographics(person_data)
    
    json_output_path = analysis.post_process_data(person_data, video_stats, video_name, analysis_dir)

    return {
        "video_name": video_name,
        "processing_time": processing_time,
        "persons_detected": len(person_data),
        "faces_detected": video_stats['total_faces_detected'],
        "json_output_path": json_output_path,
        "faces_output_dir": faces_output_dir,
        "video_output_path": video_out_path
    }

def process_dataset(dataset_dir, output_base_dir="output"):
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        return None
    
    os.makedirs(output_base_dir, exist_ok=True)
    
    checkpoint_file = os.path.join(output_base_dir, "checkpoint.json")
    processed_videos = {}
    
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                processed_videos = json.load(f)
            print(f"Loaded checkpoint with {sum(len(videos) for videos in processed_videos.values())} processed videos")
        except Exception as e:
            print(f"Error loading checkpoint file: {str(e)}")
            processed_videos = {}
    
    results = {}
    
    class_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()], key=lambda x: x.name)
    
    total_videos = 0
    class_video_counts = {}
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        video_files = []
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            video_files.extend(list(class_dir.glob(f"*{ext}")))
        
        video_files = sorted(video_files, key=lambda x: x.name)
        
        class_video_counts[class_name] = len(video_files)
        total_videos += len(video_files)
    
    print(f"Found {total_videos} videos in {len(class_dirs)} classes")
    
    videos_processed = 0
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        print(f"\nProcessing class: {class_name} ({class_video_counts[class_name]} videos)")
        
        class_output_dir = os.path.join(output_base_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)
        
        video_files = []
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            video_files.extend(list(class_dir.glob(f"*{ext}")))
        
        video_files = sorted(video_files, key=lambda x: x.name)
        
        if not video_files:
            print(f"  No video files found in {class_dir}")
            continue
        
        class_results = []
        
        if class_name not in processed_videos:
            processed_videos[class_name] = []
        
        for video_idx, video_file in enumerate(video_files):
            video_path = str(video_file)
            video_name = video_file.stem
            
            if video_name in processed_videos[class_name]:
                print(f"Skipping already processed video: {video_name} [{video_idx+1}/{len(video_files)}]")
                videos_processed += 1
                continue
            
            print(f"\nProcessing video: {video_name} [{video_idx+1}/{len(video_files)}] - Overall progress: {videos_processed}/{total_videos}")
            
            try:
                video_output_dir = os.path.join(class_output_dir, video_name)
                faces_dir = os.path.join(video_output_dir, "faces")
                analysis_dir = os.path.join(video_output_dir, "analysis")
                
                result = process_video(
                    input_video_path=video_path,
                    faces_output_dir=faces_dir,
                    analysis_dir=analysis_dir,
                    crime_category=class_name,
                    video_out_path=os.path.join(video_output_dir, f"{video_name}_out.mp4")
                )
                
                if result:
                    class_results.append(result)
                    
                    processed_videos[class_name].append(video_name)
                    
                    try:
                        with open(checkpoint_file, 'w') as f:
                            json.dump(processed_videos, f)
                    except Exception as e:
                        print(f"Warning: Failed to save checkpoint: {str(e)}")
                    
            except Exception as e:
                print(f"Error processing video {video_name}: {str(e)}")
            
            videos_processed += 1
        
        results[class_name] = class_results
    
    try:
        with open(checkpoint_file, 'w') as f:
            json.dump(processed_videos, f)
    except Exception as e:
        print(f"Warning: Failed to save final checkpoint: {str(e)}")
    
    return results 

def draw_boxes(frame, boxes, track_ids, confidences, track_history, face_data=None):

    vis_frame = frame.copy()
    
    colors = [
        (0, 0, 255),     
        (0, 255, 0),     
        (255, 0, 0),     
        (255, 255, 0),   
        (255, 0, 255),   
        (0, 255, 255),   
        (128, 0, 255),   
        (255, 128, 0),   
        (0, 255, 128),
        (128, 128, 255)  
    ]
    
    for i, (box, track_id, confidence) in enumerate(zip(boxes, track_ids, confidences)):
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        color = colors[track_id % len(colors)]
        
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
        
        cv2.putText(vis_frame, f"ID:{track_id} {confidence:.2f}", 
                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        points = track_history[track_id]
        for j in range(1, len(points)):
            if points[j-1] is None or points[j] is None:
                continue
            p1 = (int(points[j-1][0]), int(points[j-1][1]))
            p2 = (int(points[j][0]), int(points[j][1]))
            cv2.line(vis_frame, p1, p2, color, 2)
            
        if face_data and track_id in face_data:
            for face_info in face_data.get(track_id, {}).get('faces_detected', []):
                if 'bbox' in face_info and face_info['bbox']:
                    fx, fy, fw, fh = face_info['bbox']
                    cv2.rectangle(vis_frame, 
                                 (fx, fy), 
                                 (fx + fw, fy + fh), 
                                 (0, 255, 0), 2)
                    face_conf = face_info.get('confidence', 0)
                    cv2.putText(vis_frame, f"Face: {face_conf:.2f}", 
                               (fx, fy-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    return vis_frame

def process_face(person_data, track_id, face_img, face_confidence, faces_output_dir, frame_count, threshold=0.9, facial_area=None, x1=0, y1=0):

    if face_confidence < threshold:
        return False, None
    
    if len(person_data[track_id]['faces_detected']) >= 5:
        return False, None  # Já temos 5 faces, não salva mais
    
    person_data[track_id]['face_confidence_values'].append(face_confidence)
    
    person_folder = os.path.join(faces_output_dir, f'person_{track_id}')
    Path(person_folder).mkdir(parents=True, exist_ok=True)
    
    face_index = len(person_data[track_id]['faces_detected'])
    
    face_path = os.path.join(person_folder, 
                           f'face_{face_index}_conf_{face_confidence:.2f}.jpg')
    
    if len(face_img.shape) == 3 and face_img.shape[2] == 3: 
        face_img_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
    else:
        face_img_bgr = face_img 
    
    cv2.imwrite(face_path, face_img_bgr)
    
    face_bbox = None
    if facial_area:
        try:
            if isinstance(facial_area, dict) and all(k in facial_area for k in ['x', 'y', 'w', 'h']):
                fx, fy, fw, fh = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
            elif isinstance(facial_area, (list, tuple)) and len(facial_area) >= 4:
                fx, fy, fw, fh = facial_area[:4]
            else:
                fx, fy, fw, fh = 0, 0, 0, 0
                
            abs_fx = x1 + fx
            abs_fy = y1 + fy
            face_bbox = [int(abs_fx), int(abs_fy), int(fw), int(fh)]
        except Exception as e:
            print(f"Error calculating face coordinates: {e}")
    
    face_data = {
        'path': face_path,
        'confidence': face_confidence,
        'frame_id': frame_count,
        'bbox': face_bbox, 
        'demographics': {}  
    }
    
    person_data[track_id]['faces_detected'].append(face_data)
    
    try_extract_demographics(person_data, track_id, face_path, face_index)
    
    return True, face_path

def try_extract_demographics(person_data, track_id, face_path, face_index):
    try:
        demog = DeepFace.analyze(
            img_path=face_path,
            actions=['race', 'gender', 'age'],
            enforce_detection=False,
            silent=True
        )
        
        if demog and len(demog) > 0:
            demog_result = demog[0]
            face_demographics = {}
            
            face_demographics['face_confidence'] = demog_result.get('face_confidence', 0)
            
            if 'race' in demog_result and isinstance(demog_result['race'], dict):
                face_demographics['race_scores'] = demog_result['race'].copy()
                dominant_race = max(demog_result['race'].items(), key=lambda x: x[1])
                face_demographics['dominant_race'] = dominant_race[0]
                face_demographics['dominant_race_confidence'] = dominant_race[1]
            
            if 'gender' in demog_result or 'dominant_gender' in demog_result:
                gender_key = 'dominant_gender' if 'dominant_gender' in demog_result else 'gender'
                face_demographics['gender'] = demog_result[gender_key]
                if 'gender' in demog_result and isinstance(demog_result['gender'], dict):
                    face_demographics['gender_scores'] = demog_result['gender'].copy()
                    face_demographics['gender_confidence'] = demog_result['gender'].get(face_demographics['gender'], 0)
                else:
                    face_demographics['gender_confidence'] = demog_result.get('gender_confidence', 0)
            
            if 'age' in demog_result:
                try:
                    age = float(demog_result['age'])
                    face_demographics['age'] = age
                    if age < 18:
                        age_range = "under_18"
                    elif age < 30:
                        age_range = "18-29"
                    elif age < 45:
                        age_range = "30-44"
                    elif age < 60:
                        age_range = "45-59"
                    else:
                        age_range = "60+"
                    face_demographics['age_range'] = age_range
                except (ValueError, TypeError) as e:
                    print(f"Error parsing age for ID {track_id}: {str(e)}")
                    face_demographics['age'] = None
                    face_demographics['age_range'] = "unknown"
            else:
                face_demographics['age'] = None
                face_demographics['age_range'] = "unknown"
            
            person_data[track_id]['faces_detected'][face_index]['demographics'] = face_demographics
            
            
    except Exception as e:
        print(f"Error analyzing demographics for ID {track_id}, face {face_index}: {str(e)}")

def consolidate_demographics(person_data):
    for track_id, data in person_data.items():
        faces = data.get('faces_detected', [])
        
        if not faces:
            continue
            
        consolidated = {
            'race_scores': {},
            'age_sum': 0,
            'age_count': 0,
            'age_ranges': {},
            'gender_counts': {},
            'total_confidence': 0,
            'face_count': len(faces)
        }
        
        for face in faces:
            demographics = face.get('demographics', {})
            face_confidence = demographics.get('face_confidence', 0)
            
            if face_confidence < 0.5:
                consolidated['face_count'] -= 1
                continue
                
            consolidated['total_confidence'] += face_confidence
            
            if 'race_scores' in demographics and isinstance(demographics['race_scores'], dict):
                for race, score in demographics['race_scores'].items():
                    if race not in consolidated['race_scores']:
                        consolidated['race_scores'][race] = 0
                    consolidated['race_scores'][race] += score * face_confidence
            
            if 'age' in demographics and demographics['age'] is not None:
                consolidated['age_sum'] += demographics['age'] * face_confidence
                consolidated['age_count'] += face_confidence
                
                if 'age_range' in demographics and demographics['age_range'] != 'unknown':
                    age_range = demographics['age_range']
                    if age_range not in consolidated['age_ranges']:
                        consolidated['age_ranges'][age_range] = 0
                    consolidated['age_ranges'][age_range] += face_confidence
            
            if 'gender' in demographics and demographics['gender']:
                gender = demographics['gender']
                if gender not in consolidated['gender_counts']:
                    consolidated['gender_counts'][gender] = 0
                consolidated['gender_counts'][gender] += face_confidence
        
        if consolidated['face_count'] == 0:
            continue
            
        if consolidated['race_scores']:
            dominant_race = max(consolidated['race_scores'].items(), key=lambda x: x[1])
            data['demographics']['dominant_race'] = dominant_race[0]
            data['demographics']['dominant_race_confidence'] = dominant_race[1] / consolidated['total_confidence'] if consolidated['total_confidence'] > 0 else 0
            data['demographics']['race_scores'] = {k: v / consolidated['total_confidence'] for k, v in consolidated['race_scores'].items()}
        
        if consolidated['age_count'] > 0:
            data['demographics']['age'] = consolidated['age_sum'] / consolidated['age_count']
            
            if consolidated['age_ranges']:
                dominant_age_range = max(consolidated['age_ranges'].items(), key=lambda x: x[1])
                data['demographics']['age_range'] = dominant_age_range[0]
                data['demographics']['age_range_confidence'] = dominant_age_range[1] / consolidated['total_confidence'] if consolidated['total_confidence'] > 0 else 0
        
        if consolidated['gender_counts']:
            dominant_gender = max(consolidated['gender_counts'].items(), key=lambda x: x[1])
            data['demographics']['gender'] = dominant_gender[0]
            data['demographics']['gender_confidence'] = dominant_gender[1] / consolidated['total_confidence'] if consolidated['total_confidence'] > 0 else 0
        
        data['demographics']['analyzed_faces'] = consolidated['face_count']
        data['demographics']['total_confidence'] = consolidated['total_confidence']
    
    return person_data 