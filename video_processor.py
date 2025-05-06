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
    """
    Process a video file for person tracking and face detection.
    
    Args:
        input_video_path (str): Path to input video file
        faces_output_dir (str, optional): Directory to save detected faces
        analysis_dir (str, optional): Directory to save analysis results
        crime_category (str, optional): Crime category label
        video_out_path (str, optional): Path to save output video
        
    Returns:
        dict: Dictionary with results and statistics
    """
    # Get video name without extension
    video_name = os.path.splitext(os.path.basename(input_video_path))[0]
    
    # Set default output paths if not provided
    if faces_output_dir is None:
        faces_output_dir = f'detected_faces_{video_name}'
    
    if analysis_dir is None:
        analysis_dir = f'analysis_{video_name}'
        
    if video_out_path is None:
        video_out_path = f'{video_name}_out.mp4'

    # Create necessary directories
    Path(faces_output_dir).mkdir(parents=True, exist_ok=True)
    Path(analysis_dir).mkdir(parents=True, exist_ok=True)

    # Get video properties
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

    #cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS),
    #                        (frame.shape[1], frame.shape[0]))

    model = YOLO("yolo11n.pt")

    # Store the track history
    track_history = defaultdict(lambda: [])

    # Colors for visualization
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

    # Analytics dictionaries
    faces_per_id = {}
    person_data = defaultdict(lambda: analysis.create_default_person_data(video_name))

    # Create video statistics object
    video_stats = analysis.create_video_stats(
        video_name=video_name,
        video_resolution=video_resolution,
        fps=fps,
        total_frames=total_frames,
        video_duration=video_duration,
        crime_category=crime_category
    )

    # Initialize frame counter
    frame_count = 0

    # Calculate total frames for progress bar
    pbar = tqdm(total=total_frames, desc=f"Processing {video_name}")

    start_time = time.time()
    max_processing_time = 180  # 3 minutes in seconds

    # Start processing the video
    while ret:
        # Check if processing time exceeded 3 minutes
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
        
        # Estimate lighting conditions occasionally (every 50 frames)
        if frame_count % 50 == 0:
            lighting = analysis.estimate_lighting(frame)
            # Update if we have a valid estimate (not unknown)
            if lighting != "unknown":
                video_stats['lighting_condition'] = lighting
        
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, verbose=False, persist=True, classes=0, iou=0.4, conf=MODEL_TRACK_CONFIDENCE,)
        
        for result in results:
            # Continue only if there are detected boxes with tracking IDs
            if result.boxes and hasattr(result.boxes, 'id') and result.boxes.id is not None:
                video_stats['valid_frames'] += 1
                
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2 format
                track_ids = result.boxes.id.int().cpu().tolist()
                confidences = result.boxes.conf.cpu().numpy()
                
                # Process each detection
                for i, (box, track_id, confidence) in enumerate(zip(boxes, track_ids, confidences)):
                    x1, y1, x2, y2 = box
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    if confidence > detection_threshold:
                        # Set person_id once
                        if person_data[track_id]['person_id'] is None:
                            person_data[track_id]['person_id'] = track_id
                        
                        # Calculate bounding box dimensions
                        width = x2 - x1
                        height = y2 - y1
                        
                        # Update track history for visualization
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        track_history[track_id].append((float(center_x), float(center_y)))
                        if len(track_history[track_id]) > 30:  # retain 30 tracks for 30 frames
                            track_history[track_id].pop(0)
                        
                        # Update analytics data
                        if person_data[track_id]['first_seen_frame'] is None:
                            person_data[track_id]['first_seen_frame'] = frame_count
                        person_data[track_id]['last_seen_frame'] = frame_count
                        person_data[track_id]['frames_detected'] += 1
                        person_data[track_id]['frame_ids'].append(frame_count)
                        person_data[track_id]['confidence_values'].append(float(confidence))
                        person_data[track_id]['bounding_boxes'].append([int(x1), int(y1), int(x2), int(y2)])
                        person_data[track_id]['positions'].append([float(center_x), float(center_y)])
                        
                        # Process face detection if we haven't saved 5 faces for this ID yet with confidence > 90%
                        if len(person_data[track_id]['faces_detected']) < 5:
                            # Extract the person region
                            person_region = frame[y1:y2, x1:x2]
                            
                            if person_region.size > 0:  # Check if region is valid
                                # Save temporary frame for DeepFace
                                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                                    temp_frame_path = temp_file.name
                                    cv2.imwrite(temp_frame_path, person_region)
                                
                                try:
                                    # Detect faces in the person region
                                    resultado = DeepFace.extract_faces(
                                        img_path=temp_frame_path,
                                        detector_backend='retinaface',
                                        enforce_detection=False,
                                        expand_percentage=30,
                                        align=True
                                    )
                                    
                                    # Only process the first face (index 0) if available
                                    if resultado and len(resultado) > 0:
                                        face = resultado[0]  # Get only the first face
                                        face_img = (face['face'] * 255).astype(np.uint8)
                                        face_confidence = face.get('confidence', 1.0)
                                        facial_area = face.get('facial_area', [0, 0, 0, 0])
                                        
                                        # Process face using the new function
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
                                
                                # Clean up temporary file
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

    # Consolidate demographic data from all faces for each person
    person_data = consolidate_demographics(person_data)
    
    # Process data and generate analysis results
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
    """
    Process all videos in the dataset directory structure.
    
    Args:
        dataset_dir (str): Path to the dataset directory with class subdirectories
        output_base_dir (str): Base directory for outputs
        
    Returns:
        dict: Summary of processing results
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        return None
    
    # Create output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Checkpoint file to track processed videos
    checkpoint_file = os.path.join(output_base_dir, "checkpoint.json")
    processed_videos = {}
    
    # Load checkpoint if exists
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                processed_videos = json.load(f)
            print(f"Loaded checkpoint with {sum(len(videos) for videos in processed_videos.values())} processed videos")
        except Exception as e:
            print(f"Error loading checkpoint file: {str(e)}")
            processed_videos = {}
    
    results = {}
    
    # Get all class directories (subdirectories)
    class_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()], key=lambda x: x.name)
    
    # Count total videos for progress reporting
    total_videos = 0
    class_video_counts = {}
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        video_files = []
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            video_files.extend(list(class_dir.glob(f"*{ext}")))
        
        # Sort video files alphabetically by name
        video_files = sorted(video_files, key=lambda x: x.name)
        
        class_video_counts[class_name] = len(video_files)
        total_videos += len(video_files)
    
    print(f"Found {total_videos} videos in {len(class_dirs)} classes")
    
    # Initialize progress tracking
    videos_processed = 0
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        print(f"\nProcessing class: {class_name} ({class_video_counts[class_name]} videos)")
        
        # Create class-specific output directories
        class_output_dir = os.path.join(output_base_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)
        
        # Get all video files in this class directory
        video_files = []
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            video_files.extend(list(class_dir.glob(f"*{ext}")))
        
        # Sort video files alphabetically by name
        video_files = sorted(video_files, key=lambda x: x.name)
        
        if not video_files:
            print(f"  No video files found in {class_dir}")
            continue
        
        class_results = []
        
        # Initialize class in processed_videos if not exists
        if class_name not in processed_videos:
            processed_videos[class_name] = []
        
        # Process each video
        for video_idx, video_file in enumerate(video_files):
            video_path = str(video_file)
            video_name = video_file.stem
            
            # Check if video has already been processed
            if video_name in processed_videos[class_name]:
                print(f"Skipping already processed video: {video_name} [{video_idx+1}/{len(video_files)}]")
                videos_processed += 1
                continue
            
            print(f"\nProcessing video: {video_name} [{video_idx+1}/{len(video_files)}] - Overall progress: {videos_processed}/{total_videos}")
            
            try:
                # Create video-specific output directories
                video_output_dir = os.path.join(class_output_dir, video_name)
                faces_dir = os.path.join(video_output_dir, "faces")
                analysis_dir = os.path.join(video_output_dir, "analysis")
                
                # Process the video
                result = process_video(
                    input_video_path=video_path,
                    faces_output_dir=faces_dir,
                    analysis_dir=analysis_dir,
                    crime_category=class_name,
                    video_out_path=os.path.join(video_output_dir, f"{video_name}_out.mp4")
                )
                
                if result:
                    class_results.append(result)
                    
                    # Add to processed videos list and save checkpoint
                    processed_videos[class_name].append(video_name)
                    
                    # Save checkpoint after each video
                    try:
                        with open(checkpoint_file, 'w') as f:
                            json.dump(processed_videos, f)
                    except Exception as e:
                        print(f"Warning: Failed to save checkpoint: {str(e)}")
                    
            except Exception as e:
                print(f"Error processing video {video_name}: {str(e)}")
            
            videos_processed += 1
        
        # Store class results
        results[class_name] = class_results
    
    # Final checkpoint save
    try:
        with open(checkpoint_file, 'w') as f:
            json.dump(processed_videos, f)
    except Exception as e:
        print(f"Warning: Failed to save final checkpoint: {str(e)}")
    
    return results 

def draw_boxes(frame, boxes, track_ids, confidences, track_history, face_data=None):
    """
    Desenha caixas delimitadoras para pessoas e faces no frame.
    
    Args:
        frame (ndarray): Frame a ser modificado
        boxes (list): Lista de bounding boxes no formato [x1, y1, x2, y2]
        track_ids (list): Lista com IDs de rastreamento correspondentes
        confidences (list): Lista de valores de confiança
        track_history (dict): Histórico de posições para cada ID
        face_data (dict, optional): Dados de faces detectadas por ID
    
    Returns:
        ndarray: Frame com as caixas delimitadoras desenhadas
    """
    # Create a copy of the frame
    vis_frame = frame.copy()
    
    # Cores para diferentes pessoas (até 10 cores diferentes, depois recicla)
    colors = [
        (0, 0, 255),     # Vermelho
        (0, 255, 0),     # Verde
        (255, 0, 0),     # Azul
        (255, 255, 0),   # Ciano
        (255, 0, 255),   # Magenta
        (0, 255, 255),   # Amarelo
        (128, 0, 255),   # Roxo
        (255, 128, 0),   # Laranja
        (0, 255, 128),   # Verde-água
        (128, 128, 255)  # Rosa
    ]
    
    # Draw each person bounding box
    for i, (box, track_id, confidence) in enumerate(zip(boxes, track_ids, confidences)):
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Use color based on track_id
        color = colors[track_id % len(colors)]
        
        # Draw person bounding box
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
        
        # Add track ID and confidence
        cv2.putText(vis_frame, f"ID:{track_id} {confidence:.2f}", 
                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw track history (last 30 positions)
        points = track_history[track_id]
        for j in range(1, len(points)):
            if points[j-1] is None or points[j] is None:
                continue
            # Get points
            p1 = (int(points[j-1][0]), int(points[j-1][1]))
            p2 = (int(points[j][0]), int(points[j][1]))
            cv2.line(vis_frame, p1, p2, color, 2)
            
        # Draw faces if available
        if face_data and track_id in face_data:
            for face_info in face_data.get(track_id, {}).get('faces_detected', []):
                # Only process if we have a face bounding box
                if 'bbox' in face_info and face_info['bbox']:
                    fx, fy, fw, fh = face_info['bbox']
                    # Draw face bounding box with green color
                    cv2.rectangle(vis_frame, 
                                 (fx, fy), 
                                 (fx + fw, fy + fh), 
                                 (0, 255, 0), 2)
                    # Add face confidence
                    face_conf = face_info.get('confidence', 0)
                    cv2.putText(vis_frame, f"Face: {face_conf:.2f}", 
                               (fx, fy-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    return vis_frame

def process_face(person_data, track_id, face_img, face_confidence, faces_output_dir, frame_count, threshold=0.9, facial_area=None, x1=0, y1=0):
    """
    Processa uma face detectada e salva até 5 faces que estejam acima do threshold.
    
    Args:
        person_data (dict): Dicionário com dados das pessoas detectadas
        track_id (int): ID de rastreamento da pessoa
        face_img (ndarray): Imagem da face detectada
        face_confidence (float): Valor de confiança da face detectada
        faces_output_dir (str): Diretório para salvar as faces
        frame_count (int): Número do frame atual
        threshold (float): Limite mínimo de confiança para considerar a face
        facial_area (list/tuple/dict, optional): Área facial relativa à pessoa
        x1, y1 (int, optional): Posição superior esquerda da pessoa
        
    Returns:
        bool: Indica se a face foi processada e salva
        str: Caminho da face salva ou None
    """
    # Verificar se a confiança está acima do limite mínimo
    if face_confidence < threshold:
        return False, None
    
    # Verificar se já temos 5 faces para esta pessoa
    if len(person_data[track_id]['faces_detected']) >= 5:
        return False, None  # Já temos 5 faces, não salva mais
    
    # Se chegou aqui, vamos salvar a face
    person_data[track_id]['face_confidence_values'].append(face_confidence)
    
    # Create person-specific folder
    person_folder = os.path.join(faces_output_dir, f'person_{track_id}')
    Path(person_folder).mkdir(parents=True, exist_ok=True)
    
    # Determine o índice da face atual
    face_index = len(person_data[track_id]['faces_detected'])
    
    # Save face with unique name and confidence
    face_path = os.path.join(person_folder, 
                           f'face_{face_index}_conf_{face_confidence:.2f}.jpg')
    
    # Converter imagem de RGB para BGR antes de salvar (DeepFace retorna RGB, OpenCV espera BGR)
    if len(face_img.shape) == 3 and face_img.shape[2] == 3:  # Verifica se é colorida e tem 3 canais
        face_img_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
    else:
        face_img_bgr = face_img  # Manter como está se for grayscale ou outro formato
    
    # Salvar imagem no formato correto
    cv2.imwrite(face_path, face_img_bgr)
    
    # Calculate absolute face position for visualization
    face_bbox = None
    if facial_area:
        try:
            # Handle different formats of facial_area
            if isinstance(facial_area, dict) and all(k in facial_area for k in ['x', 'y', 'w', 'h']):
                fx, fy, fw, fh = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
            elif isinstance(facial_area, (list, tuple)) and len(facial_area) >= 4:
                fx, fy, fw, fh = facial_area[:4]
            else:
                fx, fy, fw, fh = 0, 0, 0, 0
                
            # Convert to absolute coordinates
            abs_fx = x1 + fx
            abs_fy = y1 + fy
            face_bbox = [int(abs_fx), int(abs_fy), int(fw), int(fh)]
        except Exception as e:
            print(f"Error calculating face coordinates: {e}")
    
    # Create face data dictionary
    face_data = {
        'path': face_path,
        'confidence': face_confidence,
        'frame_id': frame_count,
        'bbox': face_bbox,  # Store bounding box for visualization
        'demographics': {}  # Placeholder for demographic data
    }
    
    # Add face to the list of detected faces
    person_data[track_id]['faces_detected'].append(face_data)
    
    # Process demographics for each face
    try_extract_demographics(person_data, track_id, face_path, face_index)
    
    return True, face_path

def try_extract_demographics(person_data, track_id, face_path, face_index):
    """
    Tenta extrair informações demográficas de uma face detectada.
    
    Args:
        person_data (dict): Dicionário com dados das pessoas detectadas
        track_id (int): ID de rastreamento da pessoa
        face_path (str): Caminho do arquivo da face
        face_index (int): Índice da face na lista de faces detectadas
    """
    try:
        # Analyze demographics using DeepFace
        demog = DeepFace.analyze(
            img_path=face_path,
            actions=['race', 'gender', 'age'],
            enforce_detection=False,
            silent=True
        )
        
        if demog and len(demog) > 0:
            # Extract demographics from first result
            demog_result = demog[0]
            face_demographics = {}
            
            # Store face confidence
            face_demographics['face_confidence'] = demog_result.get('face_confidence', 0)
            
            # Store complete race dictionary instead of just dominant race
            if 'race' in demog_result and isinstance(demog_result['race'], dict):
                face_demographics['race_scores'] = demog_result['race'].copy()
                # Still calculate dominant race for convenience
                dominant_race = max(demog_result['race'].items(), key=lambda x: x[1])
                face_demographics['dominant_race'] = dominant_race[0]
                face_demographics['dominant_race_confidence'] = dominant_race[1]
            
            # Get gender
            if 'gender' in demog_result or 'dominant_gender' in demog_result:
                gender_key = 'dominant_gender' if 'dominant_gender' in demog_result else 'gender'
                face_demographics['gender'] = demog_result[gender_key]
                if 'gender' in demog_result and isinstance(demog_result['gender'], dict):
                    face_demographics['gender_scores'] = demog_result['gender'].copy()
                    face_demographics['gender_confidence'] = demog_result['gender'].get(face_demographics['gender'], 0)
                else:
                    face_demographics['gender_confidence'] = demog_result.get('gender_confidence', 0)
            
            # Get age
            if 'age' in demog_result:
                try:
                    age = float(demog_result['age'])
                    face_demographics['age'] = age
                    # Define age range buckets
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
            
            # Store demographics in the face data
            person_data[track_id]['faces_detected'][face_index]['demographics'] = face_demographics
            
            # We'll consolidate all face demographics at the end, so we're not updating
            # the person demographics here anymore
            
    except Exception as e:
        print(f"Error analyzing demographics for ID {track_id}, face {face_index}: {str(e)}")

def consolidate_demographics(person_data):
    """
    Consolida os dados demográficos de todas as faces detectadas para cada pessoa,
    calculando raça dominante e idade dominante com base na confiança de cada análise.
    
    Args:
        person_data (dict): Dicionário com dados das pessoas detectadas
        
    Returns:
        dict: Dicionário atualizado com dados consolidados
    """
    for track_id, data in person_data.items():
        faces = data.get('faces_detected', [])
        
        if not faces:
            continue
            
        # Initialize consolidated demographic data
        consolidated = {
            'race_scores': {},
            'age_sum': 0,
            'age_count': 0,
            'age_ranges': {},
            'gender_counts': {},
            'total_confidence': 0,
            'face_count': len(faces)
        }
        
        # Aggregate data from all faces
        for face in faces:
            demographics = face.get('demographics', {})
            face_confidence = demographics.get('face_confidence', 0)
            
            # Skip faces with low confidence
            if face_confidence < 0.5:
                consolidated['face_count'] -= 1
                continue
                
            consolidated['total_confidence'] += face_confidence
            
            # Aggregate race scores weighted by face confidence
            if 'race_scores' in demographics and isinstance(demographics['race_scores'], dict):
                for race, score in demographics['race_scores'].items():
                    if race not in consolidated['race_scores']:
                        consolidated['race_scores'][race] = 0
                    consolidated['race_scores'][race] += score * face_confidence
            
            # Aggregate age data
            if 'age' in demographics and demographics['age'] is not None:
                consolidated['age_sum'] += demographics['age'] * face_confidence
                consolidated['age_count'] += face_confidence
                
                # Count age ranges
                if 'age_range' in demographics and demographics['age_range'] != 'unknown':
                    age_range = demographics['age_range']
                    if age_range not in consolidated['age_ranges']:
                        consolidated['age_ranges'][age_range] = 0
                    consolidated['age_ranges'][age_range] += face_confidence
            
            # Count genders
            if 'gender' in demographics and demographics['gender']:
                gender = demographics['gender']
                if gender not in consolidated['gender_counts']:
                    consolidated['gender_counts'][gender] = 0
                consolidated['gender_counts'][gender] += face_confidence
        
        # Skip if no valid faces
        if consolidated['face_count'] == 0:
            continue
            
        # Determine dominant race
        if consolidated['race_scores']:
            dominant_race = max(consolidated['race_scores'].items(), key=lambda x: x[1])
            data['demographics']['dominant_race'] = dominant_race[0]
            data['demographics']['dominant_race_confidence'] = dominant_race[1] / consolidated['total_confidence'] if consolidated['total_confidence'] > 0 else 0
            data['demographics']['race_scores'] = {k: v / consolidated['total_confidence'] for k, v in consolidated['race_scores'].items()}
        
        # Calculate average age
        if consolidated['age_count'] > 0:
            data['demographics']['age'] = consolidated['age_sum'] / consolidated['age_count']
            
            # Determine dominant age range
            if consolidated['age_ranges']:
                dominant_age_range = max(consolidated['age_ranges'].items(), key=lambda x: x[1])
                data['demographics']['age_range'] = dominant_age_range[0]
                data['demographics']['age_range_confidence'] = dominant_age_range[1] / consolidated['total_confidence'] if consolidated['total_confidence'] > 0 else 0
        
        # Determine dominant gender
        if consolidated['gender_counts']:
            dominant_gender = max(consolidated['gender_counts'].items(), key=lambda x: x[1])
            data['demographics']['gender'] = dominant_gender[0]
            data['demographics']['gender_confidence'] = dominant_gender[1] / consolidated['total_confidence'] if consolidated['total_confidence'] > 0 else 0
        
        # Add summary of face analysis
        data['demographics']['analyzed_faces'] = consolidated['face_count']
        data['demographics']['total_confidence'] = consolidated['total_confidence']
    
    return person_data 