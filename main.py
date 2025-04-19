import os
import random
import tempfile
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from deepface import DeepFace

from Tracker import Tracker

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process video for person tracking and face detection')
parser.add_argument('--input', type=str, default='vid_1.mp4', help='Path to input video')
#args = parser.parse_args()

# Input video path
input_video_path = 'vid_1.mp4'

# Get video name without extension
video_name = os.path.splitext(os.path.basename(input_video_path))[0]

video_out_path = f'{video_name}_out.mp4'
faces_output_dir = f'detected_faces_{video_name}'

# Create directory for saving faces if it doesn't exist
Path(faces_output_dir).mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(input_video_path)
ret, frame = cap.read()

cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS),
                          (frame.shape[1], frame.shape[0]))

model = YOLO("yolo11n.pt")

tracker = Tracker()

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

detection_threshold = 0.7
frame_interval = 2
face_confidence_threshold = 0.8  # 80% confidence threshold for face detection

# Dictionary to keep track of how many faces we've saved for each ID
faces_per_id = {}

# Calculate total frames for progress bar
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
pbar = tqdm(total=total_frames, desc="Processing video")

while ret:
    for _ in range(frame_interval - 1):
        ret, frame = cap.read()
        if not ret:
            break
        pbar.update(1)
    
    if not ret:
        break
        
    results = model(frame, verbose=False, classes=0)

    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            if score > detection_threshold:
                detections.append([x1, y1, x2, y2, score])

        tracker.update(frame, detections)

        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id

            # Draw bounding box
            #cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
            
            # Process face detection if we haven't saved 5 faces for this ID yet
            if track_id not in faces_per_id or faces_per_id[track_id] < 5:
                # Extract the person region
                person_region = frame[int(y1):int(y2), int(x1):int(x2)]
                
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
                            expand_percentage=20,
                            align=True
                        )
                        
                        for i, face in enumerate(resultado):
                            face_img = (face['face'] * 255).astype(np.uint8)
                            confidence = face.get('confidence', 1.0)
                            facial_area = face.get('facial_area', [0, 0, 0, 0])

                            # Only save face if confidence is above threshold
                            if confidence >= face_confidence_threshold:
                                if isinstance(face_img, np.ndarray):
                                    # Ensure the image is in BGR format (OpenCV format)
                                    if len(face_img.shape) == 3 and face_img.shape[2] == 3:
                                        # Create person-specific folder
                                        person_folder = os.path.join(faces_output_dir, f'person_{track_id}')
                                        Path(person_folder).mkdir(parents=True, exist_ok=True)
                                        
                                        # Save face with unique name
                                        face_path = os.path.join(person_folder, f'face_{faces_per_id.get(track_id, 0)}.jpg')
                                        cv2.imwrite(face_path, face_img)
                                        
                                        # Update counter for this ID
                                        faces_per_id[track_id] = faces_per_id.get(track_id, 0) + 1
                            
                    except Exception as e:
                        print(f"Error processing face for ID {track_id}: {str(e)}")
                    
                    # Clean up temporary file
                    os.unlink(temp_frame_path)

    cap_out.write(frame)
    ret, frame = cap.read()
    pbar.update(1)

pbar.close()
cap.release()
cap_out.release()
cv2.destroyAllWindows()