import os
import json
from collections import defaultdict
from datetime import datetime
import numpy as np
import cv2
from pathlib import Path

def estimate_lighting(frame):
    """
    Estimate lighting conditions in a frame.
    
    Args:
        frame: Numpy array containing image frame
        
    Returns:
        str: Lighting condition description
    """
    if frame is None or frame.size == 0:
        return "unknown"
    
    
    # Convert to grayscale and calculate mean brightness
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    
    # Determine lighting condition based on brightness
    if mean_brightness < 40:
        return "very_dark"
    elif mean_brightness < 80:
        return "dark"
    elif mean_brightness < 120:
        return "moderate"
    elif mean_brightness < 200:
        return "bright"
    else:
        return "very_bright"

def post_process_data(person_data, video_stats, video_name, analysis_dir):
    """
    Process collected data and generate analysis results.
    
    Args:
        person_data: Dictionary containing data for each detected person
        video_stats: Dictionary containing video statistics
        video_name: Name of the video being processed
        analysis_dir: Directory to save analysis results
        
    Returns:
        str: Path to the saved JSON file
    """
    # Calculate aggregate video statistics
    video_stats['total_persons_detected'] = len(person_data)
    video_stats['total_faces_detected'] = sum(len(data['faces_detected']) for data in person_data.values())
    
    # Calculate average bounding box sizes
    for person_id, data in person_data.items():
        if data['bounding_boxes']:
            widths = []
            heights = []
            for box in data['bounding_boxes']:
                widths.append(box[2] - box[0])
                heights.append(box[3] - box[1])
            
            avg_width = sum(widths) / len(widths)
            avg_height = sum(heights) / len(heights)
            data['avg_bounding_box'] = {
                'width': avg_width,
                'height': avg_height,
                'area': avg_width * avg_height
            }
    
    # Calculate video-level demographic distributions
    race_count = defaultdict(int)
    gender_count = defaultdict(int)
    age_range_count = defaultdict(int)
    
    for person_id, data in person_data.items():
        # Handle race
        if data['demographics']['race']:
            race_key = data['demographics']['race']
            if isinstance(race_key, dict):
                # Convert dict to string representation
                race_key = max(race_key.items(), key=lambda x: x[1])[0]
            race_count[race_key] += 1
        
        # Handle gender
        if data['demographics']['gender']:
            gender_key = data['demographics']['gender']
            if isinstance(gender_key, dict):
                # Convert dict to string representation
                gender_key = max(gender_key.items(), key=lambda x: x[1])[0]
            gender_count[gender_key] += 1
        
        # Handle age
        if 'age_range' in data['demographics'] and data['demographics']['age_range']:
            age_range_key = data['demographics']['age_range']
            if isinstance(age_range_key, dict):
                # Unlikely but just to be safe
                age_range_key = str(age_range_key)
            age_range_count[age_range_key] += 1
    
    # Calculate distributions as percentages
    total_with_race = sum(race_count.values())
    if total_with_race > 0:
        for race, count in race_count.items():
            video_stats['demographics']['race_distribution'][race] = count / total_with_race
        # Find dominant race
        if race_count:
            video_stats['demographics']['dominant_race'] = max(race_count.items(), key=lambda x: x[1])[0]
    
    total_with_gender = sum(gender_count.values())
    if total_with_gender > 0:
        for gender, count in gender_count.items():
            video_stats['demographics']['gender_distribution'][gender] = count / total_with_gender
        # Find dominant gender
        if gender_count:
            video_stats['demographics']['dominant_gender'] = max(gender_count.items(), key=lambda x: x[1])[0]
    
    total_with_age_range = sum(age_range_count.values())
    if total_with_age_range > 0:
        for age_range, count in age_range_count.items():
            video_stats['demographics']['age_distribution'][age_range] = count / total_with_age_range
        # Find dominant age range
        if age_range_count:
            video_stats['demographics']['dominant_age_range'] = max(age_range_count.items(), key=lambda x: x[1])[0]
    
    # Prepare data for JSON export
    export_data = {
        'video_stats': video_stats,
        'persons': {}
    }
    
    # Convert person data to serializable format for JSON
    for person_id, data in person_data.items():
        # Ensure demographics dictionary has all required fields with default values
        demographics = data.get('demographics', {})
        if 'race' not in demographics:
            demographics['race'] = None
        if 'gender' not in demographics:
            demographics['gender'] = None
        if 'age' not in demographics:
            demographics['age'] = None
        if 'age_range' not in demographics:
            demographics['age_range'] = None
        if 'race_confidence' not in demographics:
            demographics['race_confidence'] = 0
        if 'gender_confidence' not in demographics:
            demographics['gender_confidence'] = 0
        if 'age_confidence' not in demographics:
            demographics['age_confidence'] = 0
            
        # Convert numpy values to standard Python types and ensure no dict values
        export_data['persons'][str(person_id)] = {
            'video_id': data['video_id'],
            'person_id': data['person_id'],
            'frames_detected': data['frames_detected'],
            'first_seen_frame': data['first_seen_frame'],
            'last_seen_frame': data['last_seen_frame'],
            'frame_ids': data['frame_ids'],
            'confidence_values': [float(c) for c in data['confidence_values']],
            'avg_confidence': sum(data['confidence_values']) / len(data['confidence_values']) if data['confidence_values'] else 0,
            'avg_bounding_box': data.get('avg_bounding_box', {}),
            'faces_detected': data['faces_detected'],
            'face_confidence_values': [float(c) for c in data['face_confidence_values']],
            'avg_face_confidence': sum(data['face_confidence_values']) / len(data['face_confidence_values']) if data['face_confidence_values'] else 0,
            'best_face_path': data['best_face_path'],
            'best_face_score': float(data['best_face_score']),
        }
    
    # Save to JSON
    json_output_path = os.path.join(analysis_dir, f'{video_name}_analysis.json')
    with open(json_output_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    return json_output_path

def create_default_person_data(video_name):
    """
    Create default structure for person data.
    
    Args:
        video_name: Name of the video
        
    Returns:
        dict: Default person data structure
    """
    return {
        'video_id': video_name,
        'person_id': None,
        'first_seen_frame': None,
        'last_seen_frame': None,
        'frames_detected': 0,
        'frame_ids': [],
        'confidence_values': [],
        'bounding_boxes': [],
        'positions': [],
        'faces_detected': [],
        'face_confidence_values': [],
        'demographics': {
            'race': None,
            'gender': None,
            'age': None,
            'age_range': None,  # Ensure this field always exists
            'race_confidence': 0,
            'gender_confidence': 0,
            'face_confidence': 0
        },
        'best_face_path': None,
        'best_face_score': 0
    }

def create_video_stats(video_name, video_resolution, fps, total_frames, video_duration, crime_category="unknown"):
    """
    Create default structure for video statistics.
    
    Args:
        video_name: Name of the video
        video_resolution: Resolution of the video
        fps: Frames per second
        total_frames: Total number of frames
        video_duration: Duration in seconds
        crime_category: Category of crime in the video
        
    Returns:
        dict: Default video statistics structure
    """
    return {
        'video_id': video_name,
        'resolution': video_resolution,
        'fps': fps,
        'duration': video_duration,
        'total_frames': total_frames,
        'valid_frames': 0,
        'total_persons_detected': 0,
        'total_faces_detected': 0,
        'crime_category': crime_category,
        'demographics': {
            'race_distribution': {},
            'gender_distribution': {},
            'age_distribution': {},
            'dominant_race': None,
            'dominant_gender': None,
            'dominant_age_range': None
        },
        'lighting_condition': "unknown",
        'processing_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    } 