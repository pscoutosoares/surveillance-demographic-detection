import json
import glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

class VideoAnalysisStudy:
    def __init__(self, data_dir="resultados", output_dir="analysis_results"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        self.tables_dir = self.output_dir / "tables"
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        
        self.all_videos = []
        self.all_persons = []
        self.crime_categories = set()
        self.category_data = defaultdict(list)
        
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = plt.cm.tab10.colors
        
    def find_analysis_files(self):
        return glob.glob(str(self.data_dir) + "/**/*_analysis.json", recursive=True)
    
    def _fill_demographics(self, video_data):
        persons = video_data.get('persons', {})
        demographics = {
            'race_distribution': {},
            'gender_distribution': {},
            'age_distribution': {},
            'dominant_race': None,
            'dominant_gender': None,
            'dominant_age_range': None
        }

        race_counts = Counter()
        gender_counts = Counter()
        age_counts = Counter()

        for person_id, person_data in persons.items():
            faces = person_data.get('faces_detected', [])
            for face in faces:
                face_demographics = face.get('demographics', {})
                
                race_scores = face_demographics.get('race_scores', {})
                for race, score in race_scores.items():
                    race_counts[race] += score
                
                gender_scores = face_demographics.get('gender_scores', {})
                for gender, score in gender_scores.items():
                    gender_counts[gender] += score
                
                age_range = face_demographics.get('age_range')
                if age_range:
                    age_counts[age_range] += 1

        total_faces = sum(race_counts.values())
        if total_faces > 0:
            demographics['race_distribution'] = {race: count/total_faces for race, count in race_counts.items()}
            demographics['gender_distribution'] = {gender: count/total_faces for gender, count in gender_counts.items()}
            demographics['age_distribution'] = {age: count/total_faces for age, count in age_counts.items()}

            if race_counts:
                demographics['dominant_race'] = max(race_counts.items(), key=lambda x: x[1])[0]
            if gender_counts:
                demographics['dominant_gender'] = max(gender_counts.items(), key=lambda x: x[1])[0]
            if age_counts:
                demographics['dominant_age_range'] = max(age_counts.items(), key=lambda x: x[1])[0]

        return demographics

    def load_data(self):
        """Load data from all analysis files."""
        print("Loading data from analysis files...")
        analysis_files = self.find_analysis_files()
        print(f"Found {len(analysis_files)} analysis files")
        
        for file_path in analysis_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                video_stats = data.get('video_stats', {})
                video_id = video_stats.get('video_id', 'unknown')
                category = video_stats.get('crime_category', 'unknown')
                persons = data.get('persons', {})
                
                video_stats['demographics'] = self._fill_demographics(data)
                
                self.all_videos.append(video_stats)
                self.crime_categories.add(category)
                self.category_data[category].append(video_stats)
                
                for person_id, person_data in persons.items():
                    person_data['video_id'] = video_id
                    person_data['category'] = category
                    self.all_persons.append(person_data)
            
            except Exception as e:
                print(f"Error loading file {file_path}: {str(e)}")
        
        print(f"Loaded data from {len(self.all_videos)} videos across {len(self.crime_categories)} categories")
        print(f"Total persons data: {len(self.all_persons)}")
        
        self.videos_df = pd.DataFrame(self.all_videos)
        self.persons_df = pd.DataFrame(self.all_persons)
        
        self.videos_df.to_csv(self.tables_dir / "videos_data.csv", index=False)
        self.persons_df.to_csv(self.tables_dir / "persons_data.csv", index=False)
        
        return len(self.all_videos) > 0

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze video detection results')
    parser.add_argument('--data_dir', default='resultados', help='Directory containing analysis JSON files')
    parser.add_argument('--output_dir', default='analysis_results', help='Directory to save analysis results')
    
    args = parser.parse_args()
    