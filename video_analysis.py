import os
import json
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from collections import defaultdict, Counter

class VideoAnalysisStudy:
    def __init__(self, data_dir="resultados", output_dir="analysis_results"):
        """
        Initialize the video analysis study.
        
        Args:
            data_dir: Directory containing the video analysis results
            output_dir: Directory where analysis results will be saved
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        self.tables_dir = self.output_dir / "tables"
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data structures
        self.all_videos = []
        self.all_persons = []
        self.crime_categories = set()
        self.category_data = defaultdict(list)
        
        # Set plot style
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = plt.cm.tab10.colors
        
    def find_analysis_files(self):
        """Find all analysis JSON files recursively in the data directory."""
        return glob.glob(str(self.data_dir) + "/**/*_analysis.json", recursive=True)
    
    def _fill_demographics(self, video_data):
        """Preenche os dados demográficos usando as informações disponíveis no arquivo."""
        persons = video_data.get('persons', {})
        demographics = {
            'race_distribution': {},
            'gender_distribution': {},
            'age_distribution': {},
            'dominant_race': None,
            'dominant_gender': None,
            'dominant_age_range': None
        }

        # Contadores para distribuição
        race_counts = Counter()
        gender_counts = Counter()
        age_counts = Counter()

        # Processar cada pessoa detectada
        for person_id, person_data in persons.items():
            # Extrair informações de faces detectadas
            faces = person_data.get('faces_detected', [])
            for face in faces:
                # Extrair informações demográficas da face
                face_demographics = face.get('demographics', {})
                
                # Atualizar contadores de raça
                race_scores = face_demographics.get('race_scores', {})
                for race, score in race_scores.items():
                    race_counts[race] += score
                
                # Atualizar contadores de gênero
                gender_scores = face_demographics.get('gender_scores', {})
                for gender, score in gender_scores.items():
                    gender_counts[gender] += score
                
                # Atualizar contadores de idade
                age_range = face_demographics.get('age_range')
                if age_range:
                    age_counts[age_range] += 1

        # Converter contadores para distribuição
        total_faces = sum(race_counts.values())
        if total_faces > 0:
            demographics['race_distribution'] = {race: count/total_faces for race, count in race_counts.items()}
            demographics['gender_distribution'] = {gender: count/total_faces for gender, count in gender_counts.items()}
            demographics['age_distribution'] = {age: count/total_faces for age, count in age_counts.items()}

            # Determinar valores dominantes
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
        
        # Collect data from each file
        for file_path in analysis_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                video_stats = data.get('video_stats', {})
                video_id = video_stats.get('video_id', 'unknown')
                category = video_stats.get('crime_category', 'unknown')
                persons = data.get('persons', {})
                
                # Preencher dados demográficos
                video_stats['demographics'] = self._fill_demographics(data)
                
                # Store video data
                self.all_videos.append(video_stats)
                self.crime_categories.add(category)
                self.category_data[category].append(video_stats)
                
                # Process each person
                for person_id, person_data in persons.items():
                    # Add video and category info to person data
                    person_data['video_id'] = video_id
                    person_data['category'] = category
                    self.all_persons.append(person_data)
            
            except Exception as e:
                print(f"Error loading file {file_path}: {str(e)}")
        
        print(f"Loaded data from {len(self.all_videos)} videos across {len(self.crime_categories)} categories")
        print(f"Total persons data: {len(self.all_persons)}")
        
        # Convert to DataFrames for easier analysis
        self.videos_df = pd.DataFrame(self.all_videos)
        self.persons_df = pd.DataFrame(self.all_persons)
        
        # Save raw data for reference
        self.videos_df.to_csv(self.tables_dir / "videos_data.csv", index=False)
        self.persons_df.to_csv(self.tables_dir / "persons_data.csv", index=False)
        
        return len(self.all_videos) > 0
    
    def analyze_crime_categories(self):
        """
        Analyze basic statistics across crime categories.
        """
        print("Analyzing crime categories...")
        
        # Create a DataFrame with category-level statistics
        stats_by_category = []
        
        for category in self.crime_categories:
            videos = self.category_data[category]
            num_videos = len(videos)
            
            # Calculate aggregate statistics
            total_persons = sum(v.get('total_persons_detected', 0) for v in videos)
            total_faces = sum(v.get('total_faces_detected', 0) for v in videos)
            total_frames = sum(v.get('total_frames', 0) for v in videos)
            valid_frames = sum(v.get('valid_frames', 0) for v in videos)
            total_duration = sum(v.get('duration', 0) for v in videos)
            
            # Calculate derived metrics
            avg_persons_per_video = total_persons / num_videos if num_videos else 0
            avg_faces_per_video = total_faces / num_videos if num_videos else 0
            person_detection_rate = total_faces / total_persons if total_persons else 0
            frame_detection_rate = valid_frames / total_frames if total_frames else 0
            person_density = total_persons / total_duration if total_duration else 0
            
            # Store statistics
            stats_by_category.append({
                'category': category,
                'num_videos': num_videos,
                'total_persons': total_persons,
                'total_faces': total_faces,
                'avg_persons_per_video': avg_persons_per_video,
                'avg_faces_per_video': avg_faces_per_video,
                'person_detection_rate': person_detection_rate,
                'frame_detection_rate': frame_detection_rate,
                'person_density': person_density,
                'avg_duration': total_duration / num_videos if num_videos else 0
            })
        
        # Convert to DataFrame
        self.category_stats_df = pd.DataFrame(stats_by_category)
        self.category_stats_df.to_csv(self.tables_dir / "category_statistics.csv", index=False)
        
        # Create visualizations
        self._create_category_comparison_plots()
        
        return True
    
    def _create_category_comparison_plots(self):
        """Create visualizations comparing key metrics across crime categories."""
        if self.category_stats_df.empty:
            print("No category statistics available for visualization")
            return
        
        # 1. Bar plot for persons and faces per video
        plt.figure(figsize=(12, 8))
        
        # Get categories and data
        categories = self.category_stats_df['category'].tolist()
        persons_per_video = self.category_stats_df['avg_persons_per_video'].tolist()
        faces_per_video = self.category_stats_df['avg_faces_per_video'].tolist()
        
        # Set up bar positions
        x = np.arange(len(categories))
        width = 0.35
        
        # Create bars
        fig, ax = plt.subplots(figsize=(12, 8))
        bars1 = ax.bar(x - width/2, persons_per_video, width, label='Persons Detected', color=self.colors[0])
        bars2 = ax.bar(x + width/2, faces_per_video, width, label='Faces Detected', color=self.colors[1])
        
        # Add labels and title
        ax.set_xlabel('Crime Category', fontsize=14)
        ax.set_ylabel('Average Count per Video', fontsize=14)
        ax.set_title('Average Persons and Faces Detected by Crime Category', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend()
        
        # Add value labels on bars
        self._add_value_labels(ax, bars1)
        self._add_value_labels(ax, bars2)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "persons_faces_by_category.png", dpi=300)
        plt.close()
        
        # 2. Bar plot for detection rates
        plt.figure(figsize=(12, 8))
        
        # Get data
        person_detection_rate = self.category_stats_df['person_detection_rate'].tolist()
        frame_detection_rate = self.category_stats_df['frame_detection_rate'].tolist()
        
        # Create bars
        fig, ax = plt.subplots(figsize=(12, 8))
        bars1 = ax.bar(x - width/2, person_detection_rate, width, label='Face Detection Rate', color=self.colors[2])
        bars2 = ax.bar(x + width/2, frame_detection_rate, width, label='Frame Detection Rate', color=self.colors[3])
        
        # Add labels and title
        ax.set_xlabel('Crime Category', fontsize=14)
        ax.set_ylabel('Detection Rate', fontsize=14)
        ax.set_title('Detection Rates by Crime Category', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend()
        
        # Add value labels on bars
        self._add_value_labels(ax, bars1, fmt='{:.2f}')
        self._add_value_labels(ax, bars2, fmt='{:.2f}')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "detection_rates_by_category.png", dpi=300)
        plt.close()
        
        # 3. Heatmap for all metrics (normalized)
        plt.figure(figsize=(14, 10))
        
        # Select numeric columns to include
        numeric_cols = ['avg_persons_per_video', 'avg_faces_per_video', 
                       'person_detection_rate', 'frame_detection_rate', 
                       'person_density', 'avg_duration']
        
        # Create a pivot table with categories as rows and metrics as columns
        heatmap_data = self.category_stats_df.set_index('category')[numeric_cols].copy()
        
        # Normalize data for better visualization
        for col in heatmap_data.columns:
            if heatmap_data[col].max() > 0:
                heatmap_data[col] = heatmap_data[col] / heatmap_data[col].max()
        
        # Create heatmap
        plt.figure(figsize=(14, 8))
        sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.5)
        plt.title('Normalized Metrics Comparison Across Crime Categories', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "metrics_heatmap_by_category.png", dpi=300)
        plt.close()
        
        # 4. Person density chart
        plt.figure(figsize=(12, 8))
        
        # Sort categories by person density for better visualization
        density_data = self.category_stats_df.sort_values('person_density', ascending=False)
        
        # Create bar chart
        plt.bar(density_data['category'], density_data['person_density'], color=self.colors[4])
        plt.xlabel('Crime Category', fontsize=14)
        plt.ylabel('Person Density (persons/second)', fontsize=14)
        plt.title('Person Density by Crime Category', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.plots_dir / "person_density_by_category.png", dpi=300)
        plt.close()
        
        print("Created category comparison visualizations")
    
    def analyze_demographics(self):
        """
        Analyze demographic information (gender, age, race) across crime categories.
        """
        print("Analyzing demographics across crime categories...")
        
        # Extract demographic data from persons with faces
        demographic_data = []
        
        for person in self.all_persons:
            # Check if the person has face data with demographics
            faces = person.get('faces_detected', [])
            if faces:
                # Processar cada face detectada
                for face in faces:
                    face_demographics = face.get('demographics', {})
                    if face_demographics:
                        category = person.get('category', 'unknown')
                        
                        # Extrair distribuições
                        race_scores = face_demographics.get('race_scores', {})
                        gender_scores = face_demographics.get('gender_scores', {})
                        
                        # Encontrar valores dominantes
                        dominant_race = max(race_scores.items(), key=lambda x: x[1])[0] if race_scores else None
                        dominant_gender = max(gender_scores.items(), key=lambda x: x[1])[0] if gender_scores else None
                        
                        # Extrair informações demográficas
                        demographic_data.append({
                            'category': category,
                            'gender': dominant_gender,
                            'age': face_demographics.get('age'),
                            'age_range': face_demographics.get('age_range'),
                            'dominant_race': dominant_race,
                            'gender_distribution': gender_scores,
                            'race_distribution': race_scores
                        })
        
        # Convert to DataFrame
        self.demographics_df = pd.DataFrame(demographic_data)
        
        # Filtrar por distribuição mínima para melhorar a qualidade
        min_distribution = 0.6
        self.demographics_df = self.demographics_df[
            (self.demographics_df['gender_distribution'].apply(lambda x: max(x.values()) if x else 0) >= min_distribution) | 
            (self.demographics_df['race_distribution'].apply(lambda x: max(x.values()) if x else 0) >= min_distribution)
        ]
        
        # Save to CSV
        self.demographics_df.to_csv(self.tables_dir / "demographics_data.csv", index=False)
        
        # Create visualizations
        self._create_demographic_visualizations()
        
        return len(self.demographics_df) > 0
    
    def _create_demographic_visualizations(self):
        """Create visualizations of demographic patterns across crime categories."""
        if self.demographics_df.empty:
            print("No demographic data available for visualization")
            return
        
        # 1. Gender distribution by crime category
        gender_data = self.demographics_df.dropna(subset=['gender'])
        
        if not gender_data.empty:
            plt.figure(figsize=(12, 8))
            
            # Count gender by category
            gender_counts = pd.crosstab(
                gender_data['category'], 
                gender_data['gender'], 
                normalize='index'
            ) * 100
            
            # Create stacked bar chart
            gender_counts.plot(kind='bar', stacked=True, colormap='Set3', figsize=(12, 8))
            plt.title('Gender Distribution by Crime Category', fontsize=16)
            plt.xlabel('Crime Category', fontsize=14)
            plt.ylabel('Percentage (%)', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Gender', title_fontsize=12)
            
            # Add percentage labels
            for i, (idx, row) in enumerate(gender_counts.iterrows()):
                cumulative = 0
                for col, val in row.items():
                    if val > 5:  # Only show label if segment is large enough
                        plt.text(i, cumulative + val/2, f'{val:.1f}%', 
                                ha='center', va='center', color='black', fontweight='bold')
                    cumulative += val
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / "gender_distribution_by_category.png", dpi=300)
            plt.close()
            
            # Add a detailed explanation of the visualization
            with open(self.plots_dir / "gender_distribution_explanation.txt", 'w') as f:
                f.write("# Gender Distribution by Crime Category\n\n")
                f.write("This visualization shows the gender distribution across different crime categories.\n\n")
                f.write("## Motivation:\n")
                f.write("Understanding gender patterns in different types of crimes can reveal important sociological insights and help in developing more targeted crime prevention strategies. Different crime categories may show gender imbalances that could inform security protocols and risk assessment.\n\n")
                f.write("## Insights:\n")
                f.write("- Look for categories with higher male or female representation\n")
                f.write("- Consider how gender distribution might align with existing criminology research\n")
                f.write("- Evaluate whether surveillance systems might have gender detection biases\n")
                f.write("- Assess how this information might inform security planning\n")
        
        # 2. Age distribution by crime category
        age_data = self.demographics_df.dropna(subset=['age_range'])
        
        if not age_data.empty:
            plt.figure(figsize=(14, 10))
            
            # Define age range order
            age_order = ['under_18', '18-29', '30-44', '45-59', '60+', 'unknown']
            
            # Count age ranges by category
            age_counts = pd.crosstab(
                age_data['category'], 
                age_data['age_range'], 
                normalize='index'
            ) * 100
            
            # Reorder columns if they exist
            available_age_ranges = [age for age in age_order if age in age_counts.columns]
            age_counts = age_counts[available_age_ranges]
            
            # Create heatmap
            plt.figure(figsize=(14, 10))
            sns.heatmap(age_counts, annot=True, cmap="YlGnBu", fmt=".1f", linewidths=0.5)
            plt.title('Age Range Distribution by Crime Category (%)', fontsize=16)
            plt.xlabel('Age Range', fontsize=14)
            plt.ylabel('Crime Category', fontsize=14)
            plt.tight_layout()
            plt.savefig(self.plots_dir / "age_distribution_heatmap.png", dpi=300)
            plt.close()
            
            # Add explanation
            with open(self.plots_dir / "age_distribution_explanation.txt", 'w') as f:
                f.write("# Age Distribution by Crime Category\n\n")
                f.write("This heatmap shows the percentage distribution of age ranges across different crime categories.\n\n")
                f.write("## Motivation:\n")
                f.write("Age demographics in different crime contexts can help understand which age groups are more involved in or targeted by specific types of crimes. This information is valuable for:\n\n")
                f.write("- Tailoring security measures to address threats from specific age demographics\n")
                f.write("- Understanding victimology patterns across different crime types\n")
                f.write("- Developing age-appropriate intervention strategies\n")
                f.write("- Identifying potential vulnerabilities in surveillance systems related to age detection\n\n")
                f.write("## How to Interpret:\n")
                f.write("- Each cell shows the percentage of individuals in that age range for a particular crime category\n")
                f.write("- Darker colors indicate higher percentages\n")
                f.write("- Look for patterns where certain crime categories have significantly different age distributions\n")
            
            # Also create a violin plot for numeric age distribution
            numeric_age_data = self.demographics_df.dropna(subset=['age'])
            
            if len(numeric_age_data) > 0:
                plt.figure(figsize=(14, 10))
                sns.violinplot(x='category', y='age', data=numeric_age_data, palette='Set3')
                plt.title('Age Distribution by Crime Category', fontsize=16)
                plt.xlabel('Crime Category', fontsize=14)
                plt.ylabel('Age', fontsize=14)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(self.plots_dir / "age_violin_by_category.png", dpi=300)
                plt.close()
        
        # 3. Racial distribution by crime category
        race_data = self.demographics_df.dropna(subset=['dominant_race'])
        
        if not race_data.empty:
            plt.figure(figsize=(14, 10))
            
            # Count races by category
            race_counts = pd.crosstab(
                race_data['category'], 
                race_data['dominant_race'], 
                normalize='index'
            ) * 100
            
            # Create heatmap
            plt.figure(figsize=(14, 10))
            sns.heatmap(race_counts, annot=True, cmap="YlGnBu", fmt=".1f", linewidths=0.5)
            plt.title('Racial Distribution by Crime Category (%)', fontsize=16)
            plt.xlabel('Race', fontsize=14)
            plt.ylabel('Crime Category', fontsize=14)
            plt.tight_layout()
            plt.savefig(self.plots_dir / "race_distribution_heatmap.png", dpi=300)
            plt.close()
            
            # Add explanation
            with open(self.plots_dir / "race_distribution_explanation.txt", 'w') as f:
                f.write("# Racial Distribution by Crime Category\n\n")
                f.write("This heatmap shows the percentage distribution of detected racial categories across different crime types.\n\n")
                f.write("## Motivation:\n")
                f.write("Understanding racial patterns in surveillance footage can help identify potential biases in both detection systems and security practices. This analysis is important for:\n\n")
                f.write("- Evaluating whether facial recognition systems have different accuracy rates across racial groups\n")
                f.write("- Identifying potential biases in surveillance implementation\n")
                f.write("- Understanding demographic factors in different crime contexts\n")
                f.write("- Ensuring fair and unbiased security practices\n\n")
                f.write("## Important Considerations:\n")
                f.write("- This data reflects what the AI detected, not necessarily ground truth\n")
                f.write("- Facial recognition systems can have varying accuracy across different racial groups\n")
                f.write("- These patterns should be interpreted with caution and not used to reinforce stereotypes\n")
                f.write("- The aim should be to identify and address potential biases in surveillance systems\n")
        
        print("Created demographic visualizations")
    
    def analyze_spatial_temporal_patterns(self):
        """
        Analyze spatial and temporal patterns of persons in videos.
        """
        print("Analyzing spatial and temporal patterns...")
        
        # Extract temporal data (when persons appear in videos)
        temporal_data = []
        
        for person in self.all_persons:
            category = person.get('category', 'unknown')
            video_id = person.get('video_id', 'unknown')
            
            # Check for temporal information
            if 'first_seen_frame' in person and 'last_seen_frame' in person and person['first_seen_frame'] is not None:
                first_frame = person['first_seen_frame']
                last_frame = person['last_seen_frame']
                frames_detected = person.get('frames_detected', 0)
                
                # Calculate frame duration
                frame_duration = last_frame - first_frame if first_frame is not None and last_frame is not None else 0
                
                # Add data
                temporal_data.append({
                    'category': category,
                    'video_id': video_id,
                    'person_id': person.get('person_id', ''),
                    'first_frame': first_frame,
                    'last_frame': last_frame,
                    'duration_frames': frame_duration,
                    'frames_detected': frames_detected,
                    'detection_rate': frames_detected / frame_duration if frame_duration > 0 else 0
                })
        
        # Convert to DataFrame
        self.temporal_df = pd.DataFrame(temporal_data)
        
        # Save to CSV
        self.temporal_df.to_csv(self.tables_dir / "temporal_patterns.csv", index=False)
        
        # Create visualizations
        self._create_temporal_visualizations()
        
        # Extract spatial data (positions)
        spatial_data = []
        
        for person in self.all_persons:
            category = person.get('category', 'unknown')
            video_id = person.get('video_id', 'unknown')
            
            # Extract positions
            positions = person.get('positions', [])
            bounding_boxes = person.get('bounding_boxes', [])
            
            # Process positions if available
            if positions:
                for position in positions:
                    if isinstance(position, list) and len(position) >= 2:
                        spatial_data.append({
                            'category': category,
                            'video_id': video_id,
                            'person_id': person.get('person_id', ''),
                            'x': position[0],
                            'y': position[1]
                        })
        
        # Convert to DataFrame
        self.spatial_df = pd.DataFrame(spatial_data)
        
        # Save to CSV
        self.spatial_df.to_csv(self.tables_dir / "spatial_patterns.csv", index=False)
        
        # Create visualizations
        self._create_spatial_visualizations()
        
        return True
    
    def _create_temporal_visualizations(self):
        """Create visualizations of temporal patterns."""
        if self.temporal_df.empty:
            print("No temporal data available for visualization")
            return
        
        # 1. Person appearance duration by crime category
        plt.figure(figsize=(12, 8))
        
        # Create box plot
        sns.boxplot(x='category', y='duration_frames', data=self.temporal_df)
        plt.title('Person Appearance Duration by Crime Category', fontsize=16)
        plt.xlabel('Crime Category', fontsize=14)
        plt.ylabel('Duration (frames)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.plots_dir / "appearance_duration_boxplot.png", dpi=300)
        plt.close()
        
        # Add explanation
        with open(self.plots_dir / "temporal_patterns_explanation.txt", 'w') as f:
            f.write("# Temporal Patterns in Crime Videos\n\n")
            f.write("## Person Appearance Duration Analysis\n\n")
            f.write("This boxplot shows how long individuals typically appear in videos across different crime categories.\n\n")
            f.write("### Motivation:\n")
            f.write("Understanding the temporal dynamics of person appearances in different crime types is crucial for:\n\n")
            f.write("- Optimizing surveillance systems to capture the most critical moments\n")
            f.write("- Determining appropriate recording durations for different security scenarios\n")
            f.write("- Understanding the typical timeline and duration of different criminal activities\n")
            f.write("- Developing more effective real-time monitoring protocols\n\n")
            f.write("### Key Insights:\n")
            f.write("- Longer durations may indicate crimes where perpetrators remain in the scene longer\n")
            f.write("- Shorter durations may suggest 'hit and run' style crimes requiring faster response\n")
            f.write("- Wide variation in durations might indicate more complex or variable criminal behavior\n")
            f.write("- Categories with similar duration profiles might benefit from similar surveillance approaches\n")
        
        # 2. Violin plot for more detailed distribution
        plt.figure(figsize=(12, 8))
        
        # Create violin plot
        sns.violinplot(x='category', y='duration_frames', data=self.temporal_df)
        plt.title('Distribution of Person Appearance Duration by Crime Category', fontsize=16)
        plt.xlabel('Crime Category', fontsize=14)
        plt.ylabel('Duration (frames)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.plots_dir / "appearance_duration_violin.png", dpi=300)
        plt.close()
        
        # 3. Detection rate (how consistently persons are detected)
        plt.figure(figsize=(12, 8))
        
        # Create box plot for detection rate
        sns.boxplot(x='category', y='detection_rate', data=self.temporal_df)
        plt.title('Person Detection Rate by Crime Category', fontsize=16)
        plt.xlabel('Crime Category', fontsize=14)
        plt.ylabel('Detection Rate (frames detected / total duration)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.plots_dir / "detection_rate_boxplot.png", dpi=300)
        plt.close()
        
        print("Created temporal pattern visualizations")
    
    def _create_spatial_visualizations(self):
        """Create visualizations of spatial patterns."""
        if self.spatial_df.empty:
            print("No spatial data available for visualization")
            return
        
        # Create heatmaps of person positions by crime category
        for category in self.spatial_df['category'].unique():
            # Filter data for this category
            category_data = self.spatial_df[self.spatial_df['category'] == category]
            
            if len(category_data) < 10:  # Skip if too few data points
                continue
                
            plt.figure(figsize=(10, 8))
            
            # Create 2D histogram (heatmap)
            plt.hist2d(category_data['x'], category_data['y'], bins=20, cmap='hot')
            plt.colorbar(label='Count')
            plt.title(f'Spatial Distribution of Persons in {category} Videos', fontsize=16)
            plt.xlabel('X Position', fontsize=14)
            plt.ylabel('Y Position', fontsize=14)
            plt.tight_layout()
            plt.savefig(self.plots_dir / f"spatial_heatmap_{category}.png", dpi=300)
            plt.close()
        
        # Add explanation
        with open(self.plots_dir / "spatial_patterns_explanation.txt", 'w') as f:
            f.write("# Spatial Patterns in Crime Videos\n\n")
            f.write("These heatmaps show where individuals are most commonly detected in videos for each crime category.\n\n")
            f.write("## Motivation:\n")
            f.write("Spatial analysis of person positions in surveillance footage reveals critical insights about crime dynamics:\n\n")
            f.write("- Identifies activity hotspots where crimes typically occur within the camera frame\n")
            f.write("- Reveals common entry/exit points or movement patterns\n")
            f.write("- Helps optimize camera placement for better coverage of high-activity areas\n")
            f.write("- Shows differences in spatial patterns across different crime types\n\n")
            f.write("## How to Interpret the Heatmaps:\n")
            f.write("- Brighter/warmer colors indicate areas with higher concentrations of detected persons\n")
            f.write("- These represent the most frequent positions where people appear in the videos\n")
            f.write("- The patterns may reveal staging areas, confrontation points, or escape routes\n")
            f.write("- Different crime categories may show distinct spatial signatures\n\n")
            f.write("## Practical Applications:\n")
            f.write("- Security teams can focus attention on high-activity zones\n")
            f.write("- Camera systems can be configured to prioritize processing in hotspot areas\n")
            f.write("- Architectural and environmental design can be improved to reduce blind spots\n")
            f.write("- Security personnel can be positioned more effectively based on typical spatial patterns\n")
        
        print("Created spatial pattern visualizations")
    
    def create_summary_report(self):
        """Create a summary report of all analyses."""
        report_path = self.output_dir / "analysis_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("# Video Analysis Summary Report\n\n")
            
            f.write("## Overview\n\n")
            f.write(f"- Total videos analyzed: {len(self.all_videos)}\n")
            f.write(f"- Crime categories: {', '.join(sorted(self.crime_categories))}\n")
            f.write(f"- Total persons detected: {len(self.all_persons)}\n\n")
            
            f.write("## Key Findings\n\n")
            
            # Category statistics
            if hasattr(self, 'category_stats_df'):
                f.write("### Crime Category Analysis\n\n")
                for _, row in self.category_stats_df.iterrows():
                    category = row['category']
                    f.write(f"**{category}**:\n")
                    f.write(f"- Videos: {row['num_videos']}\n")
                    f.write(f"- Persons detected: {row['total_persons']}\n")
                    f.write(f"- Faces detected: {row['total_faces']}\n")
                    f.write(f"- Average persons per video: {row['avg_persons_per_video']:.2f}\n")
                    f.write(f"- Person density: {row['person_density']:.2f} persons/second\n\n")
            
            # Demographics
            if hasattr(self, 'demographics_df') and not self.demographics_df.empty:
                f.write("### Demographic Patterns\n\n")
                f.write("The demographic analysis reveals patterns in gender, age, and racial distributions across different crime categories. ")
                f.write("These patterns may reflect both the actual demographics of individuals involved in these scenarios and potential biases in the detection system.\n\n")
                
                f.write("See the demographic visualizations for detailed breakdowns by crime category.\n\n")
            
            # Spatial-temporal patterns
            f.write("### Spatial and Temporal Patterns\n\n")
            f.write("The analysis of when and where individuals appear in videos shows distinct patterns for different crime categories. ")
            f.write("These patterns can inform surveillance strategies, camera placement, and security protocols.\n\n")
            
            f.write("See the spatial and temporal visualizations for detailed insights.\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("Based on the analysis, consider the following recommendations:\n\n")
            f.write("1. **Optimize surveillance coverage** based on the spatial hotspots identified for each crime type\n")
            f.write("2. **Adjust recording durations** based on the typical appearance duration for different crime scenarios\n")
            f.write("3. **Address potential biases** in face detection across different demographic groups\n")
            f.write("4. **Develop targeted security protocols** for different crime categories based on their unique patterns\n")
            f.write("5. **Improve detection systems** for crime categories with lower detection rates\n\n")
            
            f.write("## Next Steps\n\n")
            f.write("To further enhance this analysis, consider:\n\n")
            f.write("1. Collecting more data for underrepresented crime categories\n")
            f.write("2. Implementing more advanced trajectory analysis to track movement patterns\n")
            f.write("3. Correlating analysis with environmental factors and time of day\n")
            f.write("4. Developing predictive models based on the patterns identified\n")
        
        print(f"Summary report created at {report_path}")
        return True
    
    def _add_value_labels(self, ax, bars, fmt='{:.1f}'):
        """Add value labels on top of bars in bar charts."""
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   fmt.format(height), ha='center', va='bottom')
    
    def run_analysis(self):
        """Run the complete analysis pipeline."""
        print("Starting comprehensive video analysis...")
        
        # Step 1: Load data
        if not self.load_data():
            print("Failed to load data. Analysis cannot continue.")
            return False
        
        # Step 2: Analyze crime categories
        self.analyze_crime_categories()
        
        # Step 3: Analyze demographics
        self.analyze_demographics()
        
        # Step 4: Analyze spatial-temporal patterns
        self.analyze_spatial_temporal_patterns()
        
        # Step 5: Create summary report
        self.create_summary_report()
        
        print(f"Analysis complete! Results saved to {self.output_dir}")
        return True

# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze video detection results')
    parser.add_argument('--data_dir', default='resultados', help='Directory containing analysis JSON files')
    parser.add_argument('--output_dir', default='analysis_results', help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = VideoAnalysisStudy(data_dir=args.data_dir, output_dir=args.output_dir)
    analyzer.run_analysis() 