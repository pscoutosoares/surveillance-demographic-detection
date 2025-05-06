#!/usr/bin/env python3
import os
import argparse
import sys
from pathlib import Path
import time

from video_processor import process_video, process_dataset

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Process videos for person tracking and face analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Processing mode')
    
    # Single video mode
    video_parser = subparsers.add_parser('video', help='Process a single video')
    video_parser.add_argument('input_path',type=str, help='Path to input video file')
    video_parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    video_parser.add_argument('--category', type=str, help='Crime category (optional)')
    
    # Dataset mode
    dataset_parser = subparsers.add_parser('dataset', help='Process a dataset of videos')
    dataset_parser.add_argument('--dataset_dir', type=str, default='dataset', help='Path to dataset directory')
    dataset_parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    dataset_parser.add_argument('--force-reprocess', action='store_true', 
                               help='Force reprocessing of all videos, ignoring checkpoints')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode is None:
        parser.print_help()
        sys.exit(1)
    
    if args.mode == 'video' and not os.path.isfile(args.input_path):
        print(f"Error: Input video file not found: {args.input_path}")
        sys.exit(1)
    
    if args.mode == 'dataset' and not os.path.isdir(args.dataset_dir):
        print(f"Error: Dataset directory not found: {args.dataset_dir}")
        sys.exit(1)
    
    return args

def main():
    """Main entry point for the application."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Record start time
    start_time = time.time()
    
    if args.mode == 'video':
        print(f"Processing video: {args.input_path}")
        
        # Create specific output directories
        video_name = os.path.splitext(os.path.basename(args.input_path))[0]
        output_dir = os.path.join(args.output_dir, args.category or "unclassified")
        video_output_dir = os.path.join(output_dir, video_name)
        faces_dir = os.path.join(video_output_dir, "faces")
        analysis_dir = os.path.join(video_output_dir, "analysis")
        video_out_path = os.path.join(video_output_dir, f"{video_name}_out.mp4")
        
        # Create base output directory
        os.makedirs(video_output_dir, exist_ok=True)
        
        # Process the video
        result = process_video(
            input_video_path=args.input_path,
            faces_output_dir=faces_dir,
            analysis_dir=analysis_dir,
            crime_category=args.category,
            video_out_path=video_out_path
        )
        
        if result:
            print(f"\nProcessed video: {args.input_path}")
            print(f"Processing time: {result['processing_time']:.2f} seconds")
            print(f"Detected {result['persons_detected']} persons with {result['faces_detected']} faces")
            print(f"Results saved to {result['json_output_path']}")
            print(f"Face images saved in {result['faces_output_dir']}")
            print(f"Video output saved as {result['video_output_path']}")
    
    elif args.mode == 'dataset':
        print(f"Processing dataset: {args.dataset_dir}")
        
        # If force reprocessing, delete checkpoint file
        if args.force_reprocess:
            checkpoint_file = os.path.join(args.output_dir, "checkpoint.json")
            if os.path.exists(checkpoint_file):
                print(f"Force reprocessing requested. Removing checkpoint file: {checkpoint_file}")
                try:
                    os.remove(checkpoint_file)
                except Exception as e:
                    print(f"Error removing checkpoint file: {str(e)}")
        
        # Process all videos in the dataset
        results = process_dataset(args.dataset_dir, args.output_dir)
        
        if results:
            # Count total videos processed
            total_videos = sum(len(videos) for videos in results.values())
            print(f"\nProcessed {total_videos} videos across {len(results)} classes")
            
            # Print summary for each class
            for class_name, class_results in results.items():
                if class_results:
                    print(f"  Class '{class_name}': {len(class_results)} videos processed")
    
    # Calculate and display total execution time
    total_time = time.time() - start_time
    minutes, seconds = divmod(total_time, 60)
    hours, minutes = divmod(minutes, 60)
    
    if hours > 0:
        print(f"Total execution time: {int(hours)} hours, {int(minutes)} minutes and {seconds:.2f} seconds")
    else:
        print(f"Total execution time: {int(minutes)} minutes and {seconds:.2f} seconds")

if __name__ == "__main__":
    main()