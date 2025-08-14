import subprocess
import os

# Configuration
VIDEO_TYPE = "shoplifting"
INPUT_DIR = rf"D:\Random Projects\Fruit Images for Object Detection\Shoplifting Detection\dataset\{VIDEO_TYPE}"
OUTPUT_DIR = rf"D:\Random Projects\Fruit Images for Object Detection\Shoplifting Detection\dataset\frames\{VIDEO_TYPE}"
FFMPEG_PATH = r"E:\Media\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"

# Select the first 20 videos from the directory
video_files = [
    "shoplifting-1.mp4", "shoplifting-10.mp4", "shoplifting-11.mp4", "shoplifting-12.mp4",
    "shoplifting-13.mp4", "shoplifting-14.mp4", "shoplifting-15.mp4", "shoplifting-16.mp4",
    "shoplifting-17.mp4", "shoplifting-18.mp4", "shoplifting-19.mp4", "shoplifting-2.mp4",
    "shoplifting-20.mp4", "shoplifting-21.mp4", "shoplifting-22.mp4", "shoplifting-23.mp4",
    "shoplifting-24.mp4", "shoplifting-25.mp4", "shoplifting-26.mp4", "shoplifting-27.mp4"
]

def extract_frames():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Ensured output directory exists: {OUTPUT_DIR}")

    print(f"Starting frame extraction for {VIDEO_TYPE} videos...")

    for video_file in video_files:
        input_video = os.path.join(INPUT_DIR, video_file)
        
        # Create a subdirectory for each video's frames
        video_name = os.path.splitext(video_file)[0]
        video_output_dir = os.path.join(OUTPUT_DIR, video_name)
        os.makedirs(video_output_dir, exist_ok=True)
        
        output_frames_pattern = os.path.join(video_output_dir, f"{video_name}_frame_%04d.jpg")

        # FFmpeg command to extract 1 frame per second
        command = [
            FFMPEG_PATH,
            "-i", input_video,
            "-vf", "fps=1",
            "-q:v", "2", # Quality scale for JPG
            output_frames_pattern
        ]

        print(f"\nProcessing {video_file}...")
        print(f"Running command: {' '.join(command)}")

        try:
            # Run the ffmpeg command
            process = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            print(f"Finished processing {video_file}.")
            if process.stdout:
                print("STDOUT:\n", process.stdout)
            if process.stderr:
                print("STDERR:\n", process.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {video_file}:")
            if e.stdout:
                print("STDOUT:\n", e.stdout)
            if e.stderr:
                print("STDERR:\n", e.stderr)
            print(f"Command failed with exit code {e.returncode}")
        except FileNotFoundError:
            print(f"Error: FFmpeg not found. Please ensure FFmpeg is installed and in your system's PATH, or specify the full path to ffmpeg.exe in the script.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    print(f"\nAll selected {VIDEO_TYPE} video frames extraction process completed.")

if __name__ == "__main__":
    extract_frames()