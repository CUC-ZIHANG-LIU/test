import os
import json
from moviepy.editor import VideoFileClip

def process_videos(input_base_path, output_video_path, output_audio_path, output_json_path):
    categories = ["baby", "cough", "dog", "drum", "fireworks", "gun", "hammers","sneeze"]
    data = []

    for category in categories:
        input_path = os.path.join(input_base_path, category, "videos")
        if not os.path.exists(input_path):
            print(f"Skipping {category}, folder not found: {input_path}")
            continue

        video_category_path = output_video_path#os.path.join(output_video_path, category)
        audio_category_path = output_audio_path#os.path.join(output_audio_path, category)

        os.makedirs(video_category_path, exist_ok=True)
        os.makedirs(audio_category_path, exist_ok=True)

        for file_name in os.listdir(input_path):
            if not file_name.endswith(".mp4"):
                continue

            original_video_path = os.path.join(input_path, file_name)
            base_name = os.path.splitext(file_name)[0]

            # Renamed video file path
            new_video_name = f"{category}-{base_name}.mp4"
            renamed_video_path = os.path.join(video_category_path, new_video_name)

            # Renamed audio file path
            new_audio_name = f"{category}-{base_name}.wav"
            renamed_audio_path = os.path.join(audio_category_path, new_audio_name)

            # Copy video file to the new location with the new name
            os.rename(original_video_path, renamed_video_path)

            # Extract audio from video
            try:
                with VideoFileClip(renamed_video_path) as clip:
                    clip.audio.write_audiofile(renamed_audio_path)
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")
                continue

            # Append file paths to the data list
            data.append({
                "video": renamed_video_path,
                "audio": renamed_audio_path,
                "caption": category
            })

    # Save data to a JSON file
    with open(output_json_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

# Define paths
input_base_path = "/home/chenghaonan/qqt/data/VAS"
output_video_path = "/home/chenghaonan/qqt/data/VAS/video"
output_audio_path = "/home/chenghaonan/qqt/data/VAS/wav"
output_json_path = "/home/chenghaonan/qqt/data/VAS/data.json"

# Process the videos
process_videos(input_base_path, output_video_path, output_audio_path, output_json_path)
