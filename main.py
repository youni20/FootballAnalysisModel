from utils import read_video, save_video


def main():
    # Read Vids
    video_frames = read_video('input_videos/espvspor.mp4')

    # Save the video
    save_video(video_frames, 'output_video/output_video.avi')

if __name__ == "__main__":
    main()