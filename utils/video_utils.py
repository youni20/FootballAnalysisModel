import cv2

def read_video(video_path) -> list:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while (True):
        ret, frame = cap.read()
        if not ret:
            break # If false video has ended hence it breaks.
        frames.append(frame)
    return frames 


def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    # The x and y position (width and height of the frame.)
    for frame in output_video_frames:
        out.write(frame)
    out.release()
    