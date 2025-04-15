import pickle
import cv2
import numpy as np
import os
import sys 
sys.path.append('../')
from utils import measure_distance,measure_xy_distance

class CameraMovementEstimator:
    def __init__(self, first_frame):
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

    def get_camera_movement(self, video_frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                camera_movement = pickle.load(f)
            return camera_movement

        old_gray = cv2.cvtColor(video_frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(
            old_gray, maxCorners=100, qualityLevel=0.1, minDistance=7, blockSize=7
        )

        camera_movement = [(0.0, 0.0)]  # First frame has no movement

        for frame in video_frames[1:]:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if old_features is None or len(old_features) == 0:
                print("No features to track. Reinitializing features.")
                old_features = cv2.goodFeaturesToTrack(
                    old_gray, maxCorners=100, qualityLevel=0.1, minDistance=7, blockSize=7
                )
                if old_features is None:
                    camera_movement.append((0.0, 0.0))
                    old_gray = frame_gray.copy()
                    continue

            new_features, status, _ = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, old_features, None, **self.lk_params
            )

            if new_features is None or len(new_features) == 0:
                print("No features tracked in current frame. Using previous movement.")
                camera_movement.append(camera_movement[-1])
                old_gray = frame_gray.copy()
                continue

            good_old = old_features[status == 1]
            good_new = new_features[status == 1]

            if len(good_old) == 0 or len(good_new) == 0:
                camera_movement.append(camera_movement[-1])
                old_gray = frame_gray.copy()
                continue

            movement = np.mean(good_new - good_old, axis=0)
            camera_movement.append((float(movement[0]), float(movement[1])))

            old_gray = frame_gray.copy()
            old_features = good_new.reshape(-1, 1, 2)

        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement

    def add_adjust_positions_to_tracks(self, tracks, camera_movement):
        # Compute cumulative movement for each frame
        cumulative_movement = [(0.0, 0.0)]
        for move in camera_movement[1:]:
            prev = cumulative_movement[-1]
            cumulative_movement.append((prev[0] + move[0], prev[1] + move[1]))

        for object_type, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    if 'position' in track_info:
                        position = track_info['position']
                        movement = cumulative_movement[frame_num]
                        position_adjusted = (
                            position[0] - movement[0],
                            position[1] - movement[1]
                        )
                        tracks[object_type][frame_num][track_id]['position_adjusted'] = position_adjusted

    def draw_camera_movement(self, video_frames, camera_movement):
        output_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            if frame_num > 0:
                movement = camera_movement[frame_num]
                cv2.putText(
                    frame,
                    f"Camera Movement: ({movement[0]:.1f}, {movement[1]:.1f})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )
            output_frames.append(frame)
        return output_frames