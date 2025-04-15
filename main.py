import os
from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
import traceback

def main():
    try:
        video_path = 'input_videos/espvspor.mp4'
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return

        print("Reading video frames...")
        video_frames = read_video(video_path)
        if not video_frames:
            print("No frames read from video.")
            return
        print(f"Successfully read {len(video_frames)} frames")

        # Initialize Tracker
        print("Initializing tracker...")
        tracker = Tracker('models/best.pt')

        print("Getting object tracks...")
        tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
        if not tracks:
            print("Failed to get object tracks")
            return

        # Get object positions 
        print("Adding positions to tracks...")
        tracker.add_position_to_tracks(tracks)

        # camera movement estimator
        print("Estimating camera movement...")
        camera_movement_estimator = CameraMovementEstimator(video_frames[0])
        camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement_stub.pkl')
        camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)

        # View Transformer
        print("Transforming view...")
        view_transformer = ViewTransformer()
        view_transformer.add_transformed_position_to_tracks(tracks)

        # Interpolate Ball Positions
        print("Interpolating ball positions...")
        tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

        # Speed and distance estimator
        print("Estimating speed and distance...")
        speed_and_distance_estimator = SpeedAndDistance_Estimator()
        speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

        # Assign Player Teams
        print("Assigning player teams...")
        team_assigner = TeamAssigner()
        team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
        
        for frame_num, player_track in enumerate(tracks['players']):
            for player_id, track in player_track.items():
                team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
                tracks['players'][frame_num][player_id]['team'] = team 
                tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

        # Assign Ball Acquisition
        print("Assigning ball possession...")
        player_assigner = PlayerBallAssigner()
        team_ball_control = []
        for frame_num, player_track in enumerate(tracks['players']):
            if frame_num < len(tracks['ball']) and 1 in tracks['ball'][frame_num]:
                ball_bbox = tracks['ball'][frame_num][1]['bbox']
                assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

                if assigned_player != -1:
                    tracks['players'][frame_num][assigned_player]['has_ball'] = True
                    team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
                else:
                    team_ball_control.append(team_ball_control[-1] if team_ball_control else None)
        team_ball_control = np.array(team_ball_control)

        # Draw output 
        print("Drawing annotations...")
        output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

        print("Drawing camera movement...")
        output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

        print("Drawing speed and distance...")
        speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

        # Save video
        print("Saving output video...")
        output_path = 'output_videos/output_video.avi'
        save_video(output_video_frames, output_path)
        print(f"Successfully saved video to {output_path}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Traceback:")
        traceback.print_exc()

if __name__ == '__main__':
    main()