from utils import read_video, save_video
from trackers import Tracker
import cv2
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import numpy as np
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistanceEstimator


def main():
    #Read
    frames = read_video('input_videos/08fd33_4.mp4')

    tracker = Tracker('best (1).pt')
    tracks = tracker.get_object_tracks(frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')
    
    #Get Object Positions
    tracker.add_position_to_track(tracks)
    #Camera Movement Estimation
    camera_movement_estimator = CameraMovementEstimator(frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(frames, read_from_stop=True, stub_path='stubs/camera_movement_stubs.pkl')
    camera_movement_estimator.adjust_position_to_tracks(tracks, camera_movement_per_frame)

    #View Transformation
    view_transformer = ViewTransformer(frames[0])
    view_transformer.add_transformed_position_to_tracks(tracks)


    #Ball Interpolation
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    #Speed and Distance Estimation
    speed_and_distance_estimator = SpeedAndDistanceEstimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    #Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(frames[0],tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id,track in player_track.items():
            team = team_assigner.get_player_team(frames[frame_num],track['bbox'],player_id)
            
            #Storing team and its color in tracks
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    #Assign Ball to Player
    player_ball_assigner = PlayerBallAssigner()
    team_ball_control = [-1] 
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_ball_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control= np.array(team_ball_control)

    #Draw output
    output_video_frames = tracker.draw_annotations(frames, tracks,team_ball_control)

    #Draw Camera Movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    #Draw Speed and Distance
    output_video_frames = speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)
    
    #Save
    save_video(output_video_frames, 'output_videos/output_video3.avi')



if __name__ == '__main__':
    main()  
