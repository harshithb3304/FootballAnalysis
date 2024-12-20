import sys
sys.path.append('../')  
from utils import measure_distance, get_foot_position
import cv2

class SpeedAndDistanceEstimator:
    def __init__(self):
        self.frame_window = 5
        self.frame_rate = 24

    def add_speed_and_distance_to_tracks(self, tracks):
        total_distance = {}

        for object_name, object_tracks in tracks.items():
            if object_name == "ball" or object_name == "referees":
                continue

            number_of_frames = len(object_tracks)
            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, number_of_frames-1)

                for track_id, _ in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame]:
                        continue

                    try:
                        # Get positions and handle missing data
                        start_track = object_tracks[frame_num][track_id]
                        end_track = object_tracks[last_frame][track_id]
                        
                        start_position = start_track.get('position_transformed', start_track.get('position'))
                        end_position = end_track.get('position_transformed', end_track.get('position'))

                        if start_position is None or end_position is None:
                            continue
                        
                        # Format positions correctly
                        start_position = start_position.ravel() if hasattr(start_position, 'ravel') else start_position
                        end_position = end_position.ravel() if hasattr(end_position, 'ravel') else end_position
                        
                        # Calculate distance and speed
                        distance_covered = measure_distance(start_position, end_position)
                        time_elapsed = (last_frame-frame_num)/self.frame_rate
                        speed_mps = distance_covered/time_elapsed if time_elapsed > 0 else 0
                        speed_kmph = speed_mps * 3.6

                        # Initialize distance tracking
                        if object_name not in total_distance:
                            total_distance[object_name] = {}
                        if track_id not in total_distance[object_name]:
                            total_distance[object_name][track_id] = 0
                        
                        # Update total distance
                        total_distance[object_name][track_id] += distance_covered

                        # Update all frames in window
                        for frame_num_batch in range(frame_num, last_frame):
                            if track_id not in tracks[object_name][frame_num_batch]:
                                continue
                            tracks[object_name][frame_num_batch][track_id]['speed'] = speed_kmph
                            tracks[object_name][frame_num_batch][track_id]['distance'] = total_distance[object_name][track_id]
                    
                    except (IndexError, TypeError, ValueError) as e:
                        continue

        return tracks, total_distance

    def draw_speed_and_distance(self,video_frames,tracks):
        output_video_frames = []
        for frame_num,frame in enumerate(video_frames):
            for object, object_tracks in tracks.items():
                if object == 'ball' or object == 'referee':
                    continue

                for _,track_info in object_tracks[frame_num].items():
                    if 'speed' in track_info:
                        speed = track_info.get('speed',None)
                        distance = track_info.get('distance',None)
                        if speed is None or distance is None:
                            continue
                        bbox = track_info['bbox']
                        position = get_foot_position(bbox)
                        position = list(position)
                        position[1]+=40
                        position = tuple(map(int,position))
                        cv2.putText(frame, f'{speed:.2f} km/h', position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(frame, f'{distance:.2f} m', (position[0],position[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            output_video_frames.append(frame)

        return output_video_frames