import numpy as np
import cv2

class ViewTransformer:
    # def __init__(self):
    #     court_width = 68
    #     court_length = 23.32 #The 105m divided by number of green patches
        
    #     self.pixel_vertices = np.array([
    #         [110,1035],
    #         [265,275],
    #         [910,260],
    #         [1640,915]
    #     ])

    #     self.target_vertices = np.array([
    #         [0,court_width],
    #         [0,0],
    #         [court_length,0],
    #         [court_length,court_width]
    #     ])

    #     self.pixel_vertices = self.pixel_vertices.astype(np.float32)
    #     self.target_vertices = self.target_vertices.astype(np.float32)

    #     self.perspective_transform = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)
    def __init__(self, first_frame):
        # Standard football pitch dimensions (meters)
        self.court_width = 68
        self.court_length = 105
        
        # Get vertices from first frame
        self.pixel_vertices = self.detect_field_corners(first_frame)
        
        # Define target vertices (birds-eye view)
        self.target_vertices = np.array([
            [0, self.court_width],
            [0, 0],
            [self.court_length, 0],
            [self.court_length, self.court_width]
        ], dtype=np.float32)
        
        self.perspective_transform = cv2.getPerspectiveTransform(
            self.pixel_vertices.astype(np.float32), 
            self.target_vertices
        )
    
    def detect_field_corners(self, frame):
        # Convert to HSV for better grass detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Green color range for football field
        lower_green = np.array([35, 30, 30])
        upper_green = np.array([85, 255, 255])
        
        # Create mask for green field
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Find contours of the field
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get largest contour (field)
        field_contour = max(contours, key=cv2.contourArea)
        
        # Approximate the contour to get corners
        epsilon = 0.02 * cv2.arcLength(field_contour, True)
        approx = cv2.approxPolyDP(field_contour, epsilon, True)
        
        # Get the 4 corners (top-left, top-right, bottom-right, bottom-left)
        corners = np.array(approx).reshape(-1, 2)
        
        # Sort corners
        center = np.mean(corners, axis=0)
        angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        corners = corners[sorted_indices]
        
        return corners
    
    def transform_point(self,point):
        p = (int(point[0]), int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
        if not is_inside:
            return None
        reshaped_point = point.reshape(-1,1,2).astype(np.float32)
        transform_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transform)

        return transform_point.reshape(-1,2)


    def add_transformed_position_to_tracks(self, tracks):
        for object_name, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                if object_name == 'ball':
                    if track and 1 in track and 'position' in track[1]:
                        position = np.array(track[1]['position'])
                        position_transformed = self.transform_point(position)
                        if position_transformed is not None:
                            tracks[object_name][frame_num][1]['position_transformed'] = position_transformed
                else:
                    for track_id, track_info in track.items():
                        if 'position' in track_info:
                            position = np.array(track_info['position'])
                            position_transformed = self.transform_point(position)
                            if position_transformed is not None:
                                tracks[object_name][frame_num][track_id]['position_transformed'] = position_transformed
        
        return tracks