from ultralytics import YOLO
import supervision as sv # type: ignore
import pickle
import os
import cv2
from bbox_utils import get_center_of_bbox, get_bbox_width
import numpy as np
import pandas as pd

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()


    def interpolate_ball_position(self, ball_position):
        ball_position = [x.get(1,{}).get('bbox',[]) for x in ball_position]
        df_ball_position = pd.DataFrame(ball_position, columns=['x1','y1','x2','y2'])

        # Interpolate the ball position
        df_ball_position = df_ball_position.interpolate()
        df_ball_position = df_ball_position.bfill()

        ball_position = [{1:{'bbox':x}}for x in df_ball_position.to_numpy().tolist()]

        return ball_position




    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for x in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[x:x+batch_size], conf=0.1)
            detections += detections_batch
        return detections


    def draw_round(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-40,
            endAngle=250,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        rect_width = 40
        rect_height = 20
        x1_rect = x_center - rect_width//2
        x2_rect = x_center + rect_width//2
        y1_rect = (y2-rect_height//2) + 15
        y2_rect = (y2+rect_height//2) + 15

        # if track_id is not None:
        #     cv2.rectangle(frame, 
        #         (int(x1_rect), int(y1_rect)), 
        #         (int(x2_rect), int(y2_rect)), 
        #         color, 
        #         cv2.FILLED,
        #     )

        #     x1_text = x1_rect+15
        #     if track_id > 99:
        #         x1_text -= 10

        #     cv2.putText(
        #         frame,
        #         f"{track_id}",
        #         (int(x1_text), int(y1_rect+15)),
        #         cv2.FONT_HERSHEY_TRIPLEX,
        #         0.6,
        #         (0,0,0),
        #         thickness=0,
        #     )

        return frame
    
    def draw_traingle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame





    def get_object_track(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)
            return tracks
            


        detections = self.detect_frames(frames)

        tracks={
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            class_names = detection.names
            class_names_inverse = {v:k for k,v in class_names.items()}

            # Converting to Supervision format
            detections_supervision = sv.Detections.from_ultralytics(detection)

            # GK to Player
            for object_ind, class_id in enumerate(detections_supervision.class_id):
                if class_names[class_id] == "goalkeeper":
                    detections_supervision.class_id[object_ind] = class_names_inverse["player"]
            
            # Tracking
            detection_with_tracks = self.tracker.update_with_detections(detections_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                track_id = frame_detection[4]

                if class_id == class_names_inverse["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if class_id == class_names_inverse["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detections_supervision:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]

                if class_id == class_names_inverse["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)


        return tracks
    


    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350,850),(1900,970), (255,255,255), -1)
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]

        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]

        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        cv2.putText(
            frame,
            f"Team 1 Possession: {team_1*100:.2f}%",
            (1400,900),
            cv2.FONT_HERSHEY_TRIPLEX,
            1,
            (0,0,0),
            thickness=3,
        )

        cv2.putText(
            frame,
            f"Team 2 Possession: {team_2*100:.2f}%",
            (1400,950),
            cv2.FONT_HERSHEY_TRIPLEX,
            1,
            (0,0,0),
            thickness=3,
        )

        return frame




    def draw_annotations(self, video_frames, tracks,team_ball_control):
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Player Draw
            for track_id, player in player_dict.items():
                color_team = player.get("team_color", None)
                frame = self.draw_round(frame, player["bbox"], color_team, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_traingle(frame, player["bbox"],(0,0,255))

            # Referee Draw
            for _, referee in referee_dict.items():
                frame = self.draw_round(frame, referee["bbox"], (0,255,0))

            # Ball Draw 
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"],(0,255,0))

            # Team Possession Draw
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)





            output_video_frames.append(frame)

        return output_video_frames
    
