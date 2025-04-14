from ultralytics import YOLO
import supervision as sv
import pickle
import os

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()  


    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0 ,len(frames), batch_size):
            detections_batch = self.model.predict(frames[i: i+batch_size], conf=0.1)
            detections = detections + detections_batch
            break
        return detections       

    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None) -> dict:

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as file:
                tracks = pickle.load(file)
            return tracks        
            
        detections = self.detect_frames(frames)

        tracks = {
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frame_num, detection in enumerate(detections):
            class_names = detection.names
            class_names_inv = {v:k for k,v in class_names.items()}

            # Convert to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            #Converting goalkeeper to a player
            for obj_index, class_id in enumerate(detection_supervision.class_id):
                if class_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[obj_index] = class_names_inv["player"]

            # Tracking Objects
            detections_with_tracks = self.tracker.update_with_detections(detection_supervision)
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detections_with_tracks:
                bound_box = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if class_id == class_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bound_box":bound_box}

                if class_id == class_names_inv["referee"]:
                    tracks["referee"][frame_num][track_id] = {"bound_box":bound_box}

            for frame_detection in detection_supervision:
                bound_box = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == class_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bound_box":bound_box}

        if stub_path is not None:
            with open(stub_path, "wb") as file:
                pickle.dump(tracks, file)

        return tracks
