from ultralytics import YOLO

model = YOLO("yolov8x")
results = model.predict("input_videos/messiwinner.mp4", save=True)
# Results will contain detections

print("Results for FIRST frame:")
print(results[0])
print("=======================================")

for box in results[0].boxes:
    print(box) 