from ultralytics import YOLO

model = YOLO("models/best.pt")
results = model.predict("input_videos/espvspor.mp4", save=True)
# Results will contain detections

print("Results for FIRST frame:")
print(results[0])
print("=======================================")

for box in results[0].boxes:
    print(box) 