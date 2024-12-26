from ultralytics import YOLO

# Force using CPU
device = "cpu"
print(f"Running on: {device}")

# Load the trained model
model = YOLO("../model/best.pt")
model.to(device)  # Send the model to CPU

# Perform prediction (Inference) for a single image
image_path = "./test_pic/6.png"  # Specify the image path
results = model.predict(
    source=image_path,  # Use a single image
    imgsz=128,          # Resize the image (must be multiples of the stride, e.g., 128, 256, 512)
    device=device       # Specify the device as CPU
)

# Count the number of intersection points
detections = results[0].boxes  # Get the results for the first image
intersection_count = sum(1 for box in detections if box.cls == 0)  # Check only class 0 (intersection points)

# Display the result
print(f"Number of intersection points in {image_path}: {intersection_count}")
