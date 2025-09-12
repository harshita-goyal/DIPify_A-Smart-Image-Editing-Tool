import cv2
import joblib
import os
import numpy as np

# Load the trained model
model_path = "object_classifier.pkl"
if not os.path.exists(model_path):
    print(f"❌ Error: Model file '{model_path}' not found.")
    exit()

model = joblib.load(model_path)

# Path to the test image
image_path = "test.jpeg"

# Check if test image exists
if not os.path.exists(image_path):
    print(f"❌ Error: test.jpeg not found or unreadable at path: {image_path}")
    exit()

# Load and process the image
image = cv2.imread(image_path)
if image is None:
    print("❌ Error: Could not read test.jpeg. Make sure it's a valid image file.")
    exit()

# Convert to grayscale for feature extraction
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
resized = cv2.resize(gray, (64, 64))  # resize for feature extraction
features = resized.flatten()

# Predict using the model
try:
    prediction = model.predict([features])[0]
    label_map = {0: "Bottle", 1: "Pen"}
    predicted_label = label_map.get(prediction, "Unknown")

    # Print prediction
    print(f"✅ Predicted Label: {predicted_label} (Class {prediction})")

    # Draw label on the image
    display_image = image.copy()
    cv2.putText(display_image, f"Detected: {predicted_label}", (1, 200),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (0, 0, 0), 3)

    # Show image
    cv2.imshow("Prediction", display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except ValueError as e:
    print(f"❌ Error during prediction: {e}")
