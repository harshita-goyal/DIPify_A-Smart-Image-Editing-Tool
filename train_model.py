import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Set dataset path and categories
dataset_path = './datasets'
categories = ['bottle', 'pen']
image_size = (64, 64)

# Load and preprocess images
data = []
labels = []

for label, category in enumerate(categories):
    folder_path = os.path.join(dataset_path, category)
    for file_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, image_size)
            data.append(img.flatten())
            labels.append(label)
        else:
            print(f"Warning: Couldn't read image {img_path}")

data = np.array(data)
labels = np.array(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Train SVM model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model trained with accuracy: {accuracy * 100:.2f}%")

# Save model
joblib.dump(model, 'object_classifier.pkl')
print("Model saved as object_classifier.pkl")
