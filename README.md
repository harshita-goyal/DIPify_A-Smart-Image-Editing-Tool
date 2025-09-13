DIPFy: Digital Image Processing Toolkit ->

📌 Overview
DIPFy is a Python-based project developed during my DRDO internship. It focuses on Digital Image Processing (DIP) using OpenCV and integrates a Tkinter-based GUI to provide a Photoshop-like experience.

The project is built in three phases:
1. Core DIP Operations (OpenCV)
2. GUI Integration (Tkinter + OpenCV)
3. Machine Learning-based Object Recognition (scikit-learn + OpenCV)

This project demonstrates how traditional image processing can be combined with GUI design and machine learning to create a complete intelligent vision system.

✨ Features

Task 1: Core DIP Operations (OpenCV)
- Image loading & saving
- Image resizing
- Color space conversions (RGB, HSV, LAB, Grayscale)
- Thresholding (Simple & Adaptive)
- Histogram analysis & Equalization
- Adaptive Histogram Equalization (CLAHE)
- Edge detection (Canny, Sobel)
- Filtering (Gaussian, Median, Bilateral, Sharpening)
- Background removal & masking
- Morphological operations (Erosion, Dilation, Opening, Closing) and many more...

Task 2: GUI Photoshop-like Application (Tkinter + OpenCV)
- Interactive GUI for all above operations
- Menu-driven buttons for easy access
- Open & Save images directly from GUI
- Real-time results preview
- One-click background removal, filters, and enhancements
- User-friendly, minimalistic design

Task 3: ML-based Object Recognition (scikit-learn + OpenCV) (In Progress)
- Dataset creation for objects (e.g., bottles, pens, etc.)
- Feature extraction (Flattening images into arrays)
- Training ML models (KNN, SVM, etc.) using scikit-learn
- Object classification & recognition
- Integration with OpenCV for real-time detection

🛠️ Tech Stack
- Python 3.x
- OpenCV (cv2)
- Tkinter (for GUI)
- NumPy
- matplotlib
- PIL
- scikit-learn (for ML-based recognition)
- OS

📂 Project Structure
│── images/                  # Sample input images
│── videos/                  # Sample videos
│── main_project.py          # Main GUI application (Tkinter + OpenCV)
│── ml_training.py           # ML model training (scikit-learn)
│── README.md                # Project documentation

🚀 Installation & Setup
- Clone the repository:
- git clone (https://github.com/harshita-goyal/DIPify_A-Smart-Image-Editing-Tool)
Install dependencies:
- pip3 install opencv-python numpy scikit-learn
- Run the project:
- python3 main_project.py

📌 Conclusion

This project highlights the integration of image processing techniques with user-friendly interfaces and extends into machine learning for object recognition. It serves as a mini-Photoshop for basic DIP tasks and is evolving into a smart recognition system. The progression from OpenCV-based DIP → GUI applications → ML-based detection reflects both practical application and research potential.
