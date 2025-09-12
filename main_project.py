from tkinter import*
from tkinter import ttk
from tkinter import filedialog
import cv2
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image, ImageTk

class DIPify:
    def __init__(self, win):
        self.win = win
        self.win.title("DIPify - A Smart Image Editing Tool")
        self.win.geometry("1000x600")
        self.win.config(bg="#3A5F83")
        self.win.resizable(False,False)

        self.image_path = None
        self.cv_img = None  # result image 
        self.tk_img = None  # uploaded image

        self.create_widgets()

########################! definition of all the panel button : ###############################
    #todo:resize: 
    def resize_image(self):  
        if self.cv_img is None:
            self.status.config(text="Please upload an image first.")
            return
        x=200
        y=200
        resized = cv2.resize(self.cv_img, (x, y))
        img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        tk_result = ImageTk.PhotoImage(img_pil)

        if not hasattr(self, 'result_img_label'):
            self.result_img_label = Label(self.win, bg="white")
            self.result_img_label.place(x=600, y=230) 

        self.result_img_label.config(image=tk_result)
        self.result_img_label.image = tk_result
        self.status.config(text=f"Image resized and shown:({x}x{y})")

    #todo: draw shapes:
    def draw_shapes(self):
        if self.cv_img is None:
            self.status.config(text="Please upload an image first.")
            return

        import numpy as np

        img = self.cv_img.copy()
        img = cv2.resize(img, (200, 200))

        # Add text
        cv2.putText(img, "DIPify", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 100, 200), 2)
        # Line
        cv2.line(img, (10, 30), (140, 30), (0, 255, 255), 2)
        # Rectangle
        cv2.rectangle(img, (20, 40), (130, 70), (0, 255, 0), 2)
        # Circle
        cv2.circle(img, (75, 100), 20, (255, 255, 0), 2)
        # Ellipse
        cv2.ellipse(img, (75, 120), (30, 10), 30, 0, 360, (255, 0, 255), 2)
        # Polygon
        pts = np.array([[20, 130], [40, 110], [75, 115], [110, 130], [75, 140]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
        # Convert to Tkinter format
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        tk_result = ImageTk.PhotoImage(img_pil)
        # Show in result frame
        if not hasattr(self, 'result_img_label'):
            self.result_img_label = Label(self.win, bg="white")
            self.result_img_label.place(x=600, y=230)
        self.result_img_label.config(image=tk_result)
        self.result_img_label.image = tk_result
        self.status.config(text="Shapes drawn successfully!")

    #todo: airthmetic opr buttons:
    def arithmetic_ops(self):
        if self.cv_img is None:
            self.status.config(text="Please upload an image first.")
            return

        try:
            img1 = cv2.resize(self.cv_img, (200, 200))
            img2 = img1.copy()
            operation = self.arith_op_var.get()

            #operation
            if operation == "Add":
                result = cv2.addWeighted(img1, 0.6, img2, 0.4, 30)
            elif operation == "Subtract":
                result = cv2.subtract(img1, img2)
            elif operation == "Multiply":
                result = cv2.multiply(img1, img2)
            elif operation == "Divide":
                result = cv2.divide(img1, img2)
            else:
                self.status.config(text="Unknown operation selected.")
                return

            img_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            tk_result = ImageTk.PhotoImage(img_pil)

            if not hasattr(self, 'result_img_label'):
                self.result_img_label = Label(self.win, bg="white")
                self.result_img_label.place(x=600, y=230)

            self.result_img_label.config(image=tk_result)
            self.result_img_label.image = tk_result
            self.status.config(text=f"{operation} operation applied successfully!")

        except Exception as e:
            self.status.config(text=f"Error: {e}")
    #todo: bitwise opr:
    def bitwise_ops(self):
        if self.cv_img is None:
            self.status.config(text="Please upload an image first.")
            return

        try:
            img1 = cv2.resize(self.cv_img, (200, 200))
            img2 = img1.copy()  # using same image as both operands

            operation = self.bitwise_op_var.get()

            if operation == "NOT":
                result = cv2.bitwise_not(img1)
            elif operation == "AND":
                result = cv2.bitwise_and(img1, img2)
            elif operation == "OR":
                result = cv2.bitwise_or(img1, img2)
            elif operation == "XOR":
                result = cv2.bitwise_xor(img1, img2)
            else:
                self.status.config(text="Unknown bitwise operation selected.")
                return

            img_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            tk_result = ImageTk.PhotoImage(img_pil)

            if not hasattr(self, 'result_img_label'):
                self.result_img_label = Label(self.win, bg="white")
                self.result_img_label.place(x=600, y=230)

            self.result_img_label.config(image=tk_result)
            self.result_img_label.image = tk_result
            self.status.config(text=f"{operation} operation applied successfully!")

        except Exception as e:
            self.status.config(text=f"Error: {e}")

    #todo: edge detection:
    def edge_detection(self):
        if self.cv_img is None:
            self.status.config(text="Please upload an image first.")
            return

        try:
            img = cv2.resize(self.cv_img, (200, 200))

            # Apply Canny edge detection
            edge_img = cv2.Canny(
                image=img,
                threshold1=50,
                threshold2=50,
                apertureSize=3,
                L2gradient=True
            )

            # Convert to RGB mode
            img_rgb = cv2.cvtColor(edge_img, cv2.COLOR_GRAY2RGB)
            img_pil = Image.fromarray(img_rgb)
            tk_result = ImageTk.PhotoImage(img_pil)

            if not hasattr(self, 'result_img_label'):
                self.result_img_label = Label(self.win, bg="white")
                self.result_img_label.place(x=600, y=230)

            self.result_img_label.config(image=tk_result)
            self.result_img_label.image = tk_result
            self.status.config(text="Edge detection applied successfully!")

        except Exception as e:
            self.status.config(text=f"Error: {e}")

    #todo: rotating:
    def rotate_image(self):
        if self.cv_img is None:
            self.status.config(text="Please upload an image first.")
            return

        try:
            img = cv2.resize(self.cv_img, (200, 200))
            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)

            angle = self.rotation_angle.get() 

            rotation_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1.0)
            rotated = cv2.warpAffine(img, M=rotation_matrix, dsize=(w, h))

            img_rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            tk_result = ImageTk.PhotoImage(img_pil)

            if not hasattr(self, 'result_img_label'):
                self.result_img_label = Label(self.win, bg="white")
                self.result_img_label.place(x=600, y=230)

            self.result_img_label.config(image=tk_result)
            self.result_img_label.image = tk_result
            self.status.config(text=f"Image rotated by {angle}° successfully!")

        except Exception as e:
            self.status.config(text=f"Error: {e}")

    #todo: blur:
    def apply_blur(self):
        if self.cv_img is None:
            self.status.config(text="Please upload an image first.")
            return

        method = self.blur_method_var.get()

        try:
            img = cv2.resize(self.cv_img, (200, 200))

            if method == "Gaussian":
                result = cv2.GaussianBlur(img, (5, 5), 0)
            elif method == "Median":
                result = cv2.medianBlur(img, 5)
            elif method == "Bilateral":
                result = cv2.bilateralFilter(img, 9, 75, 75)
            else:
                self.status.config(text="Invalid blur method selected.")
                return

            img_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            tk_result = ImageTk.PhotoImage(img_pil)

            if not hasattr(self, 'result_img_label'):
                self.result_img_label = Label(self.win, bg="white")
                self.result_img_label.place(x=600, y=230)

            self.result_img_label.config(image=tk_result)
            self.result_img_label.image = tk_result
            self.status.config(text=f"{method} blur applied successfully!")

        except Exception as e:
            self.status.config(text=f"Error: {e}")

    #todo: save image:
    def save_image(self):
        if not hasattr(self, 'result_img_label') or self.result_img_label.image is None:
            self.status.config(text="No processed image to save.")
            return

        try:
            from tkinter import filedialog
            file_path = filedialog.asksaveasfilename(defaultextension=".jpg",filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("All files", "*.*")])
            if not file_path:
                self.status.config(text="Save cancelled.")
                return

            # Convert the displayed image back to OpenCV format
            result_img = self.result_img_label.image  # PhotoImage (RGB)
            img_pil = result_img._PhotoImage__photo.convert("RGB")  # Convert back to PIL
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)  # PIL to OpenCV (BGR)

            cv2.imwrite(file_path, img_cv)
            self.status.config(text=f"Image saved at: {file_path}")

        except Exception as e:
            self.status.config(text=f"Error saving image: {e}")

    #todo: make border:
    def add_border(self):
        if self.cv_img is None:
            self.status.config(text="Please upload an image first.")
            return

        try:
            img = cv2.resize(self.cv_img, (200, 200))
            bordered = cv2.copyMakeBorder(
                img,
                top=10, bottom=10, left=10, right=10,
                borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 255) 
            )

            img_rgb = cv2.cvtColor(bordered, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            tk_result = ImageTk.PhotoImage(img_pil)

            if not hasattr(self, 'result_img_label'):
                self.result_img_label = Label(self.win, bg="white")
                self.result_img_label.place(x=600, y=230)

            self.result_img_label.config(image=tk_result)
            self.result_img_label.image = tk_result
            self.status.config(text="Border added successfully!")

        except Exception as e:
            self.status.config(text=f"Error adding border: {e}")

    #todo: play a video:
    def play_video_in_label(self):
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
        if not path:
            self.status.config(text="No video selected.")
            return
        self.image_path = path  
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            self.status.config(text="Failed to open video.")
            return

        self.status.config(text="Playing video...")
        self.playing_video = True
        self.paused = False

        def stream():
            while self.playing_video:
                if self.paused:
                    self.win.update()
                    self.win.after(100)
                    continue

                ret, frame = self.cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (250, 250))
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(frame_rgb)
                tk_img = ImageTk.PhotoImage(img_pil)

                self.input_img_label.config(image=tk_img)
                self.input_img_label.image = tk_img

                self.win.update()
                self.win.after(30)

            self.cap.release()
            self.status.config(text="Video playback stopped or finished.")
            self.playing_video = False
        self.win.after(0, stream)

    def stop_video(self):
        self.playing_video = False
        self.status.config(text="Video stopped by user.")

    def toggle_pause(self, event=None):
        if self.playing_video:
            self.paused = not self.paused
            state = "Paused" if self.paused else "Resumed"
            self.status.config(text=f"Video {state}")

    #todo: camera:
    def open_camera(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.status.config(text="Error: Cannot access camera.")
            return

        self.status.config(text="Camera is running. Press 'p' to stop.")

        while True:
            ret, frame = cap.read()
            if not ret:
                self.status.config(text="Failed to grab frame.")
                break

            frame = cv2.flip(frame, 1)
            resized = cv2.resize(frame, (500, 500))
            cv2.imshow("Camera - DIPify", resized)

            if cv2.waitKey(45) & 0xFF == ord('p'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.status.config(text="Camera stopped by- 'p'")

    #todo: morphological:
    def morphological_ops(self):
        if self.cv_img is None:
            self.status.config(text="Please upload an image first.")
            return

        try:
            img = cv2.resize(self.cv_img, (200, 200))
            operation = self.morph_op_var.get()
            kernel = np.ones((10, 10), np.uint8)

            if operation == "Erosion":
                result = cv2.erode(img, kernel, iterations=1)
            elif operation == "Dilation":
                result = cv2.dilate(img, kernel, iterations=1)
            elif operation == "Opening":
                result = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
            elif operation == "Closing":
                result = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
            elif operation == "Gradient":
                result = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel, iterations=1)
            elif operation == "Top Hat":
                result = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel, iterations=1)
            elif operation == "Black Hat":
                result = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel, iterations=1)
            else:
                self.status.config(text="Unknown morphological operation.")
                return

            img_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            tk_result = ImageTk.PhotoImage(img_pil)

            if not hasattr(self, 'result_img_label'):
                self.result_img_label = Label(self.win, bg="white")
                self.result_img_label.place(x=600, y=230)

            self.result_img_label.config(image=tk_result)
            self.result_img_label.image = tk_result
            self.status.config(text=f"{operation} applied successfully!")

        except Exception as e:
            self.status.config(text=f"Error: {e}")
        
    #todo: translate:    
    def translate_image(self):
        if self.cv_img is None:
            self.status.config(text="Please upload an image first.")
            return

        import numpy as np

        try:
            img = cv2.resize(self.cv_img, (200, 200)) 
            shift_x, shift_y = 30, 30 
            matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            translated_img = cv2.warpAffine(img, matrix, (200, 200))

            img_rgb = cv2.cvtColor(translated_img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            tk_result = ImageTk.PhotoImage(img_pil)

            if not hasattr(self, 'result_img_label'):
                self.result_img_label = Label(self.win, bg="white")
                self.result_img_label.place(x=600, y=230)

            self.result_img_label.config(image=tk_result)
            self.result_img_label.image = tk_result
            self.status.config(text=f"Image translated by ({shift_x}, {shift_y}) pixels!")

        except Exception as e:
            self.status.config(text=f"Error: {e}")

    #todo: cvt color:
    def cvt_color_convert(self):
        if self.cv_img is None:
            self.status.config(text="Please upload an image first.")
            return

        try:
            option = self.color_conv_var.get()
            img = cv2.resize(self.cv_img, (200, 200))

            if option == "GRAY":
                result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)  
            elif option == "HSV":
                result = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            elif option == "LUV":
                result = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
            elif option == "LAB":
                result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            elif option == "HLS":
                result = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
            elif option == "YCrCb":
                result = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            else:
                self.status.config(text="Invalid color conversion selected.")
                return

            # Convert to displayable format
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB) if option != "GRAY" else result
            img_pil = Image.fromarray(result_rgb)
            tk_result = ImageTk.PhotoImage(img_pil)

            if not hasattr(self, 'result_img_label'):
                self.result_img_label = Label(self.win, bg="white")
                self.result_img_label.place(x=600, y=230)

            self.result_img_label.config(image=tk_result)
            self.result_img_label.image = tk_result
            self.status.config(text=f"Color converted to {option} successfully!")

        except Exception as e:
            self.status.config(text=f"Error: {e}")

    #todo: crop:
    def crop_image(self):
        if self.cv_img is None:
            self.status.config(text="Please upload an image first.")
            return

        try:
            img = cv2.resize(self.cv_img, (200, 200)) 
            crop = img[5:205, 75:390]

            # Convert to displayable format
            img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            tk_result = ImageTk.PhotoImage(img_pil)

            if not hasattr(self, 'result_img_label'):
                self.result_img_label = Label(self.win, bg="white")
                self.result_img_label.place(x=600, y=230)

            self.result_img_label.config(image=tk_result)
            self.result_img_label.image = tk_result
            self.status.config(text=f"Cropped image shown successfully! Shape: {crop.shape}")

        except Exception as e:
            self.status.config(text=f"Error: {e}")

    #todo: blank image: 
    def create_blank_image(self):
        try:
            mode = self.blank_mode_var.get()
            size = (200, 200, 3)

            if mode == "White":
                img = np.ones(size, dtype=np.uint8) * 255
            elif mode == "Black":
                img = np.zeros(size, dtype=np.uint8)
            else:
                self.status.config(text="Invalid blank image mode selected.")
                return

            self.cv_img = img 

            img_pil = Image.fromarray(img)
            tk_result = ImageTk.PhotoImage(img_pil)

            self.input_img_label.config(image=tk_result)
            self.input_img_label.image = tk_result
            self.status.config(text=f"{mode} image created successfully!")

        except Exception as e:
            self.status.config(text=f"Error: {e}")

    #todo:  roi:
    def roi_clone(self):
        if self.cv_img is None:
            self.status.config(text="Please upload an image first.")
            return

        try:
            img = cv2.resize(self.cv_img, (200, 200))

            # ROI 
            crop = img[40:120, 70:130].copy()

            # right
            img[40:120, 130:190] = crop

            # left
            img[40:120, 10:70] = crop

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            tk_result = ImageTk.PhotoImage(img_pil)

            if not hasattr(self, 'result_img_label'):
                self.result_img_label = Label(self.win, bg="white")
                self.result_img_label.place(x=600, y=230)

            self.result_img_label.config(image=tk_result)
            self.result_img_label.image = tk_result
            self.status.config(text="ROI Clone applied on 200x200 image!")

        except Exception as e:
            self.status.config(text=f"Error: {e}")

    #todo: save a viedo:
    def save_uploaded_video(self):
            if not hasattr(self, 'cap') or not self.cap.isOpened():
                self.status.config(text="No uploaded video found.")
                return

            from tkinter.filedialog import asksaveasfilename
            save_path = asksaveasfilename(defaultextension=".avi",
                                        filetypes=[("AVI Video", "*.avi")],
                                        title="Save Processed Video As")
            if not save_path:
                self.status.config(text="Save cancelled.")
                return

            # Reset video to the beginning
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))

            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

            self.status.config(text="Saving uploaded video...")

            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                out.write(frame)

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
            out.release()
            self.status.config(text=f"Video saved to: {save_path}")

    #todo: filter the perticular color:
    def filter_color(self):
        if self.cv_img is None:
            self.status.config(text="Please upload an image first.")
            return

        import numpy as np

        img = cv2.resize(self.cv_img, (200, 200))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        choice = self.color_filter_var.get()
        if choice == "Blue":
            lower = np.array([100, 150, 50])
            upper = np.array([140, 255, 255])
        elif choice == "Green":
            lower = np.array([35, 100, 50])
            upper = np.array([85, 255, 255])
        elif choice == "Skin":
            lower = np.array([0, 20, 70])
            upper = np.array([20, 255, 255])
        else:
            self.status.config(text="Unknown filter selected.")
            return

        mask = cv2.inRange(hsv, lower, upper)
        filtered = cv2.bitwise_and(img, img, mask=mask)
        rgb = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb)
        tk_img = ImageTk.PhotoImage(img_pil)

        if not hasattr(self, 'result_img_label'):
            self.result_img_label = Label(self.win, bg="white")
            self.result_img_label.place(x=600, y=230)

        self.result_img_label.config(image=tk_img)
        self.result_img_label.image = tk_img
        self.status.config(text=f"{choice} filter applied!")

    #todo: thresholding:
    def threshold_ops(self):
        if self.cv_img is None:
            self.status.config(text="Please upload an image first.")
            return

        import numpy as np

        # Prepare a grayscale 200×200 image
        img = cv2.resize(self.cv_img, (200, 200))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        method = self.thresh_var.get()

        if method == "Simple":
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        elif method == "Otsu":
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == "Adaptive":
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
        else:
            self.status.config(text="Unknown threshold method.")
            return

        # Convert to RGB for display
        thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        img_pil = Image.fromarray(thresh_rgb)
        tk_img = ImageTk.PhotoImage(img_pil)

        if not hasattr(self, 'result_img_label'):
            self.result_img_label = Label(self.win, bg="white")
            self.result_img_label.place(x=600, y=230)

        self.result_img_label.config(image=tk_img)
        self.result_img_label.image = tk_img
        self.status.config(text=f"{method} threshold applied.")

    #todo: histogram:
    def histogram_ops(self):
        if self.cv_img is None:
            self.status.config(text="Please upload an image first.")
            return

        # Resize and split channels
        img = cv2.resize(self.cv_img, (200, 200))
        b, g, r = cv2.split(img)

        # Calculate histograms
        hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
        hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])

        # Plot using matplotlib
        plt.figure()
        plt.plot(hist_b, color='blue', label='Blue Channel')
        plt.plot(hist_g, color='green', label='Green Channel')
        plt.plot(hist_r, color='red', label='Red Channel')
        plt.legend()
        plt.title('Color Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.show()
    def show_histogram(self):
        self.histogram_ops()

    #todo: adaptive equllization:
    def equalize_histogram(self):
        if self.cv_img is None:
            self.status.config(text="Please upload an image first.")
            return

        try:
            import numpy as np

            # Resize and convert to grayscale
            img = cv2.resize(self.cv_img, (200, 200))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            choice = self.equalize_var.get()

            if choice == "Global":
                result = cv2.equalizeHist(gray)
            elif choice == "CLAHE":
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                result = clahe.apply(gray)
            else:
                self.status.config(text="Unknown equalization method.")
                return

            # Convert back to RGB for display
            result_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
            img_pil = Image.fromarray(result_rgb)
            tk_img = ImageTk.PhotoImage(img_pil)

            if not hasattr(self, 'result_img_label'):
                self.result_img_label = Label(self.win, bg="white")
                self.result_img_label.place(x=600, y=230)

            self.result_img_label.config(image=tk_img)
            self.result_img_label.image = tk_img
            self.status.config(text=f"{choice} equalization applied!")

        except Exception as e:
            self.status.config(text=f"Error: {e}")

    #todo: mouse event:
    def _mouse_draw_callback(self, event, x, y, flags, param):
    # Draw on self.mouse_img whenever the user clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(self.mouse_img, (x, y), 6, (255, 0, 0), 3)
        elif event == cv2.EVENT_RBUTTONDOWN:
            cv2.rectangle(self.mouse_img, (x, y), (x + 25, y + 20), (0, 0, 255), 3)

    def mouse_event_draw(self):
        if self.cv_img is None:
            self.status.config(text="Please upload an image first.")
            return
        self.mouse_img = cv2.resize(self.cv_img, (400, 400))

        window_name = "DIPify Mouse Drawing"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._mouse_draw_callback)

        self.status.config(text="Mouse drawing active: Left-click ▶ circle, Right-click ▶ rect. Press 'q' to exit.")

        while True:
            cv2.imshow(window_name, self.mouse_img)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

        cv2.destroyWindow(window_name)
        edited = cv2.resize(self.mouse_img, (self.cv_img.shape[1], self.cv_img.shape[0]))
        self.cv_img = edited
        img_rgb = cv2.cvtColor(edited, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        tk_img = ImageTk.PhotoImage(img_pil)
        if not hasattr(self, 'result_img_label'):
            self.result_img_label = Label(self.win, bg="white")
            self.result_img_label.place(x=600, y=230)
        self.result_img_label.config(image=tk_img)
        self.result_img_label.image = tk_img
        self.status.config(text="Mouse drawing applied to image.")

    #todo: contour object:
    def contour_detection(self):
        if self.cv_img is None:
            self.status.config(text="Please upload an image first.")
            return

        try:
            # Resize and convert to grayscale
            img = cv2.resize(self.cv_img, (200, 200))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Binarize with simple threshold
            _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)

            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Draw all contours on a copy
            contoured = img.copy()
            cv2.drawContours(contoured, contours, -1, (255, 0, 0), 2)

            # Convert and display
            rgb = cv2.cvtColor(contoured, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb)
            tk_img = ImageTk.PhotoImage(img_pil)

            if not hasattr(self, 'result_img_label'):
                self.result_img_label = Label(self.win, bg="white")
                self.result_img_label.place(x=600, y=230)

            self.result_img_label.config(image=tk_img)
            self.result_img_label.image = tk_img
            self.status.config(text=f"Detected {len(contours)} contours.")
        except Exception as e:
            self.status.config(text=f"Error: {e}")

    #todo: contour moment:
    def contour_moments(self):
        if self.cv_img is None:
            self.status.config(text="Please upload an image first.")
            return

        try:
            img = cv2.resize(self.cv_img, (200, 200))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            out = img.copy()
            area_list = []
            for c in contours:
                cv2.drawContours(out, [c], -1, (255, 50, 67), 2)

                m = cv2.moments(c)
                if m["m00"] != 0:
                    cx = int(m["m10"] / m["m00"])
                    cy = int(m["m01"] / m["m00"])
                    cv2.circle(out, (cx, cy), 3, (0, 0, 255), -1)
                else:
                    cx, cy = 0, 0

                area = cv2.contourArea(c)
                area_list.append(area)

            rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb)
            tk_img = ImageTk.PhotoImage(img_pil)

            if not hasattr(self, 'result_img_label'):
                self.result_img_label = Label(self.win, bg="white")
                self.result_img_label.place(x=600, y=230)

            self.result_img_label.config(image=tk_img)
            self.result_img_label.image = tk_img
            self.status.config(text=f"Moments shown: {len(contours)} contours, areas: {['{:.1f}'.format(a) for a in area_list]}")
        except Exception as e:
            self.status.config(text=f"Error: {e}")
            
    #todo: convex hull:
    def convex_hull(self):
        if self.cv_img is None:
            self.status.config(text="Please upload an image first.")
            return

        try:
            # Resize and preprocess
            img = cv2.resize(self.cv_img, (200, 200))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)

            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            out = img.copy()
            # Draw each contour and its convex hull bounding box
            for con in contours:
                # Optional: approximate contour to reduce points
                epsilon = 0.01 * cv2.arcLength(con, True)
                approx = cv2.approxPolyDP(con, epsilon, True)

                # Compute convex hull
                hull = cv2.convexHull(approx)

                # Draw hull outline
                cv2.drawContours(out, [hull], -1, (255, 0, 0), 2)

                # Draw bounding rect around hull
                x, y, w, h = cv2.boundingRect(hull)
                cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display result
            rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb)
            tk_img = ImageTk.PhotoImage(img_pil)

            if not hasattr(self, 'result_img_label'):
                self.result_img_label = Label(self.win, bg="white")
                self.result_img_label.place(x=600, y=230)

            self.result_img_label.config(image=tk_img)
            self.result_img_label.image = tk_img
            self.status.config(text=f"Convex hull drawn on {len(contours)} contours.")

        except Exception as e:
            self.status.config(text=f"Error: {e}")

    #todo: bg remover:
    def background_removal(self):
        if self.cv_img is None:
            self.status.config(text="Please upload an image first.")
            return

        try:
            img = cv2.resize(self.cv_img, (200, 200))
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            patch = img[:40, :40]
            hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

            hist = cv2.calcHist([hsv_patch], [0, 1], None, [180, 256], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

            back_proj = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            back_proj = cv2.filter2D(back_proj, -1, kernel)

            _, mask = cv2.threshold(back_proj, 50, 255, cv2.THRESH_BINARY_INV)

            fg = cv2.bitwise_and(img, img, mask=mask)


            fg_rgb = cv2.cvtColor(fg, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(fg_rgb)
            tk_img = ImageTk.PhotoImage(img_pil)

            if not hasattr(self, 'result_img_label'):
                self.result_img_label = Label(self.win, bg="white")
                self.result_img_label.place(x=600, y=230)

            self.result_img_label.config(image=tk_img)
            self.result_img_label.image = tk_img
            self.status.config(text="Background removed via histogram backprojection!")

        except Exception as e:
            self.status.config(text=f"Error: {e}")
            
    #todo: hough_line            
    def hough_transform(self):
        if self.cv_img is None:
            self.status.config(text="Please upload an image first.")
            return

        import numpy as np

        try:
            # Prepare image
            img = cv2.resize(self.cv_img, (200, 200))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            choice = self.hough_var.get()
            out = img.copy()

            if choice == "Standard":
                lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
                if lines is not None:
                    for line in lines:
                        rho, theta = line[0]
                        a, b = np.cos(theta), np.sin(theta)
                        x0, y0 = a*rho, b*rho
                        x1 = int(x0 + 1000*(-b))
                        y1 = int(y0 + 1000*(a))
                        x2 = int(x0 - 1000*(-b))
                        y2 = int(y0 - 1000*(a))
                        cv2.line(out, (x1, y1), (x2, y2), (0,255,0), 1)
            else:  # Probabilistic
                linesP = cv2.HoughLinesP(edges, 1, np.pi/180, 80, minLineLength=50, maxLineGap=10)
                if linesP is not None:
                    for l in linesP:
                        x1,y1,x2,y2 = l[0]
                        cv2.line(out, (x1, y1), (x2, y2), (0,255,0), 1)

            # Show result
            rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb)
            tk_img = ImageTk.PhotoImage(img_pil)

            if not hasattr(self, 'result_img_label'):
                self.result_img_label = Label(self.win, bg="white")
                self.result_img_label.place(x=600, y=230)

            self.result_img_label.config(image=tk_img)
            self.result_img_label.image = tk_img
            self.status.config(text=f"Hough ({choice}) applied.")

        except Exception as e:
            self.status.config(text=f"Error: {e}")
            
    #todo: template matching:
    def template_matching(self):
        if self.cv_img is None:
            self.status.config(text="Please upload an image first.")
            return

        try:
            img = cv2.resize(self.cv_img, (200, 200))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Define template region (you can adjust these coords or
            # make them dynamic later)
            templ = img[40:160, 60:140]
            gray_templ = cv2.cvtColor(templ, cv2.COLOR_BGR2GRAY)
            w, h = gray_templ.shape[::-1]

            # Perform matching
            res = cv2.matchTemplate(gray, gray_templ, cv2.TM_CCOEFF_NORMED)
            threshold = 0.8
            loc = np.where(res >= threshold)

            # Draw rectangles at match locations
            out = img.copy()
            for pt in zip(*loc[::-1]):
                cv2.rectangle(out, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

            # Display result
            rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb)
            tk_img = ImageTk.PhotoImage(img_pil)

            if not hasattr(self, 'result_img_label'):
                self.result_img_label = Label(self.win, bg="white")
                self.result_img_label.place(x=600, y=230)

            self.result_img_label.config(image=tk_img)
            self.result_img_label.image = tk_img
            self.status.config(text=f"Template Matching complete: {len(loc[0])} matches found.")

        except Exception as e:
            self.status.config(text=f"Error: {e}")

    #todo: grabcut:                                
    def grabcut_segmentation(self):
        if self.cv_img is None:
            self.status.config(text="Please upload an image first.")
            return

        try:
            img = cv2.resize(self.cv_img, (200, 200))
            mask = np.zeros(img.shape[:2], np.uint8)
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            rect = (20, 20, 160, 160)
            cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            result = img * mask2[:, :, np.newaxis]

            rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb)
            tk_img = ImageTk.PhotoImage(img_pil)

            if not hasattr(self, 'result_img_label'):
                self.result_img_label = Label(self.win, bg="white")
                self.result_img_label.place(x=600, y=230)

            self.result_img_label.config(image=tk_img)
            self.result_img_label.image = tk_img
            self.status.config(text="GrabCut segmentation applied!")
        except Exception as e:
            self.status.config(text=f"Error: {e}")


    #todo: object tracking and detection :
    def track_object(self):
        import threading

        if not hasattr(self, 'image_path') or not self.image_path.lower().endswith((".mp4", ".avi", ".mov")):
            self.status.config(text="Please upload a video first.")
            return

        def run_tracking():
            cap = cv2.VideoCapture(self.image_path)
            ret, frame = cap.read()
            if not ret:
                self.status.config(text="Failed to read video.")
                cap.release()
                return

            # Resize & setup initial window
            frame = cv2.resize(frame, (250, 250))
            # Fixed ROI for initialization (x, y, w, h)
            x, y, w, h = 60, 15, 70, 200  
            track_window = (x, y, w, h)

            # Set up ROI for histogram
            roi = frame[y:y+h, x:x+w]
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
            roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
            cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

            # Setup termination: either 10 iterations or move by at least 1 pt
            term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

            self.status.config(text="Tracking... press 'p' to stop")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (250, 250))
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                back_proj = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

                # Apply meanShift to get the new location
                ret2, track_window = cv2.meanShift(back_proj, track_window, term_crit)
                x, y, w, h = track_window

                # Draw rectangle on tracked object
                result_frame = cv2.rectangle(frame.copy(), (x, y), (x + w, y + h), (0, 0, 255), 2)

                # Convert and display in label
                rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(rgb)
                tk_img = ImageTk.PhotoImage(img_pil)

                if not hasattr(self, 'result_img_label'):
                    self.result_img_label = Label(self.win, bg="white")
                    self.result_img_label.place(x=600, y=230)

                self.result_img_label.config(image=tk_img)
                self.result_img_label.image = tk_img

                # Delay & break on 'p'
                if cv2.waitKey(30) & 0xFF == ord('p'):
                    break

            cap.release()
            self.status.config(text="Tracking finished.")

        threading.Thread(target=run_tracking, daemon=True).start()

    #todo: harris corner:
    def harris_corners(self):
        if self.cv_img is None:
            self.status.config(text="Please upload an image first.")
            return

        try:
            import numpy as np

            # 1. Resize and prepare grayscale float32 image
            img = cv2.resize(self.cv_img, (200, 200))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = np.float32(gray)

            dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

            dst = cv2.dilate(dst, None)

            corner_img = img.copy()
            corner_img[dst > 0.01 * dst.max()] = [0, 0, 255]

            rgb = cv2.cvtColor(corner_img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb)
            tk_img = ImageTk.PhotoImage(img_pil)

            if not hasattr(self, 'result_img_label'):
                self.result_img_label = Label(self.win, bg="white")
                self.result_img_label.place(x=600, y=230)

            self.result_img_label.config(image=tk_img)
            self.result_img_label.image = tk_img
            self.status.config(text="Harris corners detected!")

        except Exception as e:
            self.status.config(text=f"Error: {e}")
                    
    #todo: shi-Tomasi corner:
    def shi_tomasi_corners(self):
        if self.cv_img is None:
            self.status.config(text="Please upload an image first.")
            return

        try:
            import numpy as np

            img = cv2.resize(self.cv_img, (200, 200))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            corners = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=50,
                qualityLevel=0.01,
                minDistance=10
            )
            if corners is not None:
                corners = np.intp(corners)

                out = img.copy()
                for c in corners:
                    x, y = c.ravel()
                    cv2.circle(out, (x, y), 4, (0, 255, 0), -1)
            else:
                out = img.copy()
                self.status.config(text="No corners detected.")

            rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb)
            tk_img = ImageTk.PhotoImage(img_pil)

            if not hasattr(self, 'result_img_label'):
                self.result_img_label = Label(self.win, bg="white")
                self.result_img_label.place(x=600, y=230)

            self.result_img_label.config(image=tk_img)
            self.result_img_label.image = tk_img
            self.status.config(text="Shi–Tomasi corners detected!" if corners is not None else self.status.cget("text"))

        except Exception as e:
            self.status.config(text=f"Error: {e}")


    #todo: face detection by haar cascade:
    def face_detection(self):
        if self.cv_img is None:
            self.status.config(text="Please upload an image first.")
            return

        try:
            img = cv2.resize(self.cv_img, (200, 200))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            out = img.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)

            rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb)
            tk_img = ImageTk.PhotoImage(img_pil)

            if not hasattr(self, 'result_img_label'):
                self.result_img_label = Label(self.win, bg="white")
                self.result_img_label.place(x=600, y=230)

            self.result_img_label.config(image=tk_img)
            self.result_img_label.image = tk_img
            self.status.config(text=f"Detected {len(faces)} face(s).")

        except Exception as e:
            self.status.config(text=f"Error: {e}")

    #todo: detect the coordinate color:
    def detect_coordinates(self):
        if self.cv_img is None:
            self.status.config(text="Please upload an image first.")
            return

        display_img = cv2.resize(self.cv_img, (400, 300))
        window_name = "Click to detect coords/color"
        cv2.namedWindow(window_name)

        def click_event(event, x, y, flags, param):
            nonlocal display_img
            if event == cv2.EVENT_LBUTTONDOWN:
                text = f"({x}, {y})"
                cv2.putText(
                    display_img, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1
                )
                cv2.imshow(window_name, display_img)

            elif event == cv2.EVENT_RBUTTONDOWN:
                b, g, r = display_img[y, x]
                color_text = f"BGR=({b},{g},{r})"
                cv2.putText(
                    display_img, color_text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                )
                cv2.imshow(window_name, display_img)

        cv2.setMouseCallback(window_name, click_event)

        self.status.config(text="Coordinate detection active: Left-click → coords, Right-click → BGR")

        while True:
            cv2.imshow(window_name, display_img)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

        cv2.destroyWindow(window_name)
        self.status.config(text="Coordinate detection ended.")
        
    #todo: reverse a video:
    def reverse_video(self):
        import threading

        if not hasattr(self, 'image_path') or not self.image_path.lower().endswith((".mp4", ".avi", ".mov")):
            self.status.config(text="Please upload a video first.")
            return

        def run_reverse():
            cap = cv2.VideoCapture(self.image_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Resize to fit your label
                frame = cv2.resize(frame, (200, 200))
                # Convert to RGB for Tkinter
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            cap.release()

            if not frames:
                self.status.config(text="No frames found in video.")
                return

            self.status.config(text="Playing video in reverse...")

            for f in reversed(frames):
                img_pil = Image.fromarray(f)
                tk_img = ImageTk.PhotoImage(img_pil)

                if not hasattr(self, 'result_img_label'):
                    self.result_img_label = Label(self.win, bg="white")
                    self.result_img_label.place(x=600, y=230)

                self.result_img_label.config(image=tk_img)
                self.result_img_label.image = tk_img

                self.win.update()
                self.win.after(30)

            self.status.config(text="Reverse playback finished.")

        threading.Thread(target=run_reverse, daemon=True).start()


    # todo: full body detection:
    def full_body_detection(self):
        import threading

        if not hasattr(self, 'image_path') or not self.image_path.lower().endswith((".mp4", ".avi", ".mov")):
            self.status.config(text="Please upload a video first.")
            return

        def run_body_detect():
            cascade_path = cv2.data.haarcascades + 'haarcascade_fullbody.xml'
            body_cascade = cv2.CascadeClassifier(cascade_path)

            cap = cv2.VideoCapture(self.image_path)
            if not cap.isOpened():
                self.status.config(text="Failed to open video.")
                return

            self.status.config(text="Running full-body detection... Press 'q' to stop.")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (250, 250))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
                for (x, y, w, h) in bodies:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 50, 50), 2)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(rgb)
                tk_img = ImageTk.PhotoImage(img_pil)

                if not hasattr(self, 'result_img_label'):
                    self.result_img_label = Label(self.win, bg="white")
                    self.result_img_label.place(x=600, y=230)

                self.result_img_label.config(image=tk_img)
                self.result_img_label.image = tk_img

                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break

            cap.release()
            self.status.config(text="Full-body detection finished.")

        threading.Thread(target=run_body_detect, daemon=True).start()

    #todo: smile ane eye detection:
    def detect_smile(self):
        if self.cv_img is None:
            self.status.config(text="Please upload an image first.")
            return
        try:
            # Prepare
            img = cv2.resize(self.cv_img, (200, 200))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Load cascades
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            smile_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_smile.xml'
            )

            # Detect faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            out = img.copy()

            # Within each face, detect smiles
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = out[y:y+h, x:x+w]

                smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=20)
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 1)

            # Display
            rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb)
            tk_img = ImageTk.PhotoImage(img_pil)

            if not hasattr(self, 'result_img_label'):
                self.result_img_label = Label(self.win, bg="white")
                self.result_img_label.place(x=600, y=230)

            self.result_img_label.config(image=tk_img)
            self.result_img_label.image = tk_img
            self.status.config(text=f"Smile detection: {sum(len(smile_cascade.detectMultiScale(gray[y:y+h, x:x+w])) for x,y,w,h in faces)} smiles found.")

        except Exception as e:
            self.status.config(text=f"Error: {e}")


    def detect_eyes(self):
        if self.cv_img is None:
            self.status.config(text="Please upload an image first.")
            return
        try:
            # Prepare
            img = cv2.resize(self.cv_img, (200, 200))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Load cascades
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )

            # Detect faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            out = img.copy()

            # Within each face, detect eyes
            eyes_count = 0
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = out[y:y+h, x:x+w]

                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=15)
                eyes_count += len(eyes)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 1)

            # Display
            rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb)
            tk_img = ImageTk.PhotoImage(img_pil)

            if not hasattr(self, 'result_img_label'):
                self.result_img_label = Label(self.win, bg="white")
                self.result_img_label.place(x=600, y=230)

            self.result_img_label.config(image=tk_img)
            self.result_img_label.image = tk_img
            self.status.config(text=f"Eye detection: {eyes_count} eyes found.")

        except Exception as e:
            self.status.config(text=f"Error: {e}")

    #todo: webcam detection:        
    def webcam_object_detection(self):
        import threading
        import numpy as np

        ctrl = Toplevel(self.win)
        ctrl.title("Webcam Controls")
        Label(ctrl, text="Lower Hue").pack()
        lh = Scale(ctrl, from_=0, to=179, orient=HORIZONTAL); lh.set(0); lh.pack()
        Label(ctrl, text="Upper Hue").pack()
        uh = Scale(ctrl, from_=0, to=179, orient=HORIZONTAL); uh.set(179); uh.pack()
        Label(ctrl, text="Lower Sat").pack()
        ls = Scale(ctrl, from_=0, to=255, orient=HORIZONTAL); ls.set(0); ls.pack()
        Label(ctrl, text="Upper Sat").pack()
        us = Scale(ctrl, from_=0, to=255, orient=HORIZONTAL); us.set(255); us.pack()
        Label(ctrl, text="Lower Val").pack()
        lv = Scale(ctrl, from_=0, to=255, orient=HORIZONTAL); lv.set(0); lv.pack()
        Label(ctrl, text="Upper Val").pack()
        uv = Scale(ctrl, from_=0, to=255, orient=HORIZONTAL); uv.set(255); uv.pack()
        Label(ctrl, text="Threshold").pack()
        th = Scale(ctrl, from_=0, to=255, orient=HORIZONTAL); th.set(127); th.pack()

        def run_webcam():
            cap = cv2.VideoCapture(0)
            self.status.config(text="Webcam detection running. Close controls to stop.")
            while ctrl.winfo_exists():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                h, w = 200, 200
                frame = cv2.resize(frame, (w, h))
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                lower = np.array([lh.get(), ls.get(), lv.get()])
                upper = np.array([uh.get(), us.get(), uv.get()])
                mask = cv2.inRange(hsv, lower, upper)
                _, binary = cv2.threshold(mask, th.get(), 255, cv2.THRESH_BINARY)

                contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                out = frame.copy()
                cv2.drawContours(out, contours, -1, (0, 255, 0), 2)

                rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(rgb)
                tk_img = ImageTk.PhotoImage(img_pil)
                if not hasattr(self, 'result_img_label'):
                    self.result_img_label = Label(self.win, bg="black")
                    self.result_img_label.place(x=600, y=230)
                self.result_img_label.config(image=tk_img)
                self.result_img_label.image = tk_img

                self.win.update_idletasks()
                self.win.update()

            cap.release()
            self.status.config(text="Webcam detection stopped.")

        threading.Thread(target=run_webcam, daemon=True).start()


        
###############################!...main widgets...############################################### 
    def create_widgets(self):
        #title: 
        title= Label(self.win,text="DIPify - A Smart Image Editing Tool",font=("Times New Roman",25,"bold"),bg="#122547",fg="white")
        title.place(x=0,y=0,width=1000,height=40)

        #! panel and buttons:
        #todo: panel:
        paned = PanedWindow(self.win, orient=HORIZONTAL, sashrelief=RAISED, bd=3, showhandle=True, sashwidth=8)
        paned.place(x=0,y=45,width=1000,height=50)

        # First frame inside PanedWindow
        panel1 = Frame(paned, bg="lightblue", width=200, height=300)
        paned.add(panel1)  # Add to PanedWindow

        # Second frame inside PanedWindow
        panel2 = Frame(paned, bg="lightgreen", width=300, height=300)
        paned.add(panel2)
        # third frame inside PanedWindow
        panel3 = Frame(paned, bg="lightpink", width=300, height=300)
        paned.add(panel3)
        # fourth frame inside PanedWindow
        panel4 = Frame(paned, bg="lavender", width=300, height=300)
        paned.add(panel4)
        # five frame inside PanedWindow
        panel5 = Frame(paned, bg="honeydew", width=300, height=300)
        paned.add(panel5)
        # six frame inside PanedWindow
        panel6 = Frame(paned, bg="#FFDAB9", width=300, height=300)
        paned.add(panel6)

#################################!... panel buttons... #############################################
        #todo: panel_buttons: 
        resize_btn = Button(panel1, text="Resize", relief="sunken", command=self.resize_image)
        resize_btn.place(x=10, y=10)

        shapes_btn = Button(panel1, text="Draw Shapes", relief="sunken", command=self.draw_shapes)
        shapes_btn.place(x=100, y=10)

        Arithmetic_Ops = Button(panel1, text="Apply Operation", relief="sunken", command=self.arithmetic_ops)
        Arithmetic_Ops.place(x=230, y=10)
        self.arith_op_var = StringVar(self.win)
        self.arith_op_var.set("Add")
        arith_options = ["Add", "Subtract", "Multiply", "Divide"]
        op_menu = OptionMenu(panel1, self.arith_op_var, *arith_options)
        op_menu.place(x=355, y=11,width=70,height=27)

        bitwise_btn = Button(panel1, text="Apply Bitwise", relief="sunken", command=self.bitwise_ops)
        bitwise_btn.place(x=440, y=10)
        self.bitwise_op_var = StringVar(self.win)
        self.bitwise_op_var.set("NOT")  # default
        bitwise_ops_list = ["NOT", "AND", "OR", "XOR"]
        bitwise_menu = OptionMenu(panel1, self.bitwise_op_var, *bitwise_ops_list)
        bitwise_menu.place(x=550, y=11,width=60,height=27)

        edge_btn = Button(panel1, text="Edge Detect", relief="sunken", command=self.edge_detection)
        edge_btn.place(x=625, y=10)

        rotate_btn = Button(panel1, text="Rotate", relief="sunken", command=self.rotate_image)
        rotate_btn.place(x=750, y=10)
        
        self.rotation_angle = IntVar(self.win)
        self.rotation_angle.set(30) 
        angle_spin = Spinbox(panel1, from_=0, to=360, width=5, textvariable=self.rotation_angle)
        angle_spin.place(x=820, y=10)

        save_btn = Button(panel1, text="Save", relief="sunken", command=self.save_image)
        save_btn.place(x=900, y=10)

        blur_btn = Button(panel2, text="Apply Blur", relief="sunken", command=self.apply_blur)
        blur_btn.place(x=10, y=10)
        self.blur_method_var = StringVar(self.win)
        self.blur_method_var.set("Gaussian")  
        blur_options = ["Gaussian", "Median", "Bilateral"]
        blur_menu = OptionMenu(panel2, self.blur_method_var, *blur_options)
        blur_menu.place(x=105, y=11,width=80,height=27)

        border_btn = Button(panel2, text="Add Border", relief="sunken", command=self.add_border)
        border_btn.place(x=200, y=10)

        video_btn = Button(panel2, text="upload Video", relief="sunken", command=self.play_video_in_label)
        video_btn.place(x=320, y=10)
        stop_btn = Button(panel2, text="Stop Video", relief="sunken", command=self.stop_video)
        stop_btn.place(x=420, y=10)
         
        camera_btn = Button(panel2, text="Open Camera", relief="sunken", command=self.open_camera)
        camera_btn.place(x=540, y=10)

        morph_btn = Button(panel2, text="Apply Morphology", relief="sunken", command=self.morphological_ops)
        morph_btn.place(x=675, y=10)
        self.morph_op_var = StringVar(self.win)
        self.morph_op_var.set("Erosion") 
        morph_ops = ["Erosion", "Dilation", "Opening", "Closing", "Gradient", "Top Hat", "Black Hat"]
        morph_menu = OptionMenu(panel2, self.morph_op_var, *morph_ops)
        morph_menu.place(x=820, y=11, width=80,height=27)

        crop_btn = Button(panel2, text="Crop", relief="sunken", command=self.crop_image)
        crop_btn.place(x=910,y=10)

        translate_btn = Button(panel3, text="Translate", relief="sunken", command=self.translate_image)
        translate_btn.place(x=10, y=10)

        color_btn = Button(panel3, text="Convert Color", command=self.cvt_color_convert)
        color_btn.place(x=110, y=10)
        self.color_conv_var = StringVar(self.win)
        self.color_conv_var.set("GRAY")
        color_options = ["GRAY", "HSV", "LUV", "LAB", "HLS", "YCrCb"]
        color_menu = OptionMenu(panel3, self.color_conv_var, *color_options)
        color_menu.place(x=225, y=11, width=90,height=27)

        blank_btn = Button(panel3, text="Blank Img", relief="sunken", command=self.create_blank_image)
        blank_btn.place(x=323, y=10)
        self.blank_mode_var = StringVar(self.win)
        self.blank_mode_var.set("White") 
        blank_options = ["White", "Black"]
        blank_menu = OptionMenu(panel3, self.blank_mode_var, *blank_options)
        blank_menu.place(x=408, y=11, width=80,height=27)

        roi_btn = Button(panel3, text="ROI Clone", relief="sunken", command=self.roi_clone)
        roi_btn.place(x=495, y=10)

        save_video_btn = Button(panel3, text="Save Video", relief="sunken", command=self.save_uploaded_video)
        save_video_btn.place(x=600, y=10) 
  
        filter_btn = Button(panel3, text="Filter Color", relief="sunken", command=self.filter_color)
        filter_btn.place(x=711, y=10)
        self.color_filter_var = StringVar(self.win)
        self.color_filter_var.set("Blue")
        filter_options = ["Blue", "Green", "Skin"]
        filter_menu = OptionMenu(panel3, self.color_filter_var, *filter_options)
        filter_menu.place(x=810,y=11,width=80,height=27)

        moments_btn = Button(panel3, text="Moments", relief="sunken", command=self.contour_moments)
        moments_btn.place(x=895, y=10)
      
        thresh_btn = Button(panel4, text="Threshold", relief="sunken", command=self.threshold_ops)
        thresh_btn.place(x=10, y=10)
        self.thresh_var = StringVar(self.win)
        self.thresh_var.set("Simple")  # default
        thresh_options = ["Simple", "Otsu", "Adaptive"]
        thresh_menu = OptionMenu(panel4, self.thresh_var, *thresh_options)
        thresh_menu.place(x=100, y=11, width=80, height=27)

        hist_btn = Button(panel4, text="Histogram", relief="sunken", command=self.histogram_ops)
        hist_btn.place(x=190, y=10)


        eq_btn = Button(panel4, text="Equalize", relief="sunken", command=self.equalize_histogram)
        eq_btn.place(x=300, y=10)
        self.equalize_var = StringVar(self.win)
        self.equalize_var.set("Global")
        equalize_options = ["Global", "CLAHE"]
        eq_menu = OptionMenu(panel4, self.equalize_var, *equalize_options)
        eq_menu.place(x=380, y=11, width=80,height=27)

        mouse_btn = Button(panel4, text="Mouse Draw", relief="sunken", command=self.mouse_event_draw)
        mouse_btn.place(x=470, y=10)

        contour_btn = Button(panel4, text="Contour", relief="sunken", command=self.contour_detection)
        contour_btn.place(x=590, y=10)

        hull_btn = Button(panel4, text="Convex Hull", relief="sunken", command=self.convex_hull)
        hull_btn.place(x=685, y=10)

        bg_remove_btn = Button(panel4, text="BG Remove", relief="sunken", command=self.background_removal)
        bg_remove_btn.place(x=800, y=10)

        grabcut_btn = Button(panel4, text="GrabCut", relief="sunken", command=self.grabcut_segmentation)
        grabcut_btn.place(x=910, y=10)

        hough_btn = Button(panel5, text="Hough Transform", relief="sunken", command=self.hough_transform)
        hough_btn.place(x=10, y=10)
        self.hough_var = StringVar(self.win)
        self.hough_var.set("Standard")
        hough_options = ["Standard", "Probabilistic"]
        hough_menu = OptionMenu(panel5, self.hough_var, *hough_options)
        hough_menu.place(x=150, y=11, width=100,height=27)

        temp_btn = Button(panel5, text="Template Match", relief="sunken", command=self.template_matching)
        temp_btn.place(x=255, y=10)

        track_btn = Button(panel5, text="Track Object", relief="sunken", command=self.track_object)
        track_btn.place(x=395, y=10)

        harris_btn = Button(panel5, text="Harris Corner", relief="sunken", command=self.harris_corners)
        harris_btn.place(x=515, y=10)

        shi_btn = Button(panel5, text="Shi_Tomasi", relief="sunken", command=self.shi_tomasi_corners)
        shi_btn.place(x=640, y=10)

        face_btn = Button(panel5, text="Face Detect", relief="sunken", command=self.face_detection)
        face_btn.place(x=750, y=10)

        coord_btn = Button(panel5, text="Detect Coords", relief="sunken", command=self.detect_coordinates)
        coord_btn.place(x=865, y=10)

        rev_btn = Button(panel6, text="Reverse Video", relief="sunken", command=self.reverse_video)
        rev_btn.place(x=10, y=10)

        body_btn = Button(panel6, text="Body Detect", relief="sunken", command=self.full_body_detection)
        body_btn.place(x=140, y=10)

        smile_btn = Button(panel6, text="Smile Detect", relief="sunken", command=self.detect_smile)
        smile_btn.place(x=260, y=10)

        eyes_btn = Button(panel6, text="Eye Detect", relief="sunken", command=self.detect_eyes)
        eyes_btn.place(x=380, y=10)

        wcam_btn = Button(panel6, text="Webcam Detect", relief="sunken", command=self.webcam_object_detection)
        wcam_btn.place(x=490, y=10)



        #! main_frame:
        body= Frame(self.win,bg="#E0E4EE")
        body.place(x=50,y=105,width=900,height=450)

        #todo: input_frame:
        input_frame = Frame(body, bg="#B9C9F0")
        input_frame.place(x=100, y=80,width=300, height=300)

        # Image label to display uploaded image
        self.input_img_label = Label(input_frame, bg="white")
        self.input_img_label.place(relx=0.5, rely=0.5, anchor=CENTER)

        #todo: upload button:
        def upload_img():
            path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
            if not path:
                return

            self.image_path = path
            self.cv_img = cv2.imread(path)

            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.resize(img_rgb, (250, 250))  # Resize to fit the frame
            img_pil = Image.fromarray(img_rgb)
            self.tk_img = ImageTk.PhotoImage(img_pil)

            # Show image in label
            self.input_img_label.config(image=self.tk_img)
            self.status.config(text="Image uploaded successfully!")

        
        upload_btn = Button(body, text="Upload Image",font=("Arial", 13, "bold"),command=upload_img)
        upload_btn.place(x=185,y=40,height=50)


        
        # #! label option:
        # choose= Label(body,text="",bg="#97B3CF")
        # choose.place(x=450,y=20,height=30)

        #todo: result_frame:
        result_frame = Frame(body, bg="#B9C9F0")
        result_frame.place(x=500, y=80,width=300, height=300)


        #todo: coverted image:
        result_btn = Button(body, text="Convert",font=("Arial", 13, "bold"))
        result_btn.place(x=600,y=40,height=50,width=100)






        #! Status Bar
        self.status = Label(self.win, text="Welcome to DIPify!", bg="lightgray", anchor=W)
        self.status.place(x=0,y=570,width=1000,height=30)
            



if __name__ == "__main__":
    root = Tk()
    app = DIPify(root)
    root.mainloop()



