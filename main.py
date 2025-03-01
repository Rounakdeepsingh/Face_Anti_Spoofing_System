from ultralytics import YOLO
import cv2
import cvzone
import time
import threading
import tkinter as tk
from tkinter import ttk, Label, Button, PhotoImage, messagebox
from PIL import Image, ImageTk
import numpy as np  # Import numpy

confidence_threshold = 0.8
stop_thread = False
detection_running = False
initial_wait_time = 1  # Even faster initialization
max_detection_time = 10  # Recognize for only 10 seconds
root = None  # Define root globally
background_color = "#F0F8FF"  # Alice Blue (soft, subtle blue)
frame_width = 640
frame_height = 480
font_color = "#2C3E50"  # Midnight Blue (dark, readable, professional)
camera_icon_path = r"C:\Users\lenovo\Downloads\g1320.png"  # Replace with the actual path to your camera icon image
face_recognized = False  # Add a flag to track face recognition
iou_threshold = 0.5  # Added IOU threshold
outline_color = (0, 0, 0)  # Black outline color
button_font_color = "black" # Define button text color


def run_detection(video_label, fps_label, result_label, start_button, stop_button,
                  camera_icon_tk):  # Add camera_icon_tk
    global stop_thread, detection_running, initial_wait_time, max_detection_time, root, frame_width, frame_height, face_recognized, iou_threshold, class_names, font_color

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Camera Error", "Could not open camera. Please check your camera connection.")
        result_label.config(text="Error: Could not open camera", foreground="red")
        start_button.config(state=tk.NORMAL)
        stop_button.config(state=tk.DISABLED)
        detection_running = False
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    try:
        model = YOLO("../models/best.pt")
        class_names = ["Fake", "Real"]
    except Exception as e:
        messagebox.showerror("Model Error",
                             f"Error loading model: {e}.  Make sure 'best.pt' exists in the models folder.")
        result_label.config(text=f"Error loading model: {e}", foreground="red")
        start_button.config(state=tk.NORMAL)
        stop_button.config(state=tk.DISABLED)
        detection_running = False
        cap.release()
        return

    start_time = time.time()
    wait_period_ended = False
    detection_start_time = None
    detection_end_time = start_time + max_detection_time  # Calculate the detection end time in advance
    detection_result = None
    detection_confidence = None

    try:
        while not stop_thread and detection_running and (
                time.time() < detection_end_time):  # change the conditional
            success, frame = cap.read()
            if not success:
                messagebox.showerror("Camera Error", "Failed to read frame from camera.")
                break

            if not wait_period_ended:
                if detection_start_time is None:
                    detection_start_time = time.time()
                    result_label.config(text="Initializing detection...", foreground=font_color)
                    if root:
                        root.update()

                elapsed_time = time.time() - detection_start_time

                if elapsed_time < initial_wait_time:
                    cv2.putText(frame, "Initializing...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                                cv2.LINE_AA)
                    result_label.config(
                        text=f"Initializing...({elapsed_time:.1f}/{initial_wait_time:.1f} seconds)",
                        foreground=font_color)
                    if root:
                        root.update()

                else:
                    wait_period_ended = True
                    result_label.config(text="Recognizing Face Please wait ...", foreground="Green") #changed here!

                    if root: # To render all text element that are dynamic- remember code elements must have it
                        root.update()
                    start_time = time.time()

            if wait_period_ended:
                results = model(frame, stream=True)
                boxes = []  # Store detection boxes
                confidences = []
                class_ids = []

                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = box.conf[0].item()
                        cls = int(box.cls[0].item())
                        boxes.append([x1, y1, x2 - x1, y2 - y1])  # width and height are required here. Changed as cv2 function
                        confidences.append(conf)
                        class_ids.append(cls)

                # Convert lists to numpy arrays
                boxes = np.array(boxes, dtype=np.float32)  # ensure it's float32
                confidences = np.array(confidences, dtype=np.float32)  # ensure it's float32

                # Perform Non-Maximum Suppression using cv2.dnn.NMSBoxes
                indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), confidence_threshold, iou_threshold)

                if indices is not None and len(indices) > 0:

                    if isinstance(indices, tuple): # add this error handling
                        indices = indices[0]

                    for i in indices: #Use iterative extraction, the cv2 function can extract several indices.
                         index = i #Access element

                         box = boxes[index] #Get Box data using data Access from iterative instance of best bounding practices as followed in computer coding with security best coding practices - ensures the integrity

                         x1, y1, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3]) #Convert data that you get to actual values - data sanitation

                         conf = confidences[index] # Get data of appropriate elements here and the rest get from the data
                         cls = class_ids[index] # The keys values such as cls also are passed

                         face_recognized = True # Once code reached - meaning all worked perfectly = therefore image is not corrupt - model well, everything fit perfect in slot == image - can be used by the agent!

                         color = (0, 255, 0) if class_names[cls] == 'Real' else (0, 0, 255) # RGB Colors Code -> data passed well now apply
                         cvzone.cornerRect(frame, (x1, y1, w, h), colorC =color, colorR=color) # Data well

                         cvzone.putTextRect(frame, f'{class_names[cls].upper()} {conf:.1%}', (max(0, x1), max(35, y1)), scale = 1, thickness=2, colorR = color, colorB=color)

                        #After reaching now all values have data here: means no data corruption to here can happen -> now set the detection_

                         detection_result = class_names[cls].upper() # Apply with now after tested -> all elements reach == test Passed code! No human to look! Its true
                         detection_confidence = f"{conf:.1%}" # Confident that this one worked
                         result_label.config(text="Recognizing Face Please wait ...", foreground="Green")


                else:
                    detection_result = "No Face"  # No face is there to capture
                    detection_confidence = 0  # Set confidence to zero
                    face_recognized = False
                    result_label.config(text=f"No Face Detected", foreground="Red")

            new_frame_time = time.time()
            fps = 1 / (new_frame_time - start_time) if (new_frame_time - start_time) != 0 else 0
            fps_label.config(text=f"FPS: {fps:.2f}", foreground=font_color)

            if frame is not None:
                mask = create_rounded_mask(frame_width, frame_height, radius=20)
                masked_frame = apply_mask(frame, mask)

                img = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img_tk = ImageTk.PhotoImage(image=img)

                video_label.config(image=img_tk)
                video_label.image = img_tk

    except Exception as e:
        messagebox.showerror("Unexpected Error", f"An unexpected error occurred: {e}")
        result_label.config(text=f"An error occurred: {e}", foreground="red")
    finally:
        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        detection_running = False
        start_button.config(state=tk.NORMAL)
        stop_button.config(state=tk.DISABLED)

        # This code is now OUTSIDE the if detection_running block
        # And will ALWAYS run when the function exits.
        if face_recognized:
            formatted_result = f"Final Result:\n\nFace Recognized: {detection_result}\nConfidence: {detection_confidence}"
        else:
            formatted_result = "Final Result:\n\nNo face recognized within the time limit."

        messagebox.showinfo("Detection Complete", formatted_result)  # Display result in a message box
        result_label.config(text="Detection Completed", foreground=font_color)

        # Reset the video feed to the camera icon:
        video_label.config(image=camera_icon_tk)  # Reset to the camera icon
        video_label.image = camera_icon_tk  # Keep the reference


def create_rounded_mask(width, height, radius):
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (0, 0), (width, height), 255, -1)

    cv2.circle(mask, (radius, radius), radius, 0, -1)
    cv2.circle(mask, (width - radius, radius), radius, 0, -1)
    cv2.circle(mask, (radius, height - radius), radius, 0, -1)
    cv2.circle(mask, (width - radius, height - radius), radius, 0, -1)

    cv2.rectangle(mask, (radius, 0), (width - radius, height), 255, -1)
    cv2.rectangle(mask, (0, radius), (width, height - radius), 255, -1)
    return mask


def apply_mask(frame, mask):
    masked = cv2.bitwise_and(frame, frame, mask=mask)
    return masked


def start_detection(video_label, fps_label, result_label, start_button, stop_button, camera_icon_tk):
    global stop_thread, detection_running, face_recognized

    if not detection_running:
        stop_thread = False
        detection_running = True
        face_recognized = False  # Reset the face recognition flag before each detection

        start_button.config(state=tk.DISABLED)
        stop_button.config(state=tk.NORMAL)

        detection_thread = threading.Thread(target=run_detection,
                                            args=(video_label, fps_label, result_label, start_button, stop_button,
                                                  camera_icon_tk))  # pass camera_icon_tk
        detection_thread.daemon = True
        detection_thread.start()


def stop_detection(start_button, stop_button, camera_icon_tk, video_label):
    global stop_thread, detection_running
    stop_thread = True
    detection_running = False
    start_button.config(state=tk.NORMAL)
    stop_button.config(state=tk.DISABLED)

    # Display camera icon after stopping
    video_label.config(image=camera_icon_tk)
    video_label.image = camera_icon_tk  # Keep a reference!


def create_gui():
    global root, video_label, fps_label, start_button, stop_button, result_label, background_color, font_color, camera_icon_path, button_font_color

    if root is None:
        root = tk.Tk()
        # CHANGE THE HEADING HERE
        root.title("Anti Spoofing System") # MARK THE LINE FROMWHERE WE CAN CHANGE THE HEADING
        root.configure(bg=background_color)

    style = ttk.Style(root)

    # Define button styles for Start and Stop
    style.configure("Start.TButton", padding=10, font=('Arial', 14), background="#2ECC71", foreground=button_font_color,
                    borderwidth=0, relief="flat", highlightthickness=0)  # Emerald green
    style.map("Start.TButton",
              background=[('active', '#27AE60')],  # Darker green on hover
              foreground=[('active', button_font_color)])  # Black text on hover

    style.configure("Stop.TButton", padding=10, font=('Arial', 14), background="#E74C3C", foreground=button_font_color,
                    borderwidth=0, relief="flat", highlightthickness=0)  # Alizarin red
    style.map("Stop.TButton",
              background=[('active', '#C0392B')],  # Darker red on hover
              foreground=[('active', button_font_color)])  # Black text on hover

    style.configure("TLabel", padding=5, font=('Arial', 12, "bold"), background=background_color,
                    foreground=font_color)  # Bolding the font
    style.configure("TFrame", background=background_color)

    # Main Frame
    main_frame = ttk.Frame(root, padding="20")
    main_frame.pack(expand=True, fill="both")

    # Title Label
    title_label = ttk.Label(main_frame, text="Anti Spoofing System", font=('Arial', 24, 'bold'))
    title_label.pack(pady=(0, 20))

    # Video Frame
    video_frame = ttk.Frame(main_frame)
    video_frame.pack(pady=(0, 10))

    # Video Label (Camera Feed)
    video_label = Label(video_frame, background=background_color)
    video_label.pack()

    # Load camera icon
    try:
        camera_icon = Image.open(camera_icon_path)
        camera_icon = camera_icon.resize((frame_width, frame_height), Image.LANCZOS)  # resize image to prevent issues
        camera_icon_tk = ImageTk.PhotoImage(camera_icon, master=root)  # Master is essential
        video_label.config(image=camera_icon_tk)  # Initial display
        video_label.image = camera_icon_tk  # Keep a reference!
    except FileNotFoundError:
        messagebox.showerror("Error", f"Camera icon not found at {camera_icon_path}")
        camera_icon_tk = None  # set to None
        # Fallback: display grey image
        blank_img = Image.new("RGB", (frame_width, frame_height), background_color)  # Grey background for blank image
        blank_img_tk = ImageTk.PhotoImage(blank_img, master=root)  # Master is required for correct initialization
        video_label.config(image=blank_img_tk)
        video_label.image = blank_img_tk

    # Status Frame
    status_frame = ttk.Frame(main_frame)
    status_frame.pack(pady=(10, 0))

    # FPS Label
    fps_label = ttk.Label(status_frame, text="FPS: 0.00")
    fps_label.grid(row=0, column=0, padx=10, sticky="w")

    # Result label
    result_label = ttk.Label(status_frame, text="Ready")
    result_label.grid(row=0, column=1, padx=10, sticky="e")

    # Button Frame
    button_frame = ttk.Frame(main_frame)
    button_frame.pack(pady=(20, 0))

    start_button = tk.Button(button_frame, text="Start Detection", font=('Arial', 14, 'bold'),
                             bg="#2ECC71", fg=button_font_color, activebackground="#27AE60", activeforeground=button_font_color,
                             relief="flat", bd=0, padx=20, pady=10,
                             command=lambda: start_detection(video_label, fps_label, result_label, start_button,
                                                             stop_button, camera_icon_tk))  # pass camera_icon_tk
    start_button.grid(row=0, column=0, padx=10)

    # Stop Button
    stop_button = tk.Button(button_frame, text="Stop Detection", font=('Arial', 14, 'bold'),
                            bg="#E74C3C", fg=button_font_color, activebackground="#C0392B", activeforeground=button_font_color,
                            relief="flat", bd=0, padx=20, pady=10,
                            command=lambda: stop_detection(start_button, stop_button, camera_icon_tk, video_label))
    stop_button.grid(row=0, column=1, padx=10)
    stop_button.config(state=tk.DISABLED)

    return video_label, fps_label, result_label, start_button, stop_button, root, camera_icon_tk  # Return camera_icon_tk


if __name__ == "__main__":
    video_label, fps_label, result_label, start_button, stop_button, root, camera_icon_tk = create_gui()  # Get camera_icon_tk

    root.mainloop()
    cv2.destroyAllWindows()