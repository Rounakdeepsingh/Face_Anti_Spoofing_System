from ultralytics import YOLO
import cv2
import cvzone
import time
import threading
import tkinter as tk
from tkinter import ttk, Label, Button, PhotoImage, messagebox
from PIL import Image, ImageTk
import numpy as np

# Global Variables
confidence_threshold = 0.8
stop_thread = False
detection_running = False
initial_wait_time = 1
max_detection_time = 10
root = None
background_color = "#F0F8FF"
frame_width = 640
frame_height = 480
font_color = "#2C3E50"
camera_icon_path = "camera_icon.png"  # Replace with a valid path!
face_recognized = False
border_radius = 20  # Radius for rounded corners


def run_detection(video_label, fps_label, result_label, start_button, stop_button, camera_icon_tk, video_frame):
    global stop_thread, detection_running, face_recognized, border_radius

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Camera Error", "Could not open camera.")
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
        messagebox.showerror("Model Error", f"Error loading model: {e}")
        result_label.config(text=f"Error loading model: {e}", foreground="red")
        start_button.config(state=tk.NORMAL)
        stop_button.config(state=tk.DISABLED)
        detection_running = False
        cap.release()
        return

    start_time = time.time()
    wait_period_ended = False
    detection_end_time = start_time + max_detection_time
    detection_result = None
    detection_confidence = None

    try:
        while not stop_thread and detection_running and (time.time() < detection_end_time):
            success, frame = cap.read()
            if not success:
                messagebox.showerror("Camera Error", "Failed to read frame from camera.")
                break

            if not wait_period_ended:
                elapsed_time = time.time() - start_time
                if elapsed_time < initial_wait_time:
                    result_label.config(text=f"Initializing... ({elapsed_time:.1f}/{initial_wait_time:.1f} sec)",
                                        foreground=font_color)
                else:
                    wait_period_ended = True
                    result_label.config(text="Detection started", foreground=font_color)

            if wait_period_ended:
                results = model(frame, stream=True)

                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = box.conf[0].item()

                        if conf > confidence_threshold:
                            cls = int(box.cls[0].item())
                            face_recognized = True

                            color = (0, 255, 0) if class_names[cls] == 'Real' else (0, 0, 255)
                            cvzone.cornerRect(frame, (x1, y1, x2 - x1, y2 - y1), colorC=color, colorR=color)
                            cvzone.putTextRect(frame, f'{class_names[cls].upper()} {conf:.1%}',
                                               (max(0, x1), max(35, y1)), scale=1, thickness=2, colorR=color)

                            detection_result = class_names[cls].upper()
                            detection_confidence = f"{conf:.1%}"

                        if r.boxes:
                            result_label.config(text=f"Result: {detection_result}, Confidence: {detection_confidence}",
                                                foreground=font_color)

            fps = 1 / (time.time() - start_time) if (time.time() - start_time) !=0 else 0
            fps_label.config(text=f"FPS: {fps:.2f}", foreground=font_color)

            if frame is not None:  # Check if frame is valid
                mask = create_rounded_mask(frame_width, frame_height, border_radius)  # Use border radius
                masked_frame = apply_mask(frame, mask)

                img = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img_tk = ImageTk.PhotoImage(image=img)

                video_label.config(image=img_tk)
                video_label.image = img_tk  # Keep a reference!

    except Exception as e:
        messagebox.showerror("Unexpected Error", f"An error occurred: {e}")
    finally:
        if cap is not None and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        detection_running = False
        start_button.config(state=tk.NORMAL)
        stop_button.config(state=tk.DISABLED)

        formatted_result = f"Final Result:\n\nFace Recognized: {detection_result}\nConfidence: {detection_confidence}" \
            if face_recognized else "Final Result:\n\nNo face recognized within the time limit."

        messagebox.showinfo("Detection Complete", formatted_result)
        result_label.config(text="Detection Completed", foreground=font_color)

        # Display rounded camera icon after stopping
        video_label.config(image=camera_icon_tk)
        video_label.image = camera_icon_tk


def create_rounded_mask(width, height, radius):
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (0, 0), (width, height), 255, -1)

    cv2.circle(mask, (radius, radius), radius, 0, -1)  # Top left
    cv2.circle(mask, (width - radius, radius), radius, 0, -1)  # Top right
    cv2.circle(mask, (radius, height - radius), radius, 0, -1)  # Bottom left
    cv2.circle(mask, (width - radius, height - radius), radius, 0, -1)  # Bottom right

    cv2.rectangle(mask, (radius, 0), (width - radius, height), 255, -1)  # Fill top and bottom
    cv2.rectangle(mask, (0, radius), (width, height - radius), 255, -1)  # Fill left and right

    return mask


def apply_mask(frame, mask):
    masked = cv2.bitwise_and(frame, frame, mask=mask)
    return masked


def start_detection(video_label, fps_label, result_label, start_button, stop_button, camera_icon_tk, video_frame):
    global stop_thread, detection_running, face_recognized

    if not detection_running:
        stop_thread = False
        detection_running = True
        face_recognized = False

        start_button.config(state=tk.DISABLED)
        stop_button.config(state=tk.NORMAL)

        threading.Thread(target=run_detection,
                         args=(video_label, fps_label, result_label, start_button, stop_button, camera_icon_tk, video_frame),
                         daemon=True).start()


def stop_detection(start_button, stop_button, camera_icon_tk, video_label):
    global stop_thread, detection_running
    stop_thread = True
    detection_running = False
    start_button.config(state=tk.NORMAL)
    stop_button.config(state=tk.DISABLED)
    video_label.config(image=camera_icon_tk)
    video_label.image = camera_icon_tk  # Keep a reference!


def create_gui():
    global root, background_color, font_color, camera_icon_path, frame_width, frame_height, border_radius

    if root is None:
        root = tk.Tk()
        root.title("YOLO Detection")
        root.configure(bg=background_color)

    style = ttk.Style(root)
    style.configure("TFrame", background=background_color) # Apply to all frames

    main_frame = ttk.Frame(root, padding="20")
    main_frame.pack(expand=True, fill="both")

    title_label = ttk.Label(main_frame, text="YOLO Detection", font=('Arial', 24, 'bold'))
    title_label.pack(pady=(0, 20))

    # Create a Canvas to hold the video
    video_frame = tk.Canvas(main_frame, width=frame_width, height=frame_height, background=background_color, highlightthickness=0)
    video_frame.pack()

    # Create a Label on the Canvas
    video_label = Label(video_frame, background=background_color)
    video_label.place(x=0, y=0, anchor="nw")  # Place the label at the top-left corner

    try:
        camera_icon = Image.open(camera_icon_path)
    except FileNotFoundError:
        messagebox.showerror("Error", f"Camera icon not found at {camera_icon_path}")
        camera_icon = Image.new("RGB", (frame_width, frame_height), background_color)  # Create a blank image instead

    camera_icon = camera_icon.resize((frame_width, frame_height), Image.LANCZOS)

    # Apply rounded corners to initial camera icon
    mask = create_rounded_mask(frame_width, frame_height, border_radius)
    camera_icon_array = np.array(camera_icon)
    masked_camera_icon = apply_mask(camera_icon_array, mask)
    masked_camera_icon = Image.fromarray(masked_camera_icon)
    masked_camera_icon_tk = ImageTk.PhotoImage(masked_camera_icon, master=root)  # create it again!

    video_label.config(image=masked_camera_icon_tk)
    video_label.image = masked_camera_icon_tk  # keep reference!

    fps_label = ttk.Label(main_frame, text="FPS: --", font=('Arial', 12))
    fps_label.pack()

    result_label = ttk.Label(main_frame, text="Status: Idle", font=('Arial', 12))
    result_label.pack()

    button_frame = ttk.Frame(main_frame)
    button_frame.pack(pady=(20, 0))

    start_button = ttk.Button(button_frame, text="Start Detection",
                              command=lambda: start_detection(video_label, fps_label, result_label, start_button,
                                                              stop_button, masked_camera_icon_tk, video_frame))
    start_button.grid(row=0, column=0, padx=10)

    stop_button = ttk.Button(button_frame, text="Stop Detection",
                             command=lambda: stop_detection(start_button, stop_button, masked_camera_icon_tk,
                                                              video_label))
    stop_button.grid(row=0, column=1, padx=10)
    stop_button.config(state=tk.DISABLED)

    # Apply border radius using tk.Canvas.create_round_rectangle
    video_frame.create_round_rectangle = lambda x, y, width, height, radius, **kwargs: video_frame.create_polygon(
        x + radius, y,
        x + width - radius, y,
        x + width, y + radius,
        x + width, y + height - radius,
        x + width - radius, y + height,
        x + radius, y + height,
        x, y + height - radius,
        x, y + radius,
        smooth=True,
        **kwargs
    ) # Lambda is crucial for setting function in order to set the keyword arguments such as width or height
    video_frame.create_round_rectangle(0, 0, frame_width, frame_height, radius=border_radius, fill=background_color, outline="")

    video_frame.lower(video_label)  # Send the rounded rectangle to the bottom
    return video_label, fps_label, result_label, start_button, stop_button, root, masked_camera_icon_tk, video_frame


if __name__ == "__main__":
    video_label, fps_label, result_label, start_button, stop_button, root, camera_icon_tk, video_frame = create_gui()
    root.mainloop()