from ultralytics import YOLO
import cv2
import cvzone
import time
import threading
import tkinter as tk
from tkinter import ttk, Label, Button, PhotoImage, messagebox
from PIL import Image, ImageTk, ImageSequence
import numpy as np

confidence_threshold = 0.8
stop_thread = False
detection_running = False
initial_wait_time = 1
max_detection_time = 10
root = None
background_color = "skyblue"
frame_width = 640
frame_height = 480
font_color = "#2C3E50"
gif_path = r"C:\Users\lenovo\Downloads\Face ID.gif"  # Replace with YOUR GIF path
iou_threshold = 0.5
multiple_faces_detected = False


def run_detection(video_label, fps_label, result_label, start_button, stop_button, gif_frames):
    global stop_thread, detection_running, initial_wait_time, max_detection_time, root, frame_width, frame_height, iou_threshold, class_names, multiple_faces_detected

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
        model = YOLO("../models/best.pt")  #  your YOLO model
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
    detection_start_time = None
    detection_end_time = start_time + max_detection_time
    detection_result = None
    detection_confidence = None
    gif_frame_index = 0
    last_gif_update = time.time()
    face_recognized_this_frame = False

    try:
        while not stop_thread and detection_running and time.time() < detection_end_time:
            success, frame = cap.read()
            if not success:
                messagebox.showerror("Camera Error", "Failed to read frame.")
                break

            if not wait_period_ended:
                if detection_start_time is None:
                    detection_start_time = time.time()
                    result_label.config(text="Initializing...", foreground=font_color)
                    if root:
                        root.update()

                elapsed_time = time.time() - detection_start_time
                if elapsed_time < initial_wait_time:
                    # GIF handling (non-blocking)
                    current_time = time.time()
                    if current_time - last_gif_update > 0.04:
                        if gif_frames:
                            gif_frame = gif_frames[gif_frame_index % len(gif_frames)]
                            gif_frame_index += 1
                            try:
                                gif_frame_tk = ImageTk.PhotoImage(image=gif_frame)
                                video_label.config(image=gif_frame_tk)
                                video_label.image = gif_frame_tk
                            except tk.TclError as e:
                                print(f"TclError during GIF update: {e}")
                                break
                            except Exception as e:
                                print(f"Unexpected error: {e}")
                                break
                        last_gif_update = current_time

                    result_label.config(text=f"Initializing...({elapsed_time:.1f}/{initial_wait_time:.1f}s)",
                                        foreground=font_color)
                    if root:
                        root.update()

                    continue

                else:
                    wait_period_ended = True
                    result_label.config(text="Detection started....", foreground=font_color)
                    if root:
                        root.update()
                    start_time = time.time()

            if wait_period_ended:
                results = model(frame, stream=True)
                boxes = []
                confidences = []
                class_ids = []
                faces_detected_count = 0
                face_recognized_this_frame = False

                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = box.conf[0].item()
                        cls = int(box.cls[0].item())
                        boxes.append([x1, y1, x2 - x1, y2 - y1])
                        confidences.append(conf)
                        class_ids.append(cls)

                boxes = np.array(boxes, dtype=np.float32)
                confidences = np.array(confidences, dtype=np.float32)
                indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), confidence_threshold, iou_threshold)

                if indices is not None and len(indices) > 0:
                    faces_detected_count = len(indices)
                    if faces_detected_count > 1:
                        multiple_faces_detected = True

                    for i in indices:
                        index = i[0] if isinstance(i, np.ndarray) else i
                        box = boxes[index]
                        x1, y1, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                        conf = confidences[index]
                        cls = class_ids[index]
                        face_recognized_this_frame = True

                        # --- Coloring the rectangle ---
                        color = (0, 255, 0) if class_names[cls] == 'Real' else (0, 0, 255)  # Green for Real, Red for Fake
                        cvzone.cornerRect(frame, (x1, y1, w, h), colorC=color, colorR=color)  # Use cvzone for rectangle
                        cvzone.putTextRect(frame, f'{class_names[cls].upper()} {conf:.1%}', (max(0, x1), max(35, y1)),
                                           scale=1, thickness=2, colorR=color, colorB=color)

                        if faces_detected_count == 1 and not multiple_faces_detected:
                            detection_result = class_names[cls].upper()
                            detection_confidence = f"{conf:.1%}"

                if faces_detected_count == 0:
                    result_label.config(text="No Face Detected......", foreground="red")
                elif faces_detected_count == 1:
                    result_label.config(text="Recognizing the face...please wait ...", foreground="green")
                else:
                    result_label.config(text=f"Multiple Faces Detected.....try again.... ({faces_detected_count})", foreground="red")

            new_frame_time = time.time()
            fps = 1 / (new_frame_time - start_time) if (new_frame_time - start_time) != 0 else 0
            fps_label.config(text=f"FPS: {fps:.2f}", foreground=font_color)

            if frame is not None and wait_period_ended:
                mask = create_rounded_mask(frame_width, frame_height, radius=20)
                masked_frame = apply_mask(frame, mask)

                img = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img_tk = ImageTk.PhotoImage(image=img)
                video_label.config(image=img_tk)
                video_label.image = img_tk

            if root:
                root.update()

    except Exception as e:
        messagebox.showerror("Unexpected Error", f"An error occurred: {e}")
        result_label.config(text=f"Error: {e}", foreground="red")
    finally:
        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        detection_running = False
        start_button.config(state=tk.NORMAL)
        stop_button.config(state=tk.DISABLED)

        if multiple_faces_detected:
            formatted_result = "Multiple Faces Detected"
        elif face_recognized_this_frame:
            formatted_result = f"Face Recognized: {detection_result}\nConfidence: {detection_confidence}"
        else:
            formatted_result = "No face recognized."

        messagebox.showinfo("Detection Complete", formatted_result)
        result_label.config(text="Detection Completed", foreground=font_color)

        if gif_frames_tk:
           video_label.config(image=gif_frames_tk[0])
           video_label.image = gif_frames_tk[0]


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
    return cv2.bitwise_and(frame, frame, mask=mask)


def load_gif(path, resize_width=None, resize_height=None):
    try:
        with Image.open(path) as img:
            frames = []
            for frame in ImageSequence.Iterator(img):
                if resize_width and resize_height:
                    frame = frame.resize((resize_width, resize_height), Image.LANCZOS)
                frames.append(frame.convert('RGBA'))
            return frames
    except FileNotFoundError:
        messagebox.showerror("Error", f"GIF file not found at {path}")
        return []
    except Exception as e:
        messagebox.showerror("Error", f"Error loading GIF: {e}")
        return []


def start_detection(video_label, fps_label, result_label, start_button, stop_button, gif_frames):
    global stop_thread, detection_running, multiple_faces_detected

    if not detection_running:
        stop_thread = False
        detection_running = True
        multiple_faces_detected = False
        start_button.config(state=tk.DISABLED)
        stop_button.config(state=tk.NORMAL)
        detection_thread = threading.Thread(target=run_detection,
                                            args=(video_label, fps_label, result_label, start_button, stop_button,
                                                  gif_frames))
        detection_thread.daemon = True
        detection_thread.start()


def stop_detection(start_button, stop_button, gif_frames, video_label):
    global stop_thread, detection_running
    stop_thread = True
    detection_running = False
    start_button.config(state=tk.NORMAL)
    stop_button.config(state=tk.DISABLED)
    if gif_frames_tk:
        video_label.config(image=gif_frames_tk[0])
        video_label.image = gif_frames_tk[0]


def create_gui():
    global root, video_label, fps_label, start_button, stop_button, result_label, background_color, font_color, gif_path, gif_frames_tk

    if root is None:
        root = tk.Tk()
        root.title("YOLO Detection")
        root.configure(bg=background_color)

    style = ttk.Style(root)
    style.configure("Start.TButton", padding=10, font=('Arial', 14), background="#2ECC71", foreground="black",
                    borderwidth=0, relief="flat")
    style.map("Start.TButton", background=[('active', '#27AE60')], foreground=[('active', 'black')])
    style.configure("Stop.TButton", padding=10, font=('Arial', 14), background="#E74C3C", foreground="black",
                    borderwidth=0, relief="flat")
    style.map("Stop.TButton", background=[('active', '#C0392B')], foreground=[('active', 'black')])
    style.configure("TLabel", padding=5, font=('Arial', 12, "bold"), background=background_color, foreground=font_color)
    style.configure("TFrame", background=background_color)

    main_frame = ttk.Frame(root, padding="20")
    main_frame.pack(expand=True, fill="both")

    title_label = ttk.Label(main_frame, text=" TrueSight ", font=('calibri',30, 'bold'))
    title_label.pack(pady=(0, 20))

    sub_title_label = ttk.Label(main_frame, text=" Unlocking the authenticity", font=('Arial',15, 'italic'))
    sub_title_label.pack(pady=(0, 20))

    video_frame = ttk.Frame(main_frame)
    video_frame.pack(pady=(0, 10))

    video_label = Label(video_frame, background=background_color)
    video_label.pack()

    gif_frames = load_gif(gif_path, frame_width, frame_height)
    gif_frames_tk = [ImageTk.PhotoImage(frame, master=root) for frame in gif_frames] if gif_frames else []
    if gif_frames_tk:
        video_label.config(image=gif_frames_tk[0])
        video_label.image = gif_frames_tk[0]

    status_frame = ttk.Frame(main_frame)
    status_frame.pack(pady=(10, 0))

    fps_label = ttk.Label(status_frame, text="FPS: 0.00")
    fps_label.grid(row=0, column=0, padx=10, sticky="w")

    result_label = ttk.Label(status_frame, text="Ready")
    result_label.grid(row=0, column=1, padx=10, sticky="e")

    button_frame = ttk.Frame(main_frame)
    button_frame.pack(pady=(20, 0))

    start_button = tk.Button(button_frame, text="Start recognition", font=('Arial', 14, 'bold'), bg="#2ECC71", fg="white",
                             activebackground="#27AE60", activeforeground="white", relief="flat", bd=0, padx=20,
                             pady=10,
                             command=lambda: start_detection(video_label, fps_label, result_label, start_button,
                                                             stop_button, gif_frames))
    start_button.grid(row=0, column=0, padx=10)

    stop_button = tk.Button(button_frame, text="Stop recognition", font=('Arial', 14, 'bold'), bg="#E74C3C", fg="white",
                            activebackground="#C0392B", activeforeground="white", relief="flat", bd=0, padx=20, pady=10,
                            command=lambda: stop_detection(start_button, stop_button, gif_frames, video_label))
    stop_button.grid(row=0, column=1, padx=10)
    stop_button.config(state=tk.DISABLED)

    return video_label, fps_label, result_label, start_button, stop_button, root, gif_frames


if __name__ == "__main__":
    video_label, fps_label, result_label, start_button, stop_button, root, gif_frames = create_gui()
    root.mainloop()
    cv2.destroyAllWindows()