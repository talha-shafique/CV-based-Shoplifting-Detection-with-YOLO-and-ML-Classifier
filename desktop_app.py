import customtkinter as ctk
from tkinter import filedialog
import threading
import cv2
import joblib
import pandas as pd
from ultralytics import YOLO
from PIL import Image
import numpy as np

# --- App Configuration ---
APP_TITLE = "Shoplifting Detection System"
WINDOW_SIZE = "1200x800"

# --- Model & Asset Paths ---
RF_MODEL_PATH = r'D:\Random Projects\Fruit Images for Object Detection\Shoplifting Detection\shoplifting_model.joblib'
CUSTOM_YOLO_PATH = r'D:\Random Projects\Fruit Images for Object Detection\Shoplifting Detection\runs\detect\yolo_shoplifting_custom\weights\best.pt'

# --- Main Application Class ---
class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Window Setup ---
        self.title(APP_TITLE)
        self.geometry(WINDOW_SIZE)
        ctk.set_appearance_mode("dark")

        # --- State Variables ---
        self.video_path = ""
        self.analysis_thread = None
        self.cancel_requested = threading.Event()

        # --- Load Models ---
        self.rf_model = joblib.load(RF_MODEL_PATH)
        self.yolo_model = YOLO(CUSTOM_YOLO_PATH)
        self.class_names = self.yolo_model.names

        # --- Widget Setup ---
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Top Frame for Controls
        self.top_frame = ctk.CTkFrame(self, height=100)
        self.top_frame.grid(row=0, column=0, padx=20, pady=20, sticky="ew")
        self.top_frame.grid_columnconfigure(2, weight=1) # Give weight to the label column

        self.select_video_button = ctk.CTkButton(self.top_frame, text="Select Video", command=self.select_video_command)
        self.select_video_button.grid(row=0, column=0, padx=20, pady=20)

        self.cancel_button = ctk.CTkButton(self.top_frame, text="Cancel Analysis", command=self.cancel_operation, fg_color="#db524b", hover_color="#b0423d")

        self.verdict_label = ctk.CTkLabel(self.top_frame, text="Status: Select a video to begin analysis", font=ctk.CTkFont(size=20, weight="bold"))
        self.verdict_label.grid(row=0, column=2, padx=20, pady=20, sticky="w")

        # Main Frame for Video Display
        self.video_frame = ctk.CTkFrame(self)
        self.video_frame.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")
        self.video_label = ctk.CTkLabel(self.video_frame, text="")
        self.video_label.pack(expand=True, fill="both", padx=10, pady=10)

        # Progress Bar
        self.progress_bar = ctk.CTkProgressBar(self, mode='indeterminate')

    def select_video_command(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
        if self.video_path:
            if self.analysis_thread and self.analysis_thread.is_alive(): return
            self.cancel_requested.clear()
            self.analysis_thread = threading.Thread(target=self.run_analysis, daemon=True)
            self.analysis_thread.start()

    def cancel_operation(self):
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.cancel_requested.set()

    def run_analysis(self):
        self.select_video_button.configure(state="disabled", text="Analyzing...")
        self.verdict_label.configure(text="Status: Analysis in progress...", text_color="#d3d3d3")
        self.cancel_button.grid(row=0, column=1, padx=20, pady=20)
        self.progress_bar.grid(row=2, column=0, padx=20, pady=(0, 20), sticky="ew")
        self.progress_bar.start()

        cap = cv2.VideoCapture(self.video_path)
        frame_predictions = []

        while cap.isOpened():
            if self.cancel_requested.is_set(): break
            success, frame = cap.read()
            if not success: break

            yolo_results = self.yolo_model(frame, verbose=False)[0]
            features = self._calculate_features_from_yolo(yolo_results)
            features_df = pd.DataFrame([features])
            prediction = self.rf_model.predict(features_df)[0]
            frame_predictions.append(prediction)
            annotated_frame = self._draw_boxes(frame, yolo_results, prediction)
            self._update_video_frame(annotated_frame)

        cap.release()

        if self.cancel_requested.is_set():
            self.verdict_label.configure(text="Status: Analysis Cancelled", text_color="#d3d3d3")
            self.video_label.configure(image=None, text="")
        else:
            shoplifting_frame_count = sum(frame_predictions)
            total_frames = len(frame_predictions)
            final_verdict = "Normal"
            verdict_color = "#00FF00"
            if total_frames > 0 and (shoplifting_frame_count / total_frames) > 0.05:
                final_verdict = "Shoplifting LIKELY Detected"
                verdict_color = "#FF0000"
            self.verdict_label.configure(text=f"Verdict: {final_verdict}", text_color=verdict_color)
        
        self.progress_bar.stop()
        self.progress_bar.grid_forget()
        self.cancel_button.grid_forget()
        self.select_video_button.configure(state="normal", text="Select Another Video")

    def _update_video_frame(self, frame):
        cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv2_image)
        label_w, label_h = self.video_label.winfo_width(), self.video_label.winfo_height()
        if label_w < 2 or label_h < 2: label_w, label_h = 1200, 700
        pil_image.thumbnail((label_w, label_h), Image.Resampling.LANCZOS)
        ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=pil_image.size)
        self.video_label.configure(image=ctk_image)

    def _draw_boxes(self, frame, yolo_results, prediction):
        frame_color = (0, 0, 255) if prediction == 1 else (0, 255, 0)
        for box in yolo_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            try:
                label = self.class_names[int(box.cls[0])]
                confidence = box.conf[0]
                cv2.rectangle(frame, (x1, y1), (x2, y2), frame_color, 2)
                cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, frame_color, 2)
            except (IndexError, KeyError): pass
        return frame

    def _calculate_features_from_yolo(self, yolo_results):
        person_boxes, product_boxes, bag_boxes = [], [], []
        try:
            person_id = list(self.class_names.keys())[list(self.class_names.values()).index('person')]
            bag_ids = [k for k, v in self.class_names.items() if v == 'bag']
            product_ids = [k for k, v in self.class_names.items() if v == 'object']
        except ValueError: person_id, bag_ids, product_ids = -1, [], []

        for box in yolo_results.boxes:
            class_id = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0]
            box_data = {'xtl': x1, 'ytl': y1, 'xbr': x2, 'ybr': y2, 'center': ((x1+x2)/2, (y1+y2)/2)}
            if class_id == person_id: person_boxes.append(box_data)
            elif class_id in bag_ids: bag_boxes.append(box_data)
            elif class_id in product_ids: product_boxes.append(box_data)

        min_dist = float('inf')
        if person_boxes and product_boxes:
            for p_box in person_boxes:
                for pr_box in product_boxes:
                    dist = np.linalg.norm(np.array(p_box['center']) - np.array(pr_box['center']))
                    if dist < min_dist: min_dist = dist

        # --- THIS IS THE CRITICAL FIX ---
        product_in_bag_occluded = 0
        if product_boxes and bag_boxes:
            for pr_box in product_boxes:
                for b_box in bag_boxes:
                    # Check if a product's center is inside a bag's bounding box
                    is_inside = b_box['xtl'] < pr_box['center'][0] < b_box['xbr'] and \
                                b_box['ytl'] < pr_box['center'][1] < b_box['ybr']
                    if is_inside:
                        product_in_bag_occluded = 1
                        break
                if product_in_bag_occluded: break
        # --- END OF FIX ---

        return {
            'num_people': len(person_boxes),
            'num_products': len(product_boxes),
            'num_bags': len(bag_boxes),
            'min_person_product_dist': min_dist if min_dist != float('inf') else -1,
            'product_in_bag_occluded': product_in_bag_occluded
        }

if __name__ == "__main__":
    app = App()
    app.mainloop()
