import customtkinter as ctk
import serial
import serial.tools.list_ports
import cv2
import numpy as np
import threading
import onnxruntime as ort  # <--- No PyTorch needed!
from PIL import Image

# --- CONFIG ---
BAUD_RATE = 115200
ALL_MOTORS = [11, 12, 14, 15, 16, 17]
STEPS_PER_DEG = 4096.0 / 360.0

class RandomPickerONNX(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("ML Random Picker (ONNX Version)")
        self.geometry("1000x700")
        
        self.ser = None
        self.is_running = False
        self.detected_pos = None
        self.place_pose = [90, 90, 90, 90, 90, 0] # Default
        self.ort_session = None

        # Load ONNX Model
        try:
            self.ort_session = ort.InferenceSession("ik_model.onnx")
            print("ONNX Model Loaded Successfully!")
        except Exception as e:
            print(f"Error loading ONNX model: {e}")

        self.setup_ui()
        self.refresh_ports()
        self.start_camera()

    def setup_ui(self):
        # Video Feed
        self.lbl_video = ctk.CTkLabel(self, text="Loading...", width=640, height=480)
        self.lbl_video.pack(side="left", padx=10, pady=10)
        
        # Controls
        f_right = ctk.CTkFrame(self)
        f_right.pack(side="right", fill="both", expand=True)
        
        self.cbox = ctk.CTkComboBox(f_right, values=["Scanning..."]); self.cbox.pack(pady=5)
        ctk.CTkButton(f_right, text="CONNECT", command=self.connect, fg_color="green").pack(pady=5)
        
        # Save Place Button (For demonstration, assumes you moved robot there)
        ctk.CTkButton(f_right, text="Set PLACE Pose", command=self.save_place, fg_color="blue").pack(pady=20)
        
        # Auto Start
        self.btn_auto = ctk.CTkButton(f_right, text="START PICKING", command=self.toggle_auto, fg_color="gray")
        self.btn_auto.pack(pady=40)
        
        # Sliders (Hidden but useful for manual logic if needed)
        self.sliders = {} 
        for mid in ALL_MOTORS:
            s = ctk.CTkSlider(f_right, from_=0, to=180); s.set(90)
            self.sliders[mid] = s

    # --- INFERENCE ENGINE ---
    def predict_angles(self, x, y):
        if self.ort_session is None: return [90]*6
        
        # 1. Prepare Input (Normalize 0-1, Shape [1, 2], Float32)
        input_data = np.array([[x / 640.0, y / 640.0]], dtype=np.float32)
        
        # 2. Run ONNX Inference
        outputs = self.ort_session.run(None, {'input': input_data})
        
        # 3. Process Output (Denormalize -> 0-180)
        norm_angles = outputs[0][0] # Get first batch result
        angles = (norm_angles * 180.0).astype(int)
        
        # 4. Append Gripper (0 = Open)
        return list(angles) + [0]

    # --- AUTO LOGIC ---
    def auto_loop(self):
        while self.is_running:
            if self.detected_pos:
                x, y = self.detected_pos
                print(f"Target: {x}, {y}")
                
                # 1. Predict
                pose = self.predict_angles(x, y)
                
                # 2. Pick Sequence
                # A. Approach (Gripper Open=0)
                pose[5] = 0
                self.move_robot(pose, 2.0)
                
                # B. Grip (Gripper Close=75)
                pose[5] = 75
                self.move_robot(pose, 1.0)
                
                # C. Lift (Shoulder Up)
                lift = pose.copy()
                lift[1] -= 20 
                self.move_robot(lift, 1.0)
                
                # D. Place
                place = self.place_pose.copy()
                place[5] = 75
                self.move_robot(place, 3.0)
                
                # E. Drop
                place[5] = 0
                self.move_robot(place, 1.0)
                
                # F. Home
                self.move_robot([90,90,90,90,90,0], 2.0)
                
                self.detected_pos = None
                time.sleep(1.0)
            time.sleep(0.1)

    # --- VISION ---
    def start_camera(self):
        self.cap = cv2.VideoCapture(0) # Standard Camera
        threading.Thread(target=self.cam_loop, daemon=True).start()

    def cam_loop(self):
        while True:
            ret, frame = self.cap.read()
            if not ret: continue
            
            # Simple Red Detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
            mask2 = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
            mask = mask1 | mask2
            
            # Find Blob
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                if cv2.contourArea(c) > 500:
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)
                        if self.is_running and self.detected_pos is None:
                            self.detected_pos = (cx, cy)

            img = ctk.CTkImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), size=(640, 480))
            try: self.lbl_video.configure(image=img, text="")
            except: pass

    # --- SERIAL HELPERS ---
    def connect(self):
        try:
            self.ser = serial.Serial(self.cbox.get(), BAUD_RATE)
            self.ser.dtr=False; self.ser.rts=False
        except: pass

    def refresh_ports(self):
        p = [x.device for x in serial.tools.list_ports.comports()]
        self.cbox.configure(values=p)

    def move_robot(self, angles, delay):
        if not self.ser: return
        for i, mid in enumerate(ALL_MOTORS):
            steps = int(angles[i] * STEPS_PER_DEG)
            msg = self.make_packet(mid, steps)
            self.ser.write(msg)
            if mid == 12: self.ser.write(self.make_packet(13, 4096-steps)) # Double shoulder
        time.sleep(delay)

    def make_packet(self, mid, steps):
        steps = max(0, min(4095, steps))
        cksm = (~(mid + 54 + (steps&0xFF) + (steps>>8))) & 0xFF
        return bytearray([0xFF, 0xFF, mid, 9, 3, 42, steps&0xFF, steps>>8, 0, 0, 88, 2, cksm])

    def save_place(self):
        # Reads current sliders to set Place Pose
        self.place_pose = [int(self.sliders[m].get()) for m in ALL_MOTORS]
        print(f"Place Pose Saved: {self.place_pose}")

    def toggle_auto(self):
        self.is_running = not self.is_running
        self.btn_auto.configure(text="STOP" if self.is_running else "START PICKING", 
                                fg_color="red" if self.is_running else "gray")
        if self.is_running:
            threading.Thread(target=self.auto_loop, daemon=True).start()

if __name__ == "__main__":
    RandomPickerONNX().mainloop()