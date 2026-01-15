import customtkinter as ctk
import serial
import serial.tools.list_ports
import cv2
import numpy as np
import threading
import onnxruntime as ort
from PIL import Image
import time

# --- CONFIG ---
BAUD_RATE = 115200
ALL_MOTORS = [11, 12, 14, 15, 16, 17]
STEPS_PER_DEG = 4096.0 / 360.0

class SmartRandomSorter(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Smart Random Sorter (ML Pick + Fixed Place)")
        self.geometry("1300x850")
        ctk.set_appearance_mode("Dark")

        # System State
        self.ser = None
        self.is_running = False
        self.sliders = {}
        
        # ML Model
        self.ort_session = None
        try:
            self.ort_session = ort.InferenceSession("ik_model_v2.onnx")
            print("ML Model Loaded!")
        except: print("⚠️ WARNING: ik_model_v2.onnx not found!")

        # Vision State
        self.current_frame = None
        self.picking_mode = None # For selecting color to teach
        # Stores: {1: (lower, upper), 2: ...}
        self.color_defs = {1: None, 2: None, 3: None}
        # Stores detected objects: {1: (x,y,ang), 2: ...}
        self.detected_objects = {1: None, 2: None, 3: None}

        # Waypoints for BINS (Fixed Place Locations)
        self.bin_poses = {1: None, 2: None, 3: None}

        self.setup_ui()
        self.refresh_ports()
        self.start_camera()

    def setup_ui(self):
        # === LEFT: INTERACTIVE VISION ===
        f_left = ctk.CTkFrame(self, width=650)
        f_left.pack(side="left", fill="y", padx=10, pady=10)
        
        ctk.CTkLabel(f_left, text="Interactive Vision", font=("Arial", 18, "bold")).pack(pady=10)
        self.lbl_video = ctk.CTkLabel(f_left, text="Loading...", width=640, height=480, fg_color="black")
        self.lbl_video.pack(pady=5)
        self.lbl_video.bind("<Button-1>", self.on_video_click)

        # Color Teach Buttons
        f_colors = ctk.CTkFrame(f_left)
        f_colors.pack(fill="x", pady=10)
        self.btn_c1 = ctk.CTkButton(f_colors, text="Teach Color 1", command=lambda: self.set_pick_mode(1), fg_color="gray")
        self.btn_c1.pack(side="left", padx=5, expand=True)
        self.btn_c2 = ctk.CTkButton(f_colors, text="Teach Color 2", command=lambda: self.set_pick_mode(2), fg_color="gray")
        self.btn_c2.pack(side="left", padx=5, expand=True)
        self.btn_c3 = ctk.CTkButton(f_colors, text="Teach Color 3", command=lambda: self.set_pick_mode(3), fg_color="gray")
        self.btn_c3.pack(side="left", padx=5, expand=True)

        # === RIGHT: CONTROL ===
        f_right = ctk.CTkFrame(self)
        f_right.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # Connection
        self.cbox = ctk.CTkComboBox(f_right, values=["Scanning..."]); self.cbox.pack(pady=5)
        ctk.CTkButton(f_right, text="CONNECT", command=self.connect, fg_color="green").pack(pady=5)

        # Sliders (for teaching Bins)
        f_sliders = ctk.CTkFrame(f_right); f_sliders.pack(fill="x", pady=10)
        for mid in ALL_MOTORS:
            r = ctk.CTkFrame(f_sliders, fg_color="transparent"); r.pack(fill="x")
            ctk.CTkLabel(r, text=f"J{mid}", width=30).pack(side="left")
            s = ctk.CTkSlider(r, from_=0, to=180, command=lambda v, m=mid: self.move_motor(m, v))
            s.set(90); s.pack(side="left", fill="x")
            self.sliders[mid] = s

        # Bin Teaching
        ctk.CTkLabel(f_right, text="Output Bins (Fixed Place)", font=("Arial", 14)).pack(pady=10)
        for i in range(1, 4):
            ctk.CTkButton(f_right, text=f"Save Current Pose as BIN {i}", 
                          command=lambda x=i: self.save_bin(x), fg_color="#444").pack(fill="x", pady=2)

        # Auto Start
        self.btn_auto = ctk.CTkButton(f_right, text="START ML SORTING", command=self.toggle_auto, 
                                      height=50, fg_color="gray", state="disabled")
        self.btn_auto.pack(side="bottom", fill="x", pady=20, padx=20)

    # --- VISION LOGIC ---
    def set_pick_mode(self, idx):
        self.picking_mode = idx
        # Visual feedback
        for b in [self.btn_c1, self.btn_c2, self.btn_c3]: b.configure(border_width=0)
        [self.btn_c1, self.btn_c2, self.btn_c3][idx-1].configure(border_width=2, border_color="yellow")

    def on_video_click(self, event):
        if self.picking_mode is None or self.current_frame is None: return
        
        x, y = event.x, event.y
        if 0 <= x < 640 and 0 <= y < 480:
            hsv = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2HSV)
            pixel = hsv[y, x]
            h, s, v = int(pixel[0]), int(pixel[1]), int(pixel[2])
            
            # Save Color Range
            lower = np.array([max(0, h-10), max(50, s-50), max(50, v-50)], dtype="uint8")
            upper = np.array([min(180, h+10), 255, 255], dtype="uint8")
            self.color_defs[self.picking_mode] = (lower, upper)
            
            # Button Feedback
            btn = [self.btn_c1, self.btn_c2, self.btn_c3][self.picking_mode-1]
            rgb = self.current_frame[y, x]
            hex_c = '#%02x%02x%02x' % (rgb[2], rgb[1], rgb[0])
            btn.configure(fg_color=hex_c, text=f"Color {self.picking_mode} SET")
            self.picking_mode = None # Exit mode

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        threading.Thread(target=self.cam_loop, daemon=True).start()

    def cam_loop(self):
        while True:
            ret, frame = self.cap.read()
            if not ret: continue
            self.current_frame = frame.copy()
            
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Process all 3 potential colors
            for idx, bounds in self.color_defs.items():
                if bounds is None: continue
                
                lower, upper = bounds
                mask = cv2.inRange(hsv, lower, upper)
                mask = cv2.erode(mask, None, iterations=1)
                mask = cv2.dilate(mask, None, iterations=1)
                
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                found = None
                
                if cnts:
                    c = max(cnts, key=cv2.contourArea)
                    if cv2.contourArea(c) > 500:
                        # Get Oriented Rect
                        rect = cv2.minAreaRect(c)
                        (cx, cy), (w, h), angle = rect
                        if w < h: angle = angle + 90
                        
                        # Store target
                        found = (cx, cy, angle)
                        
                        # Draw
                        box = np.int32(cv2.boxPoints(rect))
                        color = (0, 255, 0) # Green
                        cv2.drawContours(frame, [box], 0, color, 2)
                        cv2.putText(frame, f"C{idx}", (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                self.detected_objects[idx] = found

            # Display
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ctk.CTkImage(Image.fromarray(rgb), size=(640, 480))
            try: self.lbl_video.configure(image=img, text="")
            except: pass

    # --- ML & ROBOT LOGIC ---
    def predict_pick_pose(self, x, y, angle):
        if self.ort_session is None: return None
        
        # Normalize inputs (Match Training!)
        norm_x = x / 640.0
        norm_y = y / 640.0
        norm_a = angle / 180.0
        
        inp = np.array([[norm_x, norm_y, norm_a]], dtype=np.float32)
        res = self.ort_session.run(None, {'input': inp})
        
        # Denormalize outputs
        angles = (res[0][0] * 180.0).astype(int)
        return list(angles) + [0] # Add Gripper (Open)

    def toggle_auto(self):
        self.is_running = not self.is_running
        self.btn_auto.configure(text="STOP" if self.is_running else "START ML SORTING", 
                                fg_color="red" if self.is_running else "#D97C23")
        if self.is_running:
            threading.Thread(target=self.sorting_loop, daemon=True).start()

    def sorting_loop(self):
        print("Started Sorting Loop")
        self.move_robot([90,145,12,90,7,0], 2.0) # Home
        
        while self.is_running:
            # Check 1 -> 2 -> 3
            for idx in range(1, 4):
                if not self.is_running: break
                
                target = self.detected_objects[idx]
                bin_pose = self.bin_poses[idx]
                
                # If we see the color AND have a bin for it
                if target and bin_pose:
                    x, y, angle = target
                    print(f"Processing Color {idx} at {x:.0f},{y:.0f}")
                    
                    # 1. Get Pick Pose via ML
                    pick_pose = self.predict_pick_pose(x, y, angle)
                    if pick_pose is None: continue
                    
                    # 2. Execute Pick
                    pick_pose[5] = 0   # Open
                    self.move_robot(pick_pose, 2.0)
                    
                    pick_pose[5] = 75  # Close
                    self.move_robot(pick_pose, 1.0)
                    
                    # 3. Lift
                    lift = pick_pose.copy()
                    lift[1] -= 25      # Lift shoulder
                    self.move_robot(lift, 1.0)
                    
                    # 4. Move to Fixed Bin
                    place = bin_pose.copy()
                    place[5] = 75      # Keep closed
                    self.move_robot(place, 3.0)
                    
                    # 5. Drop
                    place[5] = 0       # Open
                    self.move_robot(place, 1.0)
                    
                    # 6. Home
                    self.move_robot([90,145,12,90,7,0], 2.0)
                    
                    # Wait for scene to settle
                    time.sleep(1.5)
            
            time.sleep(0.1)

    # --- HARDWARE HELPERS ---
    def connect_serial(self):
        # Use the safer logic from rec_replay.py
        try:
            port = self.cbox_ports.get()
            
            self.ser = serial.Serial()
            self.ser.port = port
            self.ser.baudrate = BAUD_RATE
            
            # Prevent Reset
            self.ser.dtr = False 
            self.ser.rts = False
            
            self.ser.open()
            
            # Double check
            self.ser.dtr = False 
            self.ser.rts = False
            
            time.sleep(2)
            self.ser.reset_input_buffer()
            
            self.log("Connected.")
            self.btn_conn.configure(state="disabled", text="LINKED")
            self.btn_auto.configure(state="normal", fg_color="#D97C23")
            
            # OPTIONAL: Lock motors immediately if you have a torque function
            # self.set_torque_logic(True) 
            
        except Exception as e:
            self.log(f"Connect Failed: {e}")

    def refresh_ports(self):
        p = [x.device for x in serial.tools.list_ports.comports()]
        self.cbox.configure(values=p if p else ["No Ports"])

    def save_bin(self, idx):
        pose = [int(self.sliders[m].get()) for m in ALL_MOTORS]
        self.bin_poses[idx] = pose
        print(f"Bin {idx} Saved: {pose}")

    def move_motor(self, mid, val):
        if self.ser and self.ser.is_open:
            self.send_packet(mid, int(val))

    def move_robot(self, angles, delay):
        if not self.ser: return
        for i, mid in enumerate(ALL_MOTORS):
            self.send_packet(mid, angles[i])
        time.sleep(delay)

    def send_packet(self, mid, angle):
        steps = int(angle * STEPS_PER_DEG)
        steps = max(0, min(4095, steps))
        cksm = (~(mid + 54 + (steps&0xFF) + (steps>>8))) & 0xFF
        msg = bytearray([0xFF, 0xFF, mid, 9, 3, 42, steps&0xFF, steps>>8, 0, 0, 88, 2, cksm])
        self.ser.write(msg)
        if mid == 12: # Double Shoulder
            inv = 4096 - steps
            cksm2 = (~(13 + 54 + (inv&0xFF) + (inv>>8))) & 0xFF
            msg2 = bytearray([0xFF, 0xFF, 13, 9, 3, 42, inv&0xFF, inv>>8, 0, 0, 88, 2, cksm2])
            self.ser.write(msg2)

if __name__ == "__main__":
    SmartRandomSorter().mainloop()