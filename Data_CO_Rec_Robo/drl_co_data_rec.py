import customtkinter as ctk
import serial
import serial.tools.list_ports
import time
import csv
import os
import threading
import cv2
import numpy as np
from PIL import Image
import datetime

# --- CONFIGURATION ---
BAUD_RATE = 115200       
DATA_ROOT = "drl_dataset"
CNN_INPUT_SIZE = (224, 224) 
DEFAULT_SPEED = 600
DEFAULT_CAM_ID = 1   # Change to your camera ID
HOME_FILE = "robot_home.csv"

# Robot IDs
ID_BASE = 11
ID_SHOULDER_L = 12
ID_ELBOW = 14
ID_PITCH = 15
ID_ROLL = 16
ID_GRIPPER = 17
ALL_MOTORS = [ID_BASE, ID_SHOULDER_L, ID_ELBOW, ID_PITCH, ID_ROLL, ID_GRIPPER]
STEPS_PER_DEG = 4096.0 / 360.0

class DRLVisionCollector(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("DRL Vision Collector V9 (Rectangles + Live Sync)")
        self.geometry("1250x850")
        ctk.set_appearance_mode("Dark")
        
        # --- State ---
        self.ser = None
        self.global_speed = DEFAULT_SPEED
        self.is_recording = False
        self.live_sync_active = False # For Live Feedback
        self.first_record_flag = True
        self.lock = threading.Lock()
        
        # Camera & Vision
        self.cap = None
        self.current_frame = None       
        self.processed_frame = None     
        self.cam_width = 640
        self.cam_height = 480
        
        # Color Tracking
        self.target_colors = [] 
        self.object_coords = [[-1,-1], [-1,-1], [-1,-1]] 
        
        # Robot State
        self.current_joints = {mid: 90 for mid in ALL_MOTORS}
        self.prev_joints = {mid: 90 for mid in ALL_MOTORS} 
        self.lbl_joint_vals = {} 
        self.sliders = {}

        # Data Setup
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.img_dir = os.path.join(DATA_ROOT, self.session_id, "images")
        self.csv_path = os.path.join(DATA_ROOT, self.session_id, "data.csv")
        os.makedirs(self.img_dir, exist_ok=True)
        self.init_csv()

        self.create_ui()
        self.refresh_ports()
        
        # START
        self.start_camera_thread()
        self.update_gui_loop()

    def init_csv(self):
        headers = ["timestamp", "img_filename", "episode_start"]
        for mid in ALL_MOTORS: headers.append(f"Raw_J{mid}")
        for mid in ALL_MOTORS: headers.append(f"Norm_J{mid}")
        for mid in ALL_MOTORS: headers.append(f"Raw_D{mid}")
        for mid in ALL_MOTORS: headers.append(f"Norm_D{mid}")
        for i in range(1, 4):
            headers.append(f"Obj{i}_X_Raw"); headers.append(f"Obj{i}_Y_Raw")
            headers.append(f"Obj{i}_X_Norm"); headers.append(f"Obj{i}_Y_Norm")
        
        with open(self.csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(headers)

    def create_ui(self):
        # === LEFT: VISION ===
        self.f_view = ctk.CTkFrame(self, width=660) 
        self.f_view.pack(side="left", fill="y", padx=10, pady=10)
        
        ctk.CTkLabel(self.f_view, text="VISION FEED (Click to Select Color)", font=("Arial", 14, "bold")).pack(pady=5)
        self.lbl_cam = ctk.CTkLabel(self.f_view, text="Loading...", width=640, height=480, fg_color="#111")
        self.lbl_cam.pack(pady=5)
        self.lbl_cam.bind("<Button-1>", self.on_cam_click) 
        self.lbl_cam.bind("<Button-3>", self.reset_colors) 
        self.lbl_vision_status = ctk.CTkLabel(self.f_view, text="Tracking: 0 Objects", text_color="cyan")
        self.lbl_vision_status.pack(pady=5)

        # === RIGHT: CONTROLS ===
        self.f_ctrl = ctk.CTkFrame(self) 
        self.f_ctrl.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        # 1. Connection
        f_top = ctk.CTkFrame(self.f_ctrl)
        f_top.pack(fill="x", pady=5)
        self.cbox_ports = ctk.CTkComboBox(f_top, width=100); self.cbox_ports.pack(side="left", padx=5)
        self.btn_connect = ctk.CTkButton(f_top, width=80, text="CONN", command=self.run_connect); self.btn_connect.pack(side="left", padx=5)
        
        # Live Feedback Switch
        self.var_sync = ctk.StringVar(value="off")
        self.sw_sync = ctk.CTkSwitch(f_top, text="Live Feedback", variable=self.var_sync, onvalue="on", offvalue="off", command=self.toggle_live_sync)
        self.sw_sync.pack(side="right", padx=10)

        # 2. Sliders
        f_joints = ctk.CTkScrollableFrame(self.f_ctrl, label_text="Joint Control", height=350)
        f_joints.pack(fill="both", expand=True, pady=10, padx=5)
        
        joints = [
            ("Base", ID_BASE), ("Shldr", ID_SHOULDER_L), 
            ("Elbow", ID_ELBOW), ("Pitch", ID_PITCH), 
            ("Roll", ID_ROLL), ("Grip", ID_GRIPPER)
        ]
        
        for name, mid in joints:
            row = ctk.CTkFrame(f_joints, fg_color="transparent")
            row.pack(fill="x", pady=5)
            ctk.CTkLabel(row, text=name, width=60, anchor="w", font=("Arial", 12, "bold")).pack(side="left")
            s = ctk.CTkSlider(row, from_=0, to=180, height=20, command=lambda v, m=mid: self.update_motor_cmd(m, v))
            s.set(90)
            s.pack(side="left", fill="x", expand=True, padx=5)
            self.sliders[mid] = s
            lbl = ctk.CTkLabel(row, text="90", width=35, font=("Consolas", 14, "bold"), text_color="#00FFFF")
            lbl.pack(side="right")
            self.lbl_joint_vals[mid] = lbl

        # 3. Recorder
        f_rec = ctk.CTkFrame(self.f_ctrl, fg_color="#222")
        f_rec.pack(fill="x", pady=10, padx=5)
        
        r1 = ctk.CTkFrame(f_rec, fg_color="transparent")
        r1.pack(fill="x", pady=5)
        ctk.CTkButton(r1, text="📸 SNAPSHOT", command=self.record_snapshot).pack(side="left", padx=10, expand=True, fill="x")
        ctk.CTkButton(r1, text="🏁 END EPISODE", command=self.end_episode, fg_color="orange").pack(side="right", padx=10)
        
        r2 = ctk.CTkFrame(f_rec, fg_color="transparent")
        r2.pack(fill="x", pady=5)
        ctk.CTkLabel(r2, text="Rec Rate (Sample/Min):").pack(side="left", padx=5)
        self.sl_freq = ctk.CTkSlider(r2, from_=10, to=30, number_of_steps=20)
        self.sl_freq.set(10)
        self.sl_freq.pack(side="left", fill="x", expand=True, padx=5)
        self.btn_auto = ctk.CTkButton(r2, text="⏺ AUTO", width=80, fg_color="green", command=self.toggle_auto)
        self.btn_auto.pack(side="right", padx=10)
        
        self.lbl_status = ctk.CTkLabel(f_rec, text="Ready", text_color="gray")
        self.lbl_status.pack(pady=5)
        
        # 4. Extra Buttons
        f_ex = ctk.CTkFrame(self.f_ctrl, fg_color="transparent")
        f_ex.pack(fill="x", pady=5)
        ctk.CTkButton(f_ex, text="SET HOME", width=80, fg_color="#555", command=self.set_home).pack(side="left", padx=5, expand=True)
        ctk.CTkButton(f_ex, text="GO HOME", width=80, fg_color="#D97C23", command=self.go_home).pack(side="right", padx=5, expand=True)

    # --- VISION LOGIC (UPDATED: RECTANGLES) ---
    def start_camera_thread(self):
        self.cap = cv2.VideoCapture(DEFAULT_CAM_ID, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)
        threading.Thread(target=self.camera_loop, daemon=True).start()

    def camera_loop(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.current_frame = frame.copy()
                
                display = frame.copy()
                hsv = cv2.cvtColor(display, cv2.COLOR_BGR2HSV)
                found_coords = [[-1,-1], [-1,-1], [-1,-1]]
                
                for i, (lower, upper) in enumerate(self.target_colors):
                    if i >= 3: break
                    mask = cv2.inRange(hsv, lower, upper)
                    mask = cv2.erode(mask, None, iterations=2)
                    mask = cv2.dilate(mask, None, iterations=2)
                    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if len(cnts) > 0:
                        c = max(cnts, key=cv2.contourArea)
                        # --- MODIFICATION: BOUNDING RECTANGLE ---
                        x, y, w, h = cv2.boundingRect(c)
                        
                        # Calculate Centroid for Data
                        cx = x + w // 2
                        cy = y + h // 2
                        found_coords[i] = [cx, cy]
                        
                        # Draw Rectangle
                        color_bgr = [(0,255,0), (0,0,255), (255,0,0)][i]
                        cv2.rectangle(display, (x, y), (x+w, y+h), color_bgr, 2)
                        cv2.putText(display, f"Obj{i+1}", (x, y-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)

                self.object_coords = found_coords
                rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                self.processed_frame = Image.fromarray(rgb)
            time.sleep(0.02)

    def update_gui_loop(self):
        if self.processed_frame is not None:
            ctk_img = ctk.CTkImage(light_image=self.processed_frame, dark_image=self.processed_frame, size=(640, 480))
            self.lbl_cam.configure(image=ctk_img, text="")
            count = sum(1 for c in self.object_coords if c[0] != -1)
            self.lbl_vision_status.configure(text=f"Tracking: {count} Objects")
        self.after(30, self.update_gui_loop)

    def on_cam_click(self, event):
        if len(self.target_colors) >= 3: return
        if self.current_frame is None: return
        x, y = event.x, event.y
        if x < 640 and y < 480:
            pixel = self.current_frame[y, x]
            pixel_hsv = cv2.cvtColor(np.uint8([[pixel]]), cv2.COLOR_BGR2HSV)[0][0]
            hue = pixel_hsv[0]
            lower = np.array([max(0, hue-15), 50, 50])
            upper = np.array([min(180, hue+15), 255, 255])
            self.target_colors.append((lower, upper))
            print(f"Added Color {len(self.target_colors)}")

    def reset_colors(self, event):
        self.target_colors = []

    # --- ROBOT CONTROL & SYNC ---
    def update_motor_cmd(self, mid, value):
        """User moves slider -> Send Command"""
        val = int(value)
        # Update UI instantly for responsiveness
        self.lbl_joint_vals[mid].configure(text=str(val))
        self.current_joints[mid] = val
        
        # Send Packet
        if self.ser and self.ser.is_open:
            steps = int(val * STEPS_PER_DEG)
            if mid == ID_SHOULDER_L:
                self.send_packet(ID_SHOULDER_L, steps)
                self.send_packet(13, 4096 - steps)
            else:
                self.send_packet(mid, steps)

    def toggle_live_sync(self):
        if self.var_sync.get() == "on":
            self.live_sync_active = True
            threading.Thread(target=self.live_sync_thread, daemon=True).start()
        else:
            self.live_sync_active = False

    def live_sync_thread(self):
        """Reads motor positions constantly and updates GUI"""
        while self.live_sync_active and self.ser and self.ser.is_open:
            for mid in ALL_MOTORS:
                raw = self.read_motor_pos(mid)
                if raw is not None:
                    deg = int(raw / STEPS_PER_DEG)
                    self.current_joints[mid] = deg
                    # Update GUI safely
                    self.sliders[mid].set(deg)
                    self.lbl_joint_vals[mid].configure(text=str(deg))
                time.sleep(0.01)
            time.sleep(0.1) # Sync rate ~10Hz

    def read_motor_pos(self, mid):
        """Modified to filter garbage data"""
        cksm = (~(mid + 64)) & 0xFF
        try:
            self.ser.reset_input_buffer()
            self.ser.write(bytearray([0xFF, 0xFF, mid, 4, 2, 56, 2, cksm]))
            t = time.time()
            while time.time() - t < 0.1:
                if self.ser.in_waiting >= 8:
                    if self.ser.read(1) == b'\xff':
                        pkt = self.ser.read(7)
                        if pkt[1] == mid: 
                            raw_val = pkt[4] + (pkt[5] << 8)
                            # --- GARBAGE FILTER ---
                            if raw_val > 4095: return None
                            return raw_val
        except: pass
        return None

    def send_packet(self, mid, steps):
        steps = max(0, min(4095, steps))
        sl, sh = self.global_speed & 0xFF, (self.global_speed >> 8) & 0xFF
        pl, ph = steps & 0xFF, (steps >> 8) & 0xFF
        cksm = (~(mid + 9 + 3 + 42 + pl + ph + 0 + 0 + sl + sh)) & 0xFF
        msg = bytearray([0xFF, 0xFF, mid, 9, 3, 42, pl, ph, 0, 0, sl, sh, cksm])
        try: self.ser.write(msg)
        except: pass

    # --- HOME LOGIC ---
    def set_home(self):
        data = [int(self.sliders[mid].get()) for mid in ALL_MOTORS]
        with open(HOME_FILE, 'w', newline='') as f:
            csv.writer(f).writerow(data)
        self.lbl_status.configure(text="Home Set", text_color="yellow")

    def go_home(self):
        if not os.path.exists(HOME_FILE): return
        with open(HOME_FILE, 'r') as f:
            row = next(csv.reader(f))
            angles = [int(x) for x in row]
            for i, mid in enumerate(ALL_MOTORS):
                self.update_motor_cmd(mid, angles[i])
                self.sliders[mid].set(angles[i])
        self.lbl_status.configure(text="Moved to Home", text_color="green")

    # --- RECORDING & EPISODE ---
    def end_episode(self):
        self.first_record_flag = True
        self.lbl_status.configure(text="Next frame = Episode Start", text_color="orange")

    def record_snapshot(self):
        with self.lock:
            if self.current_frame is None: return
            img = self.current_frame.copy()
            coords = self.object_coords.copy()
        
        joints_now = [self.current_joints[m] for m in ALL_MOTORS]
        episode_start = 0
        if self.first_record_flag:
            deltas = [0] * len(ALL_MOTORS)
            self.first_record_flag = False
            self.prev_joints = self.current_joints.copy()
            episode_start = 1
        else:
            joints_prev = [self.prev_joints[m] for m in ALL_MOTORS]
            deltas = [c - p for c, p in zip(joints_now, joints_prev)]
            self.prev_joints = self.current_joints.copy()

        norm_joints = [round(j / 180.0, 4) for j in joints_now]
        norm_deltas = [round((d + 180.0) / 360.0, 4) for d in deltas]

        vision_data = []
        for (x, y) in coords:
            vision_data.extend([x, y])
            if x != -1:
                vision_data.extend([round(x / self.cam_width, 4), round(y / self.cam_height, 4)])
            else:
                vision_data.extend([-1.0, -1.0])

        ts = int(time.time() * 1000)
        img_name = f"frame_{ts}.jpg"
        resized = cv2.resize(img, CNN_INPUT_SIZE)
        cv2.imwrite(os.path.join(self.img_dir, img_name), resized)
        
        row = [ts, img_name, episode_start] + joints_now + norm_joints + deltas + norm_deltas + vision_data
        with open(self.csv_path, 'a', newline='') as f:
            csv.writer(f).writerow(row)
        self.lbl_status.configure(text=f"Saved | Found: {sum(1 for c in coords if c[0]!=-1)}", text_color="yellow")

    def toggle_auto(self):
        if self.is_recording:
            self.is_recording = False
            self.btn_auto.configure(text="⏺ AUTO", fg_color="green")
        else:
            self.is_recording = True
            self.btn_auto.configure(text="⏹ STOP", fg_color="red")
            spm = int(self.sl_freq.get())
            threading.Thread(target=self.auto_record_loop, args=(spm,)).start()

    def auto_record_loop(self, spm):
        delay = 60.0 / spm
        while self.is_recording:
            start = time.time()
            self.record_snapshot()
            elapsed = time.time() - start
            time.sleep(max(0, delay - elapsed))

    # --- CONN ---
    def refresh_ports(self):
        ports = serial.tools.list_ports.comports()
        self.cbox_ports.configure(values=[p.device for p in ports] or ["No Ports"])

    def run_connect(self):
        p = self.cbox_ports.get()
        if p == "No Ports": return
        threading.Thread(target=self.connect_thread, args=(p,)).start()

    def connect_thread(self, port):
        self.btn_connect.configure(state="disabled", text="...")
        try:
            self.ser = serial.Serial()
            self.ser.port = port
            self.ser.baudrate = BAUD_RATE
            self.ser.dtr = False; self.ser.rts = False
            self.ser.open()
            time.sleep(2.0)
            self.ser.reset_input_buffer()
            self.btn_connect.configure(text="OK", fg_color="green")
            
            # Initial Sync
            for mid in ALL_MOTORS:
                cksm = (~(mid + 4 + 3 + 40 + 1)) & 0xFF
                self.ser.write(bytearray([0xFF, 0xFF, mid, 4, 3, 40, 1, cksm])) # Torque On
                time.sleep(0.02)
                
                raw = self.read_motor_pos(mid)
                if raw is not None:
                    deg = int(raw / STEPS_PER_DEG)
                    self.current_joints[mid] = deg
                    self.sliders[mid].set(deg)
                    self.lbl_joint_vals[mid].configure(text=str(deg))
                time.sleep(0.02)
        except Exception as e:
            self.btn_connect.configure(state="normal", text="CONN")
            self.lbl_status.configure(text=f"Err: {e}", text_color="red")

if __name__ == "__main__":
    app = DRLVisionCollector()
    app.mainloop()
