import customtkinter as ctk
import serial
import serial.tools.list_ports
import time
import csv
import os
import threading
import cv2
from PIL import Image
import datetime

# --- CONFIGURATION ---
BAUD_RATE = 115200       
DATA_ROOT = "drl_dataset_2"
CNN_INPUT_SIZE = (224, 224) 
DEFAULT_SPEED = 600
DEFAULT_CAM_ID = 1   # Change to your camera ID
HOME_FILE = "robot_home.csv" # File to store home pose

# Robot IDs
ID_BASE = 11
ID_SHOULDER_L = 12
ID_ELBOW = 14
ID_PITCH = 15
ID_ROLL = 16
ID_GRIPPER = 17
ALL_MOTORS = [ID_BASE, ID_SHOULDER_L, ID_ELBOW, ID_PITCH, ID_ROLL, ID_GRIPPER]
STEPS_PER_DEG = 4096.0 / 360.0

class DRLDataCollector(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("DRL Data Collector V10 (Session Manager)")
        self.geometry("1100x800") 
        ctk.set_appearance_mode("Dark")
        
        # State
        self.ser = None
        self.global_speed = DEFAULT_SPEED
        self.is_recording = False
        self.first_record_flag = True 
        self.current_frame = None       
        self.display_frame = None       
        self.lock = threading.Lock()
        
        # Robot State
        self.current_joints = {mid: 90 for mid in ALL_MOTORS}
        self.prev_joints = {mid: 90 for mid in ALL_MOTORS} 
        self.lbl_joint_vals = {} 
        self.sliders = {}

        # Setup First Session Immediately
        self.start_new_session()

        self.create_ui()
        self.refresh_ports()
        
        # START SYSTEMS
        self.start_camera_thread()
        self.update_gui_image()

    def start_new_session(self):
        """Creates a new timestamped folder and initializes CSV"""
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.img_dir = os.path.join(DATA_ROOT, self.session_id, "images")
        self.csv_path = os.path.join(DATA_ROOT, self.session_id, "data.csv")
        
        os.makedirs(self.img_dir, exist_ok=True)
        
        # Create CSV Header
        headers = ["timestamp", "img_filename", "episode_start"] 
        for mid in ALL_MOTORS: headers.append(f"Raw_J{mid}")
        for mid in ALL_MOTORS: headers.append(f"Norm_J{mid}")
        for mid in ALL_MOTORS: headers.append(f"Raw_D{mid}")
        for mid in ALL_MOTORS: headers.append(f"Norm_D{mid}")
        
        with open(self.csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(headers)
            
        print(f"--- NEW SESSION STARTED: {self.session_id} ---")

    def create_ui(self):
        # === LEFT: CAMERA ===
        self.f_view = ctk.CTkFrame(self, width=340) 
        self.f_view.pack(side="left", fill="y", padx=10, pady=10)
        
        ctk.CTkLabel(self.f_view, text=f"CAM {DEFAULT_CAM_ID}", font=("Arial", 16, "bold")).pack(pady=10)
        self.lbl_cam_feed = ctk.CTkLabel(self.f_view, text="Loading...", width=320, height=240, fg_color="#111")
        self.lbl_cam_feed.pack(pady=10)

        # === RIGHT: CONTROLS ===
        self.f_ctrl = ctk.CTkFrame(self) 
        self.f_ctrl.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        # 1. Connection & Speed
        f_top = ctk.CTkFrame(self.f_ctrl)
        f_top.pack(fill="x", pady=5, padx=5)
        
        self.cbox_ports = ctk.CTkComboBox(f_top, width=120, values=["Scanning..."]); self.cbox_ports.pack(side="left", padx=5)
        self.btn_connect = ctk.CTkButton(f_top, width=100, text="CONNECT", command=self.run_connect); self.btn_connect.pack(side="left", padx=5)
        
        # Speed & Home Buttons
        f_btns = ctk.CTkFrame(self.f_ctrl, fg_color="transparent")
        f_btns.pack(fill="x", pady=5)
        
        ctk.CTkLabel(f_btns, text="Speed:").pack(side="left", padx=5)
        self.ent_speed = ctk.CTkEntry(f_btns, width=50); self.ent_speed.insert(0, str(DEFAULT_SPEED)); self.ent_speed.pack(side="left")
        ctk.CTkButton(f_btns, text="Set", width=40, command=self.set_speed).pack(side="left", padx=5)
        
        ctk.CTkButton(f_btns, text="SET HOME", width=80, fg_color="#555", command=self.set_home).pack(side="right", padx=5)
        ctk.CTkButton(f_btns, text="GO HOME", width=80, fg_color="#D97C23", command=self.go_home).pack(side="right", padx=5)

        # 2. Sliders
        f_joints = ctk.CTkScrollableFrame(self.f_ctrl, label_text="Joint Control", height=300)
        f_joints.pack(fill="both", expand=True, pady=10, padx=5)
        
        joints = [
            ("Base (11)", ID_BASE), ("Shldr (12)", ID_SHOULDER_L), 
            ("Elbow (14)", ID_ELBOW), ("Pitch (15)", ID_PITCH), 
            ("Roll (16)", ID_ROLL), ("Grip (17)", ID_GRIPPER)
        ]
        
        for name, mid in joints:
            row = ctk.CTkFrame(f_joints, fg_color="transparent")
            row.pack(fill="x", pady=5)
            ctk.CTkLabel(row, text=name, width=80, anchor="w", font=("Arial", 12, "bold")).pack(side="left")
            s = ctk.CTkSlider(row, from_=0, to=180, height=20, command=lambda v, m=mid: self.update_motor(m, v))
            s.set(90)
            s.pack(side="left", fill="x", expand=True, padx=10)
            self.sliders[mid] = s
            lbl = ctk.CTkLabel(row, text="90", width=40, font=("Consolas", 16, "bold"), text_color="#00FFFF")
            lbl.pack(side="right", padx=10)
            self.lbl_joint_vals[mid] = lbl

        # 3. Recorder
        f_rec = ctk.CTkFrame(self.f_ctrl, fg_color="#2A2A2A")
        f_rec.pack(fill="x", pady=10, padx=5)
        
        # Row 1: Snapshot & End Episode
        r1 = ctk.CTkFrame(f_rec, fg_color="transparent")
        r1.pack(fill="x", pady=5)
        ctk.CTkButton(r1, text="📸 SNAPSHOT", command=self.record_snapshot, width=150).pack(side="left", padx=10)
        
        # END EPISODE BUTTON
        ctk.CTkButton(r1, text="🏁 END EPISODE", command=self.end_episode, width=150, fg_color="#D97C23").pack(side="right", padx=10)
        
        # Row 2: Auto Record (Samples Per Minute)
        r2 = ctk.CTkFrame(f_rec, fg_color="transparent")
        r2.pack(fill="x", pady=5)
        ctk.CTkLabel(r2, text="Rate:").pack(side="left", padx=10)
        
        # Slider for 10 to 30 SPM
        self.sl_freq = ctk.CTkSlider(r2, from_=10, to=30, number_of_steps=20, width=150)
        self.sl_freq.set(10)
        self.sl_freq.pack(side="left", padx=5)
        
        self.lbl_freq = ctk.CTkLabel(r2, text="10 /min", width=60)
        self.lbl_freq.pack(side="left")
        self.sl_freq.configure(command=lambda v: self.lbl_freq.configure(text=f"{int(v)} /min"))
        
        self.btn_auto = ctk.CTkButton(r2, text="⏺ START AUTO", fg_color="green", command=self.toggle_auto)
        self.btn_auto.pack(side="right", padx=10, fill="x", expand=True)

        self.lbl_status = ctk.CTkLabel(f_rec, text="Ready", text_color="gray")
        self.lbl_status.pack(pady=5)

    # --- LOGIC: Episode Management ---
    def end_episode(self):
        """Stops recording, creates NEW session folder"""
        # 1. Stop Auto Recording
        if self.is_recording:
            self.is_recording = False
            self.btn_auto.configure(text="⏺ START AUTO", fg_color="green")
            
        # 2. Reset First Flag (so next frame is start)
        self.first_record_flag = True
        
        # 3. Create NEW Folder Structure
        self.start_new_session()
        
        self.lbl_status.configure(text="New Episode Started (New Folder)", text_color="orange")

    # --- HOME LOGIC ---
    def set_home(self):
        data = [int(self.sliders[mid].get()) for mid in ALL_MOTORS]
        with open(HOME_FILE, 'w', newline='') as f:
            csv.writer(f).writerow(data)
        self.lbl_status.configure(text="Home Position Saved", text_color="yellow")

    def go_home(self):
        if not os.path.exists(HOME_FILE): 
            self.lbl_status.configure(text="No Home File Found", text_color="red")
            return
            
        with open(HOME_FILE, 'r') as f:
            row = next(csv.reader(f))
            angles = [int(x) for x in row]
            
            # Update Motors & Sliders
            for i, mid in enumerate(ALL_MOTORS):
                self.update_motor(mid, angles[i])
                self.sliders[mid].set(angles[i])
        
        self.lbl_status.configure(text="Moved to Home", text_color="green")

    # --- CAMERA LOGIC ---
    def start_camera_thread(self):
        self.cap = cv2.VideoCapture(DEFAULT_CAM_ID, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            self.lbl_cam_feed.configure(text="CAM ERROR")
            return
        threading.Thread(target=self.camera_loop, daemon=True).start()

    def camera_loop(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.current_frame = frame
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.display_frame = Image.fromarray(rgb)
            time.sleep(0.02) 

    def update_gui_image(self):
        with self.lock:
            if self.display_frame is not None:
                ctk_img = ctk.CTkImage(light_image=self.display_frame, dark_image=self.display_frame, size=(320, 240)) 
                self.lbl_cam_feed.configure(image=ctk_img, text="")
        self.after(30, self.update_gui_image)

    # --- MOTOR CONTROL ---
    def set_speed(self):
        try:
            self.global_speed = int(self.ent_speed.get())
            self.lbl_status.configure(text=f"Speed: {self.global_speed}", text_color="yellow")
        except: pass

    def update_motor(self, mid, value):
        val = int(value)
        self.current_joints[mid] = val
        if mid in self.lbl_joint_vals:
            self.lbl_joint_vals[mid].configure(text=str(val))

        if self.ser and self.ser.is_open:
            steps = int(val * STEPS_PER_DEG)
            if mid == ID_SHOULDER_L:
                self.send_packet(ID_SHOULDER_L, steps)
                self.send_packet(13, 4096 - steps)
            else:
                self.send_packet(mid, steps)

    def send_packet(self, mid, steps):
        steps = max(0, min(4095, steps))
        spd = self.global_speed
        sl, sh = spd & 0xFF, (spd >> 8) & 0xFF
        pl, ph = steps & 0xFF, (steps >> 8) & 0xFF
        cksm = (~(mid + 9 + 3 + 42 + pl + ph + 0 + 0 + sl + sh)) & 0xFF
        msg = bytearray([0xFF, 0xFF, mid, 9, 3, 42, pl, ph, 0, 0, sl, sh, cksm])
        try: self.ser.write(msg)
        except: pass

    # --- RECORDING LOGIC ---
    def record_snapshot(self):
        with self.lock:
            if self.current_frame is None: return
            img = self.current_frame.copy()
        
        joints_now = [self.current_joints[m] for m in ALL_MOTORS]
        
        episode_start_val = 0
        if self.first_record_flag:
            deltas = [0] * len(ALL_MOTORS)
            self.first_record_flag = False
            self.prev_joints = self.current_joints.copy()
            episode_start_val = 1 # Mark start of new episode
        else:
            joints_prev = [self.prev_joints[m] for m in ALL_MOTORS]
            deltas = [c - p for c, p in zip(joints_now, joints_prev)]
            self.prev_joints = self.current_joints.copy()
        
        # Normalize
        norm_joints = [round(j / 180.0, 4) for j in joints_now]
        norm_deltas = [round((d + 180.0) / 360.0, 4) for d in deltas]

        ts = int(time.time() * 1000)
        img_name = f"frame_{ts}.jpg"
        resized = cv2.resize(img, CNN_INPUT_SIZE)
        cv2.imwrite(os.path.join(self.img_dir, img_name), resized)
        
        # CSV: [timestamp, img, episode_start, RawJ, NormJ, RawD, NormD]
        row = [ts, img_name, episode_start_val] + joints_now + norm_joints + deltas + norm_deltas
        with open(self.csv_path, 'a', newline='') as f:
            csv.writer(f).writerow(row)
            
        self.lbl_status.configure(text=f"Saved: {img_name}", text_color="yellow")

    def toggle_auto(self):
        if self.is_recording:
            self.is_recording = False
            self.btn_auto.configure(text="⏺ START AUTO", fg_color="green")
        else:
            self.is_recording = True
            self.btn_auto.configure(text="⏹ STOP AUTO", fg_color="red")
            # Get Samples Per Minute
            spm = int(self.sl_freq.get())
            threading.Thread(target=self.auto_record_loop, args=(spm,)).start()

    def auto_record_loop(self, spm):
        # Calculate delay in seconds
        if spm <= 0: spm = 10
        delay = 60.0 / spm 
        
        while self.is_recording:
            start = time.time()
            self.record_snapshot()
            elapsed = time.time() - start
            time.sleep(max(0, delay - elapsed))

    # --- CONNECT & SYNC ---
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
            
            for mid in ALL_MOTORS:
                raw = self.read_motor_pos(mid)
                if raw is not None:
                    deg = int(raw / STEPS_PER_DEG)
                    self.current_joints[mid] = deg
                    self.sliders[mid].set(deg)
                    self.lbl_joint_vals[mid].configure(text=str(deg))
                time.sleep(0.02)
            self.lbl_status.configure(text="Connected & Synced", text_color="green")
        except Exception as e:
            self.btn_connect.configure(state="normal", text="CONN")
            self.lbl_status.configure(text=f"Err: {e}", text_color="red")

    def read_motor_pos(self, mid):
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
                            if raw_val > 4095: return None # Filter Garbage
                            return raw_val
        except: pass
        return None

if __name__ == "__main__":
    app = DRLDataCollector()
    app.mainloop()