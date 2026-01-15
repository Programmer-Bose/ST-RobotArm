import customtkinter as ctk
import serial
import serial.tools.list_ports
import time
import csv
import os
import threading
import math
from tkinter import filedialog 

# --- CONFIGURATION ---
BAUD_RATE = 115200       
DEFAULT_WAYPOINT_FILE = "waypoints_joint_data.csv"
HOME_FILE = "robot_home_pose.csv"
DELTA_STEP = 0.010  # 10mm increment

# Robot Geometry (Meters)
L_BASE = 0.10       
L_SHOULDER = 0.245  
L_ELBOW = 0.145     
L_PITCH = 0.155     

# Motor IDs
ID_BASE = 11
ID_SHOULDER_L = 12
ID_SHOULDER_R = 13
ID_ELBOW = 14
ID_PITCH = 15
ID_ROLL = 16
ID_GRIPPER = 17

# ORDERED LIST FOR OPERATIONS (6 Motors for Control, excluding Shoulder_R which is mirrored)
CONTROL_IDS = [ID_BASE, ID_SHOULDER_L, ID_ELBOW, ID_PITCH, ID_ROLL, ID_GRIPPER]
ALL_MOTORS_SETUP = [ID_BASE, ID_SHOULDER_L, ID_SHOULDER_R, ID_ELBOW, ID_PITCH, ID_ROLL, ID_GRIPPER]

STEPS_PER_DEG = 4096.0 / 360.0

# --- MATH CORE: KINEMATICS ---

def geometric_ik(x, y, z):
    """XYZ -> [Base, Shldr, Elbow, Pitch, Roll] (5 Angles). Returns None if unreachable."""
    # 1. Base Angle
    theta_base_rad = math.atan2(y, x)
    base_deg = math.degrees(theta_base_rad)
    if base_deg < 0: base_deg += 360
    
    # 2. Wrist Position
    r_total = math.sqrt(x**2 + y**2)
    z_total = z - L_BASE
    
    rw = r_total - L_PITCH 
    zw = z_total
    
    # 3. Law of Cosines
    D = math.sqrt(rw**2 + zw**2)
    if D > (L_SHOULDER + L_ELBOW) or D < abs(L_SHOULDER - L_ELBOW):
        return None 

    alpha = math.atan2(zw, rw)
    
    cos_beta = (L_SHOULDER**2 + D**2 - L_ELBOW**2) / (2 * L_SHOULDER * D)
    beta = math.acos(max(-1, min(1, cos_beta)))
    
    cos_gamma = (L_SHOULDER**2 + L_ELBOW**2 - D**2) / (2 * L_SHOULDER * L_ELBOW)
    gamma = math.acos(max(-1, min(1, cos_gamma)))
    
    # 4. Angles
    theta_shoulder = alpha + beta
    theta_elbow = gamma 
    theta_pitch = (math.pi - theta_elbow) - theta_shoulder
    
    val_base = base_deg
    val_shoulder = math.degrees(theta_shoulder)
    val_elbow = math.degrees(theta_elbow)
    val_pitch = 90 + math.degrees(theta_pitch)
    val_roll = 90 
    
    joints = [val_base, val_shoulder, val_elbow, val_pitch, val_roll]
    return [max(0, min(180, int(j))) for j in joints]

def forward_kinematics(angles):
    """
    Angles [Base, Shldr, Elbow, Pitch, ...] -> XYZ Coordinates
    """
    try:
        if len(angles) < 4: return [0,0,0]
        
        b = math.radians(angles[0])
        s = math.radians(angles[1])
        e_internal = math.radians(angles[2]) 
        
        # Pitch: Reconstruct relative pitch angle
        # From IK: val_pitch = 90 + degrees(theta_pitch)
        # So theta_pitch = radians(val_pitch - 90)
        p_relative = math.radians(angles[3] - 90)

        # -- RECONSTRUCT CHAIN (R-Z Plane) --
        # 1. Shoulder Tip
        r_s = L_SHOULDER * math.cos(s)
        z_s = L_SHOULDER * math.sin(s)
        
        # 2. Elbow Tip
        # Global angle of forearm = s - (pi - e_internal)
        ang_forearm = s - (math.pi - e_internal)
        r_e = L_ELBOW * math.cos(ang_forearm)
        z_e = L_ELBOW * math.sin(ang_forearm)
        
        # 3. Pitch Tip (Gripper)
        ang_tool = ang_forearm - p_relative
        r_p = L_PITCH * math.cos(ang_tool)
        z_p = L_PITCH * math.sin(ang_tool)
        
        # Total R and Z
        r_total = r_s + r_e + r_p
        z_total = L_BASE + z_s + z_e + z_p
        
        # Polar to Cartesian
        x = r_total * math.cos(b)
        y = r_total * math.sin(b)
        z = z_total
        return [x, y, z]
    except:
        return [0, 0, 0]

# --- APP ---
class RobotController(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("7-DOF Master Controller V7 (Final Fixed)")
        self.geometry("1300x900")
        ctk.set_appearance_mode("Dark")
        
        self.ser = None
        self.global_speed = 600
        self.replay_file = DEFAULT_WAYPOINT_FILE
        
        # State
        self.current_calc_angles = [90, 90, 90, 90, 90, 90] # B, S, E, P, R, G
        self.pending_angles = None
        self.is_auto_recording = False
        self.is_replaying = False
        
        self.sliders = {} 
        
        self.create_ui()
        self.refresh_ports()
        self.toggle_ui("disabled")

    def create_ui(self):
        # === 1. TOP BAR ===
        f_top = ctk.CTkFrame(self)
        f_top.pack(pady=10, padx=20, fill="x")
        
        self.cbox_ports = ctk.CTkComboBox(f_top, values=["Scanning..."], width=150)
        self.cbox_ports.pack(side="left", padx=5)
        self.btn_connect = ctk.CTkButton(f_top, text="CONNECT", command=self.run_connect, fg_color="#1F6AA5")
        self.btn_connect.pack(side="left", padx=5)
        self.lbl_status = ctk.CTkLabel(f_top, text="Disconnected", text_color="gray")
        self.lbl_status.pack(side="left", padx=10)
        
        self.btn_estop = ctk.CTkButton(f_top, text="🚨 E-STOP", fg_color="red", width=120, command=self.emergency_stop)
        self.btn_estop.pack(side="right", padx=10)

        # === MAIN CONTENT ===
        f_main = ctk.CTkFrame(self, fg_color="transparent")
        f_main.pack(fill="both", expand=True, padx=20, pady=5)
        f_main.columnconfigure(0, weight=1)
        f_main.columnconfigure(1, weight=1)

        # --- LEFT COLUMN: IK & Sliders ---
        f_left = ctk.CTkFrame(f_main)
        f_left.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        ctk.CTkLabel(f_left, text="XYZ CONTROL (Meters)", font=("Arial", 14, "bold")).pack(pady=5)
        
        self.var_auto_send = ctk.StringVar(value="off")
        self.sw_auto = ctk.CTkSwitch(f_left, text="Auto-Send Commands", variable=self.var_auto_send, onvalue="on", offvalue="off")
        self.sw_auto.pack(pady=5)

        grid_xyz = ctk.CTkFrame(f_left, fg_color="transparent")
        grid_xyz.pack(pady=5)
        
        # Init Vars FIRST
        self.ent_x_var = ctk.StringVar(value="0.200")
        self.ent_y_var = ctk.StringVar(value="0.000")
        self.ent_z_var = ctk.StringVar(value="0.150")
        
        self.create_axis_control(grid_xyz, "X", 0, self.ent_x_var)
        self.create_axis_control(grid_xyz, "Y", 1, self.ent_y_var)
        self.create_axis_control(grid_xyz, "Z", 2, self.ent_z_var)

        self.btn_calc = ctk.CTkButton(f_left, text="CALCULATE IK", fg_color="#D97C23", command=self.cmd_calculate)
        self.btn_calc.pack(pady=5)
        
        self.lbl_preview = ctk.CTkLabel(f_left, text="Preview: ---", text_color="gray", font=("Consolas", 12))
        self.lbl_preview.pack(pady=2)
        
        self.btn_confirm = ctk.CTkButton(f_left, text="✅ CONFIRM & MOVE", command=self.cmd_confirm, state="disabled", fg_color="green")
        self.btn_confirm.pack(pady=5)

        ctk.CTkLabel(f_left, text="__________________________", text_color="gray").pack(pady=5)

        # Sliders
        f_tune = ctk.CTkFrame(f_left, fg_color="transparent")
        f_tune.pack(fill="x", pady=5)
        ctk.CTkLabel(f_tune, text="FINE TUNE JOINTS", font=("Arial", 14, "bold")).pack(side="left")
        ctk.CTkButton(f_tune, text="SYNC POS", width=80, command=self.sync_motors_and_gui).pack(side="right")
        
        self.scroll_sliders = ctk.CTkScrollableFrame(f_left, height=300)
        self.scroll_sliders.pack(fill="both", expand=True, padx=5, pady=5)
        
        # 6 Sliders
        joints = [
            ("Base (11)", ID_BASE, 0), 
            ("Shldr (12)", ID_SHOULDER_L, 1), 
            ("Elbow (14)", ID_ELBOW, 2),
            ("Pitch (15)", ID_PITCH, 3),
            ("Roll (16)", ID_ROLL, 4),
            ("Grip (17)", ID_GRIPPER, 5) 
        ]
        
        for name, mid, idx in joints:
            f = ctk.CTkFrame(self.scroll_sliders, fg_color="transparent")
            f.pack(fill="x", pady=2)
            ctk.CTkLabel(f, text=name, width=70).pack(side="left")
            
            s = ctk.CTkSlider(f, from_=0, to=180, number_of_steps=180)
            s.set(90)
            s.configure(command=lambda v, m=mid, i=idx: self.on_slider_change(v, m, i))
            s.pack(side="left", fill="x", expand=True, padx=5)
            
            lbl = ctk.CTkLabel(f, text="90", width=30)
            lbl.pack(side="right")
            
            self.sliders[mid] = {'slider': s, 'label': lbl}

        # --- RIGHT COLUMN ---
        f_right = ctk.CTkFrame(f_main)
        f_right.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # Settings
        f_sets = ctk.CTkFrame(f_right)
        f_sets.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(f_sets, text="Speed:").pack(side="left", padx=5)
        self.ent_speed = ctk.CTkEntry(f_sets, width=50); self.ent_speed.insert(0, "600"); self.ent_speed.pack(side="left")
        ctk.CTkButton(f_sets, text="Set", width=40, command=self.set_speed).pack(side="left", padx=5)
        
        ctk.CTkButton(f_sets, text="Go Home", fg_color="green", width=80, command=self.go_home_file).pack(side="right", padx=5)
        ctk.CTkButton(f_sets, text="Set Home", fg_color="#555", width=80, command=self.set_home_file).pack(side="right", padx=5)

        # Recording
        f_rec = ctk.CTkFrame(f_right)
        f_rec.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(f_rec, text="WAYPOINT RECORDING", font=("Arial", 12, "bold")).pack(pady=5)
        ctk.CTkButton(f_rec, text="Record Current Pose", command=self.record_single_point).pack(pady=5)
        
        f_auto = ctk.CTkFrame(f_rec, fg_color="transparent")
        f_auto.pack(pady=5)
        ctk.CTkLabel(f_auto, text="Freq:").pack(side="left")
        self.sl_freq = ctk.CTkSlider(f_auto, from_=1, to=10, number_of_steps=9, width=100); self.sl_freq.set(1); self.sl_freq.pack(side="left")
        self.btn_auto_rec = ctk.CTkButton(f_rec, text="Start Auto-Record", fg_color="orange", command=self.toggle_auto_record)
        self.btn_auto_rec.pack(pady=5)

        # Replay
        f_play = ctk.CTkFrame(f_right)
        f_play.pack(fill="both", expand=True, padx=10, pady=10)
        ctk.CTkLabel(f_play, text="REPLAY SYSTEM", font=("Arial", 12, "bold")).pack(pady=5)
        
        f_file = ctk.CTkFrame(f_play, fg_color="transparent")
        f_file.pack(fill="x", pady=5)
        self.lbl_file = ctk.CTkLabel(f_file, text="File: default.csv", text_color="gray", width=150, anchor="w")
        self.lbl_file.pack(side="left", padx=5)
        ctk.CTkButton(f_file, text="📂 Load", width=60, command=self.load_replay_file).pack(side="right", padx=5)
        
        self.txt_log = ctk.CTkTextbox(f_play, height=150)
        self.txt_log.pack(fill="both", expand=True, padx=5, pady=5)
        ctk.CTkButton(f_play, text="▶ REPLAY LOADED FILE", command=self.replay_waypoints).pack(pady=10)

    # --- UI HELPERS ---
    def create_axis_control(self, parent, label, row, var_object):
        ctk.CTkButton(parent, text="-", width=30, command=lambda: self.delta_move(label, -1)).grid(row=row, column=0, padx=2)
        ent = ctk.CTkEntry(parent, width=70, textvariable=var_object)
        ent.grid(row=row, column=1, padx=2)
        ctk.CTkButton(parent, text="+", width=30, command=lambda: self.delta_move(label, 1)).grid(row=row, column=2, padx=2)
        ctk.CTkLabel(parent, text=label).grid(row=row, column=3, padx=5)

    # --- LOGIC: XYZ ---
    def delta_move(self, axis, direction):
        try:
            var_name = f"ent_{axis.lower()}_var"
            var = getattr(self, var_name)
            val = float(var.get())
            new_val = val + (DELTA_STEP * direction)
            var.set(f"{new_val:.3f}")
            self.cmd_calculate()
        except: pass

    def cmd_calculate(self):
        try:
            x = float(self.ent_x_var.get())
            y = float(self.ent_y_var.get())
            z = float(self.ent_z_var.get())
        except: return

        angles = geometric_ik(x, y, z) # Returns 5 Angles
        if angles:
            self.pending_angles = angles # 5 angles (Gripper stays same)
            res_str = f"B:{angles[0]} S:{angles[1]} E:{angles[2]} P:{angles[3]}"
            self.lbl_preview.configure(text=f"Preview: {res_str}", text_color="orange")
            if self.var_auto_send.get() == "on":
                self.cmd_confirm()
            else:
                self.btn_confirm.configure(state="normal")
        else:
            self.lbl_preview.configure(text="Unreachable!", text_color="red")
            self.btn_confirm.configure(state="disabled")

    def cmd_confirm(self):
        if self.pending_angles:
            self.btn_confirm.configure(state="disabled")
            # Update only first 5 in current state
            for i in range(5):
                self.current_calc_angles[i] = self.pending_angles[i]
                
            self.update_sliders_from_angles(self.pending_angles)
            self.move_robot(self.pending_angles, self.global_speed)

    # --- LOGIC: Sliders & Sync ---
    def on_slider_change(self, value, mid, idx):
        val = int(value)
        self.sliders[mid]['label'].configure(text=str(val))
        
        steps = int(val * STEPS_PER_DEG)
        if mid == ID_SHOULDER_L:
            self.send_packet(ID_SHOULDER_L, steps, self.global_speed)
            self.send_packet(ID_SHOULDER_R, 4096 - steps, self.global_speed)
        else:
            self.send_packet(mid, steps, self.global_speed)
            
        if idx < 6: 
            self.current_calc_angles[idx] = val

    def update_sliders_from_angles(self, angles):
        # Can handle 5 (IK) or 6 (Replay/Home)
        for i, val in enumerate(angles):
            if i >= len(CONTROL_IDS): break
            mid = CONTROL_IDS[i]
            if mid in self.sliders:
                self.sliders[mid]['slider'].set(val)
                self.sliders[mid]['label'].configure(text=str(val))

    # --- MOTOR SYNC (FIXED: Includes Gripper) ---
    def read_motor_initial(self, mid):
        cksm = (~(mid + 64)) & 0xFF
        try:
            self.ser.reset_input_buffer()
            self.ser.write(bytearray([0xFF, 0xFF, mid, 4, 2, 56, 2, cksm]))
            t = time.time()
            while time.time() - t < 0.1:
                if self.ser.in_waiting >= 8:
                    if self.ser.read(1) == b'\xff':
                        pkt = self.ser.read(7)
                        if pkt[1] == mid: return pkt[4] + (pkt[5] << 8)
        except: pass
        return None

    def sync_motors_and_gui(self):
        self.log("Syncing from Motors...")
        read_angles = []
        
        # Read ALL 6 CONTROL MOTORS (Base -> Gripper)
        for mid in CONTROL_IDS:
            raw = self.read_motor_initial(mid)
            if raw is not None:
                deg = int(raw / STEPS_PER_DEG)
            else:
                deg = 90
            read_angles.append(deg)
        
        # 1. Update Sliders (All 6)
        self.update_sliders_from_angles(read_angles)
        self.current_calc_angles = read_angles
        
        # 2. Update XYZ (Uses first 4/5)
        xyz = forward_kinematics(read_angles)
        self.ent_x_var.set(f"{xyz[0]:.3f}")
        self.ent_y_var.set(f"{xyz[1]:.3f}")
        self.ent_z_var.set(f"{xyz[2]:.3f}")
        
        self.log(f"Synced.")

    # --- LOGIC: Replay & Recording ---
    def load_replay_file(self):
        filename = filedialog.askopenfilename(title="Select Waypoint CSV", filetypes=[("CSV Files", "*.csv")])
        if filename:
            self.replay_file = filename
            short_name = os.path.basename(filename)
            self.lbl_file.configure(text=f"File: {short_name}", text_color="green")
            self.log(f"Loaded: {short_name}")

    def replay_waypoints(self):
        if not os.path.exists(self.replay_file):
            self.log("File not found!")
            return
        self.is_replaying = True
        self.log(f"Replaying...")
        threading.Thread(target=self.replay_thread).start()

    def replay_thread(self):
        with open(self.replay_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if not self.is_replaying: break
                try:
                    # FIX: Read 6 columns (Base to Gripper)
                    angles = [int(float(x)) for x in row[:6]]
                    self.update_sliders_from_angles(angles)
                    self.move_robot(angles, self.global_speed)
                    time.sleep(0.5)
                except: continue
        self.is_replaying = False
        self.log("Done.")

    def record_single_point(self):
        data = []
        # FIX: Record all 6 motors
        for mid in CONTROL_IDS:
            data.append(int(self.sliders[mid]['slider'].get()))
        self.write_csv(DEFAULT_WAYPOINT_FILE, data)
        self.log(f"Recorded: {data}")

    def toggle_auto_record(self):
        if self.is_auto_recording:
            self.is_auto_recording = False
            self.btn_auto_rec.configure(text="Start Auto-Record", fg_color="orange")
        else:
            self.is_auto_recording = True
            self.btn_auto_rec.configure(text="STOP REC", fg_color="red")
            freq = int(self.sl_freq.get())
            threading.Thread(target=self.auto_record_loop, args=(freq,)).start()

    def auto_record_loop(self, freq):
        delay = 1.0/freq
        while self.is_auto_recording:
            self.record_single_point()
            time.sleep(delay)

    def write_csv(self, filename, data):
        exists = os.path.isfile(filename)
        with open(filename, 'a', newline='') as f:
            w = csv.writer(f)
            # FIX: Header includes Gripper
            if not exists: w.writerow(["Base", "Shoulder", "Elbow", "Pitch", "Roll", "Gripper"])
            w.writerow(data)

    def emergency_stop(self):
        self.is_replaying = False
        self.is_auto_recording = False
        self.log("🚨 STOPPED!")

    def set_speed(self):
        try: self.global_speed = int(self.ent_speed.get())
        except: pass

    def log(self, msg):
        self.txt_log.insert("end", f"{msg}\n")
        self.txt_log.see("end")

    # --- HARDWARE ---
    def move_robot(self, angles, speed):
        # Can accept 5 angles (from IK) or 6 angles (from Replay/Home)
        for i, val in enumerate(angles):
            if i >= len(CONTROL_IDS): break
            mid = CONTROL_IDS[i]
            steps = int(val * STEPS_PER_DEG)
            if mid == ID_SHOULDER_L:
                self.send_packet(ID_SHOULDER_L, steps, speed)
                self.send_packet(ID_SHOULDER_R, 4096 - steps, speed)
            else:
                self.send_packet(mid, steps, speed)

    def send_packet(self, mid, steps, speed):
        if not self.ser: return
        steps = max(0, min(4095, steps))
        pl, ph = steps & 0xFF, (steps >> 8) & 0xFF
        sl, sh = speed & 0xFF, (speed >> 8) & 0xFF
        cksm = (~(mid + 9 + 3 + 42 + pl + ph + 0 + 0 + sl + sh)) & 0xFF
        msg = bytearray([0xFF, 0xFF, mid, 9, 3, 42, pl, ph, 0, 0, sl, sh, cksm])
        self.ser.write(msg)

    # --- HOME LOGIC (FIXED) ---
    def set_home_file(self):
        # Save all 6 slider values
        data = [int(self.sliders[mid]['slider'].get()) for mid in CONTROL_IDS]
        with open(HOME_FILE, 'w', newline='') as f:
            csv.writer(f).writerow(data)
        self.log("Home Set (Inc Gripper).")

    def go_home_file(self):
        if os.path.exists(HOME_FILE):
            with open(HOME_FILE, 'r') as f:
                row = next(csv.reader(f))
                angles = [int(x) for x in row[:6]]
                
                # Move all 6
                self.current_calc_angles = angles
                self.update_sliders_from_angles(angles)
                self.move_robot(angles, 400)
                
                # Update XYZ
                xyz = forward_kinematics(angles)
                self.ent_x_var.set(f"{xyz[0]:.3f}")
                self.ent_y_var.set(f"{xyz[1]:.3f}")
                self.ent_z_var.set(f"{xyz[2]:.3f}")

    # --- CONNECTION ---
    def refresh_ports(self):
        ports = serial.tools.list_ports.comports()
        self.cbox_ports.configure(values=[p.device for p in ports] or ["No Ports"])

    def run_connect(self):
        p = self.cbox_ports.get()
        if p == "No Ports": return
        self.btn_connect.configure(state="disabled", text="Connecting...")
        threading.Thread(target=self.connect_thread, args=(p,)).start()

    def connect_thread(self, port):
        try:
            self.ser = serial.Serial()
            self.ser.port = port
            self.ser.baudrate = BAUD_RATE
            self.ser.dtr = False; self.ser.rts = False
            self.ser.open()
            self.ser.dtr = False; self.ser.rts = False
            time.sleep(1.5)
            self.ser.reset_input_buffer()
            self.toggle_ui("normal")
            self.lbl_status.configure(text="Connected", text_color="green")
            
            # Torque On
            val = 1
            for mid in ALL_MOTORS_SETUP:
                 cksm = (~(mid + 4 + 3 + 40 + val)) & 0xFF
                 self.ser.write(bytearray([0xFF, 0xFF, mid, 4, 3, 40, val, cksm]))
                 time.sleep(0.005)
            
            # AUTO SYNC ON CONNECT
            self.sync_motors_and_gui()

        except Exception as e:
            self.lbl_status.configure(text="Error", text_color="red")
            self.btn_connect.configure(state="normal")

    def toggle_ui(self, state):
        self.btn_calc.configure(state=state)
        self.btn_auto_rec.configure(state=state)

if __name__ == "__main__":
    app = RobotController()
    app.mainloop()