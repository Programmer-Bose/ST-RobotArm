import customtkinter as ctk
import serial
import serial.tools.list_ports
import time
import csv
import os
import threading
import math
import numpy as np

# --- CONFIGURATION ---
BAUD_RATE = 115200       
DATA_FILE = "drl_training_data.csv"
DELTA_STEP = 0.010  # 10mm step

# Define your Safe Home Position (Meters)
HOME_XYZ = [0.200, 0.000, 0.200] 

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

ALL_MOTORS = [ID_BASE, ID_SHOULDER_L, ID_SHOULDER_R, ID_ELBOW, ID_PITCH, ID_ROLL, ID_GRIPPER]
STEPS_PER_DEG = 4096.0 / 360.0

# --- MATH CORE ---
def geometric_ik(x, y, z):
    """XYZ -> Angles. Returns None if unreachable."""
    # 1. Base Angle
    theta_base_rad = math.atan2(y, x)
    base_deg = math.degrees(theta_base_rad)
    if base_deg < 0: base_deg += 360
    
    # 2. Wrist Position
    r_total = math.sqrt(x**2 + y**2)
    z_total = z - L_BASE
    
    # Target Wrist Center (Back up from tip by L_PITCH)
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

# --- APP ---
class RobotController(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("7-DOF Master Controller V3 (With Home)")
        self.geometry("1100x800")
        ctk.set_appearance_mode("Dark")
        
        self.ser = None
        self.sliders = {}
        self.labels = {}
        self.current_xyz = HOME_XYZ
        self.pending_angles = None
        
        self.create_ui()
        self.refresh_ports()
        self.toggle_ui("disabled")
        
        self.bind("<Key>", self.handle_keypress)

    def create_ui(self):
        # 1. Connection & Home Panel
        frame_top = ctk.CTkFrame(self)
        frame_top.pack(pady=10, padx=20, fill="x")
        
        self.cbox_ports = ctk.CTkComboBox(frame_top, values=["Scanning..."])
        self.cbox_ports.pack(side="left", padx=10)
        self.btn_connect = ctk.CTkButton(frame_top, text="CONNECT", command=self.run_connect)
        self.btn_connect.pack(side="left", padx=10)
        
        # GO HOME BUTTON
        self.btn_home = ctk.CTkButton(frame_top, text="🏠 GO HOME", fg_color="#1F6AA5", command=self.cmd_go_home)
        self.btn_home.pack(side="right", padx=10)
        
        self.lbl_status = ctk.CTkLabel(frame_top, text="Disconnected", text_color="gray")
        self.lbl_status.pack(side="left", padx=10)

        # 2. Status & Velocity
        frame_stat = ctk.CTkFrame(self)
        frame_stat.pack(pady=5, padx=20, fill="x")
        
        ctk.CTkLabel(frame_stat, text="Speed (0-3000):").pack(side="left", padx=10)
        self.ent_speed = ctk.CTkEntry(frame_stat, width=60)
        self.ent_speed.insert(0, "600")
        self.ent_speed.pack(side="left", padx=5)
        
        self.var_safety = ctk.StringVar(value="safe")
        self.sw_safety = ctk.CTkSwitch(frame_stat, text="Auto-Execute", 
                                       variable=self.var_safety, onvalue="auto", offvalue="safe")
        self.sw_safety.pack(side="right", padx=20)

        # 3. Main Control Area
        frame_main = ctk.CTkFrame(self)
        frame_main.pack(pady=10, padx=20, fill="both", expand=True)
        frame_main.columnconfigure(0, weight=1)
        frame_main.columnconfigure(1, weight=1)

        # LEFT: XYZ
        frame_xyz = ctk.CTkFrame(frame_main, fg_color="transparent")
        frame_xyz.grid(row=0, column=0, sticky="nsew", padx=10)
        
        ctk.CTkLabel(frame_xyz, text="TARGET COORDINATES (M)", font=("Arial", 16, "bold")).pack(pady=10)
        
        grid_xyz = ctk.CTkFrame(frame_xyz, fg_color="transparent")
        grid_xyz.pack(pady=5)
        ctk.CTkLabel(grid_xyz, text="X").grid(row=0, column=0)
        self.ent_x = ctk.CTkEntry(grid_xyz, width=70); self.ent_x.grid(row=1, column=0, padx=5)
        
        ctk.CTkLabel(grid_xyz, text="Y").grid(row=0, column=1)
        self.ent_y = ctk.CTkEntry(grid_xyz, width=70); self.ent_y.grid(row=1, column=1, padx=5)
        
        ctk.CTkLabel(grid_xyz, text="Z").grid(row=0, column=2)
        self.ent_z = ctk.CTkEntry(grid_xyz, width=70); self.ent_z.grid(row=1, column=2, padx=5)
        
        # Pre-fill Home
        self.update_entries(HOME_XYZ)

        self.btn_calc = ctk.CTkButton(frame_xyz, text="CALCULATE / EXECUTE", command=self.cmd_calculate, fg_color="#D97C23")
        self.btn_calc.pack(pady=15, fill="x", padx=20)
        
        self.lbl_result = ctk.CTkLabel(frame_xyz, text="---", text_color="gray", font=("Consolas", 14))
        self.lbl_result.pack(pady=5)
        
        self.btn_confirm = ctk.CTkButton(frame_xyz, text="CONFIRM MOVE", command=self.cmd_confirm, state="disabled", fg_color="green")
        self.btn_confirm.pack(pady=5, fill="x", padx=20)
        
        ctk.CTkLabel(frame_xyz, text="Delta: W/S (X), A/D (Y), Q/E (Z)", text_color="gray").pack(pady=20)

        # RIGHT: Feedback
        frame_fb = ctk.CTkFrame(frame_main, fg_color="transparent")
        frame_fb.grid(row=0, column=1, sticky="nsew", padx=10)
        
        ctk.CTkLabel(frame_fb, text="LIVE MOTOR STATUS", font=("Arial", 16, "bold")).pack(pady=10)
        self.btn_sync = ctk.CTkButton(frame_fb, text="SYNC POSITION", command=self.sync_motors, width=100)
        self.btn_sync.pack(pady=5)
        
        self.scroll_motors = ctk.CTkScrollableFrame(frame_fb, height=300)
        self.scroll_motors.pack(fill="both", expand=True)
        
        joints = [("Base", ID_BASE), ("Shoulder", ID_SHOULDER_L), ("Elbow", ID_ELBOW), ("Pitch", ID_PITCH), ("Roll", ID_ROLL)]
        for name, mid in joints:
            row = ctk.CTkFrame(self.scroll_motors)
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text=name, width=60).pack(side="left")
            l = ctk.CTkLabel(row, text="---", width=40)
            l.pack(side="right", padx=10)
            self.labels[mid] = l

    # --- LOGIC ---
    def cmd_go_home(self):
        """Reset inputs to Home XYZ and Trigger Move"""
        self.update_entries(HOME_XYZ)
        self.lbl_result.configure(text="Going Home...", text_color="yellow")
        self.cmd_calculate()

    def update_entries(self, xyz):
        self.ent_x.delete(0, 'end'); self.ent_x.insert(0, f"{xyz[0]:.3f}")
        self.ent_y.delete(0, 'end'); self.ent_y.insert(0, f"{xyz[1]:.3f}")
        self.ent_z.delete(0, 'end'); self.ent_z.insert(0, f"{xyz[2]:.3f}")

    def cmd_calculate(self):
        try:
            x = float(self.ent_x.get())
            y = float(self.ent_y.get())
            z = float(self.ent_z.get())
        except ValueError:
            self.lbl_result.configure(text="Invalid XYZ", text_color="red")
            return

        angles = geometric_ik(x, y, z)
        
        if angles is None:
            self.lbl_result.configure(text="UNREACHABLE", text_color="red")
            self.btn_confirm.configure(state="disabled")
            return
            
        self.pending_angles = angles
        self.current_xyz = [x, y, z]
        
        res_str = f"B:{angles[0]} S:{angles[1]} E:{angles[2]} P:{angles[3]}"
        self.lbl_result.configure(text=res_str, text_color="orange")
        
        if self.var_safety.get() == "auto":
            self.cmd_confirm()
        else:
            self.lbl_result.configure(text=res_str + " (Pending)", text_color="orange")
            self.btn_confirm.configure(state="normal")

    def cmd_confirm(self):
        if not self.pending_angles: return
        self.btn_confirm.configure(state="disabled")
        self.lbl_result.configure(text="MOVING...", text_color="green")
        
        self.move_robot(self.pending_angles)
        self.save_data(self.current_xyz, self.pending_angles)

    def handle_keypress(self, event):
        key = event.keysym.lower()
        dx, dy, dz = 0, 0, 0
        if key == 'w': dx = DELTA_STEP
        elif key == 's': dx = -DELTA_STEP
        elif key == 'a': dy = DELTA_STEP
        elif key == 'd': dy = -DELTA_STEP
        elif key == 'q': dz = DELTA_STEP
        elif key == 'e': dz = -DELTA_STEP
        else: return
        
        try:
            cur_x = float(self.ent_x.get())
            cur_y = float(self.ent_y.get())
            cur_z = float(self.ent_z.get())
        except: return
        
        self.update_entries([cur_x + dx, cur_y + dy, cur_z + dz])
        self.cmd_calculate()

    # --- HARDWARE ---
    def move_robot(self, angles):
        try: speed = int(self.ent_speed.get())
        except: speed = 600
        
        ids = [ID_BASE, ID_SHOULDER_L, ID_ELBOW, ID_PITCH, ID_ROLL]
        for i, val in enumerate(angles):
            mid = ids[i]
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

    def sync_motors(self):
        ids = [ID_BASE, ID_SHOULDER_L, ID_ELBOW, ID_PITCH, ID_ROLL]
        for mid in ids:
            raw = self.read_pos(mid)
            if raw is not None:
                deg = int(raw / STEPS_PER_DEG)
                self.labels[mid].configure(text=f"{deg}°")

    def read_pos(self, mid):
        if not self.ser: return None
        cksm = (~(mid + 64)) & 0xFF
        try:
            self.ser.reset_input_buffer()
            self.ser.write(bytearray([0xFF, 0xFF, mid, 4, 2, 56, 2, cksm]))
            t = time.time()
            while time.time() - t < 0.05:
                if self.ser.in_waiting >= 8:
                    if self.ser.read(1) == b'\xff':
                        if self.ser.read(1) == b'\xff':
                            pkt = self.ser.read(6)
                            if pkt[0] == mid: return pkt[3] + (pkt[4] << 8)
        except: pass
        return None

    def save_data(self, xyz, angles):
        row = xyz + angles
        file_exists = os.path.isfile(DATA_FILE)
        try:
            with open(DATA_FILE, 'a', newline='') as f:
                w = csv.writer(f)
                if not file_exists: w.writerow(["X","Y","Z","Base","Shldr","Elbow","Pitch","Roll"])
                w.writerow(row)
        except: pass

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
            self.set_torque(True)
            self.sync_motors()
        except Exception as e:
            self.lbl_status.configure(text=f"Error: {e}", text_color="red")
            self.btn_connect.configure(state="normal", text="CONNECT")

    def set_torque(self, enable):
        if not self.ser: return
        val = 1 if enable else 0
        for mid in ALL_MOTORS:
            cksm = (~(mid + 4 + 3 + 40 + val)) & 0xFF
            self.ser.write(bytearray([0xFF, 0xFF, mid, 4, 3, 40, val, cksm]))
            time.sleep(0.005)

    def toggle_ui(self, state):
        self.btn_calc.configure(state=state)
        self.btn_sync.configure(state=state)
        self.btn_home.configure(state=state)

if __name__ == "__main__":
    app = RobotController()
    app.mainloop()