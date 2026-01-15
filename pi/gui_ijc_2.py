import customtkinter as ctk
import serial
import serial.tools.list_ports
import time
import csv
import os
import threading
import numpy as np

# --- CONFIGURATION ---
BAUD_RATE = 115200       
CSV_FILE = 'robot_home.csv'

# Motor IDs
ID_BASE = 11
ID_SHOULDER_L = 12
ID_SHOULDER_R = 13
ID_ELBOW = 14
ID_PITCH = 15
ID_ROLL = 16
ID_GRIPPER = 17

ALL_MOTORS = [ID_BASE, ID_SHOULDER_L, ID_SHOULDER_R, ID_ELBOW, ID_PITCH, ID_ROLL, ID_GRIPPER]
MAX_STEPS = 2048.0       
STEPS_PER_DEG = 4096.0 / 360.0

# --- KINEMATICS FUNCTIONS ---
def get_dh_transform(a, alpha, d, theta):
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    c_alpha = np.cos(alpha)
    s_alpha = np.sin(alpha)
    return np.array([
        [c_theta, -s_theta*c_alpha,  s_theta*s_alpha, a * c_theta],
        [s_theta,  c_theta*c_alpha, -c_theta*s_alpha, a * s_theta],
        [0,        s_alpha,          c_alpha,         d],
        [0,        0,                0,               1]
    ])

def calculate_fk(angles_deg):
    servo_offsets = np.array([0.0, 0.0, 180.0, 90.0, 0.0]) 
    
    dh_angles = np.array(angles_deg) - servo_offsets
    q = np.radians(dh_angles)

    dh_table = [
        {'a': 0,     'alpha': np.pi/2,  'd': 0.10, 'theta': q[0]}, 
        {'a': 0.245,  'alpha': 0,        'd': 0,    'theta': q[1]}, 
        {'a': 0.145,  'alpha': 0,        'd': 0,    'theta': q[2]}, 
        {'a': 0.155,  'alpha': -np.pi/2, 'd': 0,    'theta': q[3]}, 
        {'a': 0,     'alpha': 0,        'd': 0,    'theta': q[4]}  
    ]

    T_total = np.eye(4)
    for row in dh_table:
        T_link = get_dh_transform(row['a'], row['alpha'], row['d'], row['theta'])
        T_total = np.dot(T_total, T_link)

    return T_total[:3, 3]

# --- MAIN APPLICATION ---
class RobotArmApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("7-DOF Commander (Final Fixed)")
        self.geometry("1000x800")
        self.minsize(800, 600)
        ctk.set_appearance_mode("Dark")
        
        self.ser = None
        self.global_speed = 600
        self.sliders = {}
        self.labels = {}
        self.is_syncing = False
        
        self.create_header()
        self.create_connection_panel()
        self.create_status_panel()
        self.create_velocity_panel()
        self.create_controls()
        self.create_footer()

        self.refresh_ports()
        self.set_ui_state("disabled")

    def create_header(self):
        lbl = ctk.CTkLabel(self, text="ROBOT ARM CONTROL CENTER", font=("Arial", 20, "bold"))
        lbl.pack(pady=(15, 5))

    def create_connection_panel(self):
        frame = ctk.CTkFrame(self)
        frame.pack(pady=5, padx=20, fill="x")
        
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(3, weight=0)

        self.cbox_ports = ctk.CTkComboBox(frame, values=["Scanning..."], width=200)
        self.cbox_ports.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        self.btn_refresh = ctk.CTkButton(frame, text="↻", width=40, command=self.refresh_ports)
        self.btn_refresh.grid(row=0, column=1, padx=5, pady=10)

        self.btn_connect = ctk.CTkButton(frame, text="CONNECT & PING", command=self.run_connection_sequence, 
                                         fg_color="#1F6AA5")
        self.btn_connect.grid(row=0, column=2, padx=10, pady=10)
        
        self.lbl_status = ctk.CTkLabel(frame, text="Select Port...", text_color="gray")
        self.lbl_status.grid(row=1, column=0, columnspan=3, pady=(0,5))

    def create_status_panel(self):
        self.fk_frame = ctk.CTkFrame(self, fg_color="#2B2B2B")
        self.fk_frame.pack(pady=5, padx=20, fill="x")
        
        lbl_title = ctk.CTkLabel(self.fk_frame, text="END EFFECTOR POSITION (Meters)", 
                                 font=("Arial", 12, "bold"), text_color="gray")
        lbl_title.pack(pady=(5,0))
        
        grid = ctk.CTkFrame(self.fk_frame, fg_color="transparent")
        grid.pack(pady=5)
        
        self.lbl_x = ctk.CTkLabel(grid, text="X: 0.000", font=("Consolas", 18, "bold"), text_color="#4ea8de", width=100)
        self.lbl_x.grid(row=0, column=0, padx=20)
        
        self.lbl_y = ctk.CTkLabel(grid, text="Y: 0.000", font=("Consolas", 18, "bold"), text_color="#4ea8de", width=100)
        self.lbl_y.grid(row=0, column=1, padx=20)
        
        self.lbl_z = ctk.CTkLabel(grid, text="Z: 0.000", font=("Consolas", 18, "bold"), text_color="#4ea8de", width=100)
        self.lbl_z.grid(row=0, column=2, padx=20)

    def create_velocity_panel(self):
        frame = ctk.CTkFrame(self)
        frame.pack(pady=5, padx=20, fill="x")
        
        lbl = ctk.CTkLabel(frame, text="Global Speed:", font=("Arial", 12))
        lbl.pack(side="left", padx=15)
        
        self.entry_speed = ctk.CTkEntry(frame, width=80)
        self.entry_speed.insert(0, "600")
        self.entry_speed.pack(side="left", padx=5)
        
        btn_set = ctk.CTkButton(frame, text="SET", width=60, command=self.update_speed)
        btn_set.pack(side="left", padx=10)

    def create_controls(self):
        self.scroll_frame = ctk.CTkScrollableFrame(self, label_text="Joint Controls")
        self.scroll_frame.pack(pady=10, padx=20, fill="both", expand=True)

        joints = [
            ("Base (ID 11)", ID_BASE, self.cmd_single),
            ("Shoulder (ID 12+13)", ID_SHOULDER_L, self.cmd_shoulder), 
            ("Elbow (ID 14)", ID_ELBOW, self.cmd_single),
            ("Pitch (ID 15)", ID_PITCH, self.cmd_single),
            ("Roll (ID 16)", ID_ROLL, self.cmd_single),
            ("Gripper (ID 17)", ID_GRIPPER, self.cmd_single),
        ]

        for name, mid, func in joints:
            self.add_joint_control(name, mid, func)

    def add_joint_control(self, name, motor_id, func):
        frame = ctk.CTkFrame(self.scroll_frame)
        frame.pack(pady=5, padx=10, fill="x")

        info_frame = ctk.CTkFrame(frame, fg_color="transparent")
        info_frame.pack(side="top", fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(info_frame, text=name, font=("Arial", 14, "bold")).pack(side="left")
        self.labels[motor_id] = ctk.CTkLabel(info_frame, text="---", font=("Arial", 14), text_color="#1F6AA5")
        self.labels[motor_id].pack(side="right")

        slider = ctk.CTkSlider(frame, from_=0, to=180, number_of_steps=180)
        slider.set(0)
        slider.pack(pady=(0, 10), padx=10, fill="x")
        
        slider.configure(command=lambda val, m_id=motor_id, f=func: self.on_slider_move(val, m_id, f))
        self.sliders[motor_id] = slider

    def create_footer(self):
        frame = ctk.CTkFrame(self, height=50)
        frame.pack(pady=15, padx=20, fill="x", side="bottom")
        ctk.CTkButton(frame, text="GO HOME", command=self.go_home, fg_color="green").pack(side="left", padx=20, expand=True, fill="x")
        ctk.CTkButton(frame, text="SET HOME", command=self.set_home, fg_color="#D35B58").pack(side="right", padx=20, expand=True, fill="x")

    def update_fk_display(self):
        try:
            angles = []
            ids = [ID_BASE, ID_SHOULDER_L, ID_ELBOW, ID_PITCH, ID_ROLL]
            for mid in ids:
                angles.append(self.sliders[mid].get() if mid in self.sliders else 0.0)
            
            pos = calculate_fk(angles)
            self.lbl_x.configure(text=f"X: {pos[0]:.3f}")
            self.lbl_y.configure(text=f"Y: {pos[1]:.3f}")
            self.lbl_z.configure(text=f"Z: {pos[2]:.3f}")
        except: pass

    def on_slider_move(self, value, motor_id, func):
        deg = int(value)
        self.labels[motor_id].configure(text=f"{deg}°")
        
        if not self.is_syncing:
            func(motor_id, deg)
        
        self.update_fk_display()

    # --- CONNECTION ---
    def refresh_ports(self):
        ports = serial.tools.list_ports.comports()
        port_list = [p.device for p in ports] or ["No Ports"]
        self.cbox_ports.configure(values=port_list)
        self.cbox_ports.set(port_list[0])

    def run_connection_sequence(self):
        port = self.cbox_ports.get()
        if port == "No Ports": return
        
        self.btn_connect.configure(state="disabled", text="Connecting...")
        self.lbl_status.configure(text=f"Opening {port}...", text_color="yellow")
        threading.Thread(target=self._connect_thread, args=(port,)).start()

    def _connect_thread(self, port_name):
        if self.ser is None or not self.ser.is_open:
            try:
                self.ser = serial.Serial()
                self.ser.port = port_name
                self.ser.baudrate = BAUD_RATE
                self.ser.timeout = 0.05
                self.ser.dtr = False
                self.ser.rts = False
                self.ser.open()
                self.ser.dtr = False
                self.ser.rts = False
                time.sleep(1.5)
                self.ser.reset_input_buffer()
            except Exception as e:
                self.lbl_status.configure(text=f"Error: {e}", text_color="red")
                self.btn_connect.configure(state="normal", text="CONNECT")
                return

        missing = []
        for mid in ALL_MOTORS:
            if not self.ping_motor(mid): missing.append(mid)
        
        if not missing:
            self.lbl_status.configure(text="Connected. Syncing...", text_color="green")
            self.after(100, self.enable_ui_and_sync)
        else:
            self.lbl_status.configure(text=f"Missing {missing}. Syncing anyway...", text_color="orange")
            self.after(100, self.enable_ui_and_sync)

    def ping_motor(self, servo_id):
        try:
            cksm = (~(servo_id + 3)) & 0xFF
            self.ser.write(bytearray([0xFF, 0xFF, servo_id, 2, 1, cksm]))
            time.sleep(0.02)
            return self.ser.in_waiting >= 6
        except: return False

    # --- CONTROL ---
    def set_ui_state(self, state):
        for s in self.sliders.values(): s.configure(state=state)

    def enable_ui_and_sync(self):
        self.btn_connect.configure(text="CONNECTED", fg_color="green")
        self.set_ui_state("normal")
        self.is_syncing = True # LOCK COMMANDS
        
        print("Syncing Sliders...")
        read_ids = [ID_BASE, ID_SHOULDER_L, ID_ELBOW, ID_PITCH, ID_ROLL, ID_GRIPPER]
        for mid in read_ids:
            pos = self.read_position_robust(mid)
            if pos is not None:
                deg = int(pos / STEPS_PER_DEG)
                if deg > 180: deg = 180
                self.sliders[mid].set(deg)
                self.labels[mid].configure(text=f"{deg}°")
                print(f"Synced ID {mid}: {deg} deg")
            else:
                print(f"Failed ID {mid}")
        
        self.is_syncing = False # UNLOCK
        self.update_fk_display()

    def cmd_single(self, motor_id, degrees):
        steps = int(degrees * STEPS_PER_DEG)
        self.send_packet(motor_id, steps, self.global_speed)

    def cmd_shoulder(self, motor_id, degrees):
        steps_l = int(degrees * STEPS_PER_DEG)
        steps_r = 4096 - steps_l
        if steps_r < 0: steps_r = 0
        if steps_r > 4095: steps_r = 4095
        self.send_packet(ID_SHOULDER_L, steps_l, self.global_speed)
        self.send_packet(ID_SHOULDER_R, steps_r, self.global_speed)

    def send_packet(self, servo_id, steps, speed):
        if not self.ser: return
        if servo_id != ID_SHOULDER_R and steps > MAX_STEPS: steps = int(MAX_STEPS)
        
        p_l, p_h = steps & 0xFF, (steps >> 8) & 0xFF
        s_l, s_h = speed & 0xFF, (speed >> 8) & 0xFF
        cksm = (~(servo_id + 9 + 3 + 42 + p_l + p_h + 0 + 0 + s_l + s_h)) & 0xFF
        self.ser.write(bytearray([0xFF, 0xFF, servo_id, 9, 3, 42, p_l, p_h, 0, 0, s_l, s_h, cksm]))

    def read_position_robust(self, servo_id):
        # CORRECT CHECKSUM: ID + 4 + 2 + 56 + 2 = ID + 64
        cksm = (~(servo_id + 64)) & 0xFF
        try:
            self.ser.reset_input_buffer()
            self.ser.write(bytearray([0xFF, 0xFF, servo_id, 4, 2, 56, 2, cksm]))
            
            start = time.time()
            while (time.time() - start) < 0.15:
                if self.ser.in_waiting >= 8:
                    if self.ser.read(1) == b'\xff':
                        if self.ser.read(1) == b'\xff':
                            pkt = self.ser.read(6)
                            if pkt[0] == servo_id:
                                return pkt[3] + (pkt[4] << 8)
        except: pass
        return None

    def update_speed(self):
        try: self.global_speed = int(self.entry_speed.get())
        except: pass

    def set_home(self):
        data = [[m, int(s.get())] for m, s in self.sliders.items()]
        with open(CSV_FILE, 'w', newline='') as f: csv.writer(f).writerows(data)

    def go_home(self):
        if not os.path.exists(CSV_FILE): return
        with open(CSV_FILE, 'r') as f:
            for row in csv.reader(f):
                if not row: continue
                mid, deg = int(row[0]), int(row[1])
                if mid in self.sliders:
                    self.sliders[mid].set(deg)
                    self.on_slider_move(deg, mid, self.cmd_shoulder if mid == ID_SHOULDER_L else self.cmd_single)
                    self.update()
                    time.sleep(0.5)

if __name__ == "__main__":
    app = RobotArmApp()
    app.mainloop()