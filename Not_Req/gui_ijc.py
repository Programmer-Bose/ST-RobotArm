import customtkinter as ctk
import serial
import serial.tools.list_ports
import time
import csv
import os
import threading

# --- CONFIGURATION ---
BAUD_RATE = 115200       # Match your ESP32 configuration
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

class RobotArmApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window Setup
        self.title("7-DOF Robot Commander (Safe Mode)")
        self.geometry("900x800")
        self.minsize(600, 500)
        self.resizable(True, True)
        ctk.set_appearance_mode("Dark")
        
        # Variables
        self.ser = None
        self.global_speed = 600
        self.sliders = {}
        self.labels = {}
        self.joint_frames = {} 
        
        # --- UI LAYOUT ---
        self.create_header()
        self.create_connection_panel()
        self.create_velocity_panel()
        self.create_controls()
        self.create_footer()

        # Initial State
        self.refresh_ports()
        self.set_ui_state("disabled")

    def create_header(self):
        lbl = ctk.CTkLabel(self, text="ROBOT ARM CONTROL CENTER", font=("Arial", 20, "bold"))
        lbl.pack(pady=(15, 5))

    def create_connection_panel(self):
        frame = ctk.CTkFrame(self)
        frame.pack(pady=10, padx=20, fill="x")
        
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=0)
        frame.columnconfigure(2, weight=0)
        frame.columnconfigure(3, weight=2)

        self.cbox_ports = ctk.CTkComboBox(frame, values=["Scanning..."], width=200)
        self.cbox_ports.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        self.btn_refresh = ctk.CTkButton(frame, text="↻", width=40, command=self.refresh_ports)
        self.btn_refresh.grid(row=0, column=1, padx=5, pady=10)

        self.btn_connect = ctk.CTkButton(frame, text="CONNECT & PING", 
                                         command=self.run_connection_sequence, 
                                         height=35, fg_color="#1F6AA5")
        self.btn_connect.grid(row=0, column=3, padx=10, pady=10, sticky="ew")
        
        self.lbl_status = ctk.CTkLabel(frame, text="Select Port and Connect", text_color="gray")
        self.lbl_status.grid(row=1, column=0, columnspan=4, pady=(0,5))

    def create_velocity_panel(self):
        frame = ctk.CTkFrame(self)
        frame.pack(pady=5, padx=20, fill="x")
        
        lbl = ctk.CTkLabel(frame, text="Global Speed (Steps/Sec):", font=("Arial", 12))
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
            ("Shoulder (ID 12+13 Mirrored)", ID_SHOULDER_L, self.cmd_shoulder), 
            ("Elbow (ID 14)", ID_ELBOW, self.cmd_single),
            ("Wrist Pitch (ID 15)", ID_PITCH, self.cmd_single),
            ("Wrist Roll (ID 16)", ID_ROLL, self.cmd_single),
            ("Gripper (ID 17)", ID_GRIPPER, self.cmd_single),
        ]

        for name, mid, func in joints:
            self.add_joint_control(name, mid, func)

    def add_joint_control(self, name, motor_id, func):
        frame = ctk.CTkFrame(self.scroll_frame)
        frame.pack(pady=5, padx=10, fill="x")
        self.joint_frames[motor_id] = frame 

        info_frame = ctk.CTkFrame(frame, fg_color="transparent")
        info_frame.pack(side="top", fill="x", padx=10, pady=5)
        
        lbl_title = ctk.CTkLabel(info_frame, text=name, font=("Arial", 14, "bold"))
        lbl_title.pack(side="left")

        lbl_val = ctk.CTkLabel(info_frame, text="---", font=("Arial", 14), text_color="#1F6AA5")
        lbl_val.pack(side="right")
        self.labels[motor_id] = lbl_val

        slider = ctk.CTkSlider(frame, from_=0, to=180, number_of_steps=180, height=20)
        slider.set(0)
        slider.pack(pady=(0, 10), padx=10, fill="x")
        
        slider.configure(command=lambda val, m_id=motor_id, f=func: self.on_slider_move(val, m_id, f))
        self.sliders[motor_id] = slider

    def create_footer(self):
        frame = ctk.CTkFrame(self, height=50)
        frame.pack(pady=15, padx=20, fill="x", side="bottom")

        self.btn_home = ctk.CTkButton(frame, text="GO HOME", command=self.go_home, 
                                      fg_color="green", height=40, font=("Arial", 14, "bold"))
        self.btn_home.pack(side="left", padx=20, pady=10, expand=True, fill="x")

        self.btn_set_home = ctk.CTkButton(frame, text="SET HOME", command=self.set_home, 
                                          fg_color="#D35B58", height=40, font=("Arial", 14, "bold"))
        self.btn_set_home.pack(side="right", padx=20, pady=10, expand=True, fill="x")

    # --- LOGIC: PORTS ---
    def refresh_ports(self):
        ports = serial.tools.list_ports.comports()
        port_list = [port.device for port in ports]
        if not port_list:
            port_list = ["No Ports Found"]
        self.cbox_ports.configure(values=port_list)
        self.cbox_ports.set(port_list[0])

    # --- LOGIC: CONNECTION (FIXED FOR ESP32 RESET) ---
    def run_connection_sequence(self):
        selected_port = self.cbox_ports.get()
        if selected_port == "No Ports Found" or not selected_port:
            self.update_status("Error: No Port Selected", "red")
            return

        self.btn_connect.configure(state="disabled", text="Connecting...")
        self.lbl_status.configure(text=f"Opening {selected_port}...", text_color="yellow")
        
        threading.Thread(target=self._connect_thread, args=(selected_port,)).start()

    def _connect_thread(self, port_name):
        # 1. Open Serial SAFELY (No DTR/RTS spike)
        if self.ser is None or not self.ser.is_open:
            try:
                # Create object WITHOUT opening
                self.ser = serial.Serial()
                self.ser.port = port_name
                self.ser.baudrate = BAUD_RATE
                self.ser.timeout = 0.05
                
                # Force pins LOW before opening
                self.ser.dtr = False
                self.ser.rts = False
                
                self.ser.open()
                
                # Force again just to be safe
                self.ser.dtr = False
                self.ser.rts = False
                
                # Wait for electrical stabilization
                time.sleep(1.5) 
                
                self.ser.reset_input_buffer()
                self.ser.reset_output_buffer()
                
            except Exception as e:
                self.update_status(f"Connection Failed: {e}", "red")
                self.reset_connect_btn()
                return

        # 2. Ping Loop
        missing = []
        for mid in ALL_MOTORS:
            self.update_status(f"Pinging ID {mid}...", "white")
            if not self.ping_motor(mid):
                missing.append(mid)
        
        # 3. Validation
        if len(missing) == 0:
            self.update_status(f"Connected to {port_name}. All Systems Go.", "green")
            self.after(100, self.enable_ui_and_sync)
        else:
            self.update_status(f"Warning: IDs {missing} did not reply.", "orange")
            self.after(100, self.enable_ui_and_sync)

    def ping_motor(self, servo_id):
        # ST3215 Ping Instruction
        checksum = (~(servo_id + 2 + 1)) & 0xFF
        cmd = bytearray([0xFF, 0xFF, servo_id, 2, 1, checksum])
        
        try:
            self.ser.reset_input_buffer()
            self.ser.write(cmd)
            time.sleep(0.05)
            if self.ser.in_waiting >= 6:
                if self.ser.read(2) == b'\xff\xff':
                    if self.ser.read(1)[0] == servo_id:
                        return True
        except:
            return False
        return False

    def update_status(self, text, color):
        self.lbl_status.configure(text=text, text_color=color)

    def reset_connect_btn(self):
        self.btn_connect.configure(state="normal", text="CONNECT & PING")
        if self.ser:
            self.ser.close()
        self.ser = None

    # --- LOGIC: STATE & CONTROL ---
    def set_ui_state(self, state):
        for slider in self.sliders.values():
            slider.configure(state=state)
        self.btn_home.configure(state=state)
        self.btn_set_home.configure(state=state)

    def enable_ui_and_sync(self):
        self.btn_connect.configure(text="CONNECTED", fg_color="green")
        self.set_ui_state("normal")
        
        # Sync Sliders
        read_ids = [ID_BASE, ID_SHOULDER_L, ID_ELBOW, ID_PITCH, ID_ROLL, ID_GRIPPER]
        for mid in read_ids:
            pos = self.read_position(mid)
            if pos is not None:
                deg = int(pos / STEPS_PER_DEG)
                if deg > 180: deg = 180
                self.sliders[mid].set(deg)
                self.labels[mid].configure(text=f"{deg}°")

    def update_speed(self):
        try:
            val = int(self.entry_speed.get())
            self.global_speed = max(0, min(3000, val))
            self.entry_speed.configure(fg_color="#2B7A4B")
            self.after(500, lambda: self.entry_speed.configure(fg_color=["#F9F9FA", "#343638"]))
        except ValueError:
            pass

    def on_slider_move(self, value, motor_id, func):
        deg = int(value)
        self.labels[motor_id].configure(text=f"{deg}°")
        func(motor_id, deg)

    def cmd_single(self, motor_id, degrees):
        steps = int(degrees * STEPS_PER_DEG)
        self.send_packet(motor_id, steps, self.global_speed)

    def cmd_shoulder(self, motor_id, degrees):
        steps_l = int(degrees * STEPS_PER_DEG)
        steps_r = 4096 - steps_l # Mirror logic
        if steps_r < 0: steps_r = 0
        if steps_r > 4095: steps_r = 4095
        
        self.send_packet(ID_SHOULDER_L, steps_l, self.global_speed)
        self.send_packet(ID_SHOULDER_R, steps_r, self.global_speed)

    def send_packet(self, servo_id, steps, speed):
        if not self.ser: return
        if servo_id != ID_SHOULDER_R and steps > MAX_STEPS: steps = int(MAX_STEPS)

        p_low = int(steps) & 0xFF
        p_high = (int(steps) >> 8) & 0xFF
        s_low = int(speed) & 0xFF
        s_high = (int(speed) >> 8) & 0xFF

        checksum_sum = servo_id + 9 + 3 + 42 + p_low + p_high + 0 + 0 + s_low + s_high
        checksum = (~checksum_sum) & 0xFF
        
        cmd = bytearray([0xFF, 0xFF, servo_id, 9, 3, 42, p_low, p_high, 0, 0, s_low, s_high, checksum])
        self.ser.write(cmd)

    def read_position(self, servo_id):
        checksum = (~(servo_id + 68)) & 0xFF
        cmd = bytearray([0xFF, 0xFF, servo_id, 4, 2, 56, 2, checksum])
        try:
            self.ser.reset_input_buffer()
            self.ser.write(cmd)
            time.sleep(0.02)
            if self.ser.in_waiting >= 8:
                if self.ser.read(2) == b'\xff\xff':
                    packet = self.ser.read(6)
                    if packet[0] == servo_id:
                        return packet[3] + (packet[4] << 8)
        except:
            return None
        return None

    # --- HOME LOGIC ---
    def set_home(self):
        data = []
        for mid, slider in self.sliders.items():
            data.append([mid, int(slider.get())])
        try:
            with open(CSV_FILE, 'w', newline='') as f:
                csv.writer(f).writerows(data)
            self.btn_set_home.configure(fg_color="#A03030")
            self.after(500, lambda: self.btn_set_home.configure(fg_color="#D35B58"))
        except Exception as e:
            print(f"Error: {e}")

    def go_home(self):
        if not os.path.exists(CSV_FILE): return
        try:
            with open(CSV_FILE, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row: continue
                    mid = int(row[0])
                    deg = int(row[1])
                    if mid in self.sliders:
                        self.sliders[mid].set(deg)
                        self.labels[mid].configure(text=f"{deg}°")
                        
                        if mid == ID_SHOULDER_L: self.cmd_shoulder(mid, deg)
                        else: self.cmd_single(mid, deg)
                        
                        self.update()
                        time.sleep(0.5)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    app = RobotArmApp()
    app.mainloop()