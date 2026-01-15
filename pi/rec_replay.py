import customtkinter as ctk
import serial
import serial.tools.list_ports
import time
import csv
import os
import threading

# --- CONFIGURATION ---
BAUD_RATE = 115200       
DEFAULT_FILENAME = "waypoints.csv"

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

class RobotWaypointApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Robot Waypoint Recorder (Speed + Delay Control)")
        self.geometry("900x750")
        ctk.set_appearance_mode("Dark")
        
        self.ser = None
        self.waypoints = [] 
        self.is_playing = False
        
        # UI Layout
        self.create_connection_panel()
        self.create_torque_panel()
        self.create_waypoint_panel()
        self.create_list_panel()
        
        self.refresh_ports()
        self.toggle_ui_state("disabled")

    def create_connection_panel(self):
        frame = ctk.CTkFrame(self)
        frame.pack(pady=10, padx=20, fill="x")
        
        self.cbox_ports = ctk.CTkComboBox(frame, values=["Scanning..."])
        self.cbox_ports.pack(side="left", padx=10, pady=10)
        
        self.btn_connect = ctk.CTkButton(frame, text="CONNECT", command=self.run_connection)
        self.btn_connect.pack(side="left", padx=10)
        
        self.lbl_conn = ctk.CTkLabel(frame, text="Disconnected", text_color="gray")
        self.lbl_conn.pack(side="left", padx=10)

    def create_torque_panel(self):
        frame = ctk.CTkFrame(self)
        frame.pack(pady=5, padx=20, fill="x")
        
        ctk.CTkLabel(frame, text="1. TEACHING MODE", font=("Arial", 12, "bold")).pack(pady=5)
        
        grid = ctk.CTkFrame(frame, fg_color="transparent")
        grid.pack(pady=5)
        
        self.btn_torque_off = ctk.CTkButton(grid, text="UNLOCK (FREE)", 
                                            fg_color="#D35B58", hover_color="#A03030",
                                            command=lambda: self.set_torque_thread(False))
        self.btn_torque_off.grid(row=0, column=0, padx=10)
        
        self.btn_torque_on = ctk.CTkButton(grid, text="LOCK (HOLD)", 
                                           fg_color="#2B7A4B", hover_color="#1E5E38",
                                           command=lambda: self.set_torque_thread(True))
        self.btn_torque_on.grid(row=0, column=1, padx=10)
        
        self.lbl_torque_status = ctk.CTkLabel(frame, text="Status: Unknown", font=("Arial", 12))
        self.lbl_torque_status.pack(pady=5)

    def create_waypoint_panel(self):
        frame = ctk.CTkFrame(self)
        frame.pack(pady=10, padx=20, fill="x")
        
        ctk.CTkLabel(frame, text="2. CAPTURE & PLAY", font=("Arial", 12, "bold")).pack(pady=5)
        
        self.btn_capture = ctk.CTkButton(frame, text="📸 CAPTURE CURRENT POSITION", 
                                         height=50, font=("Arial", 14, "bold"),
                                         fg_color="#D97C23", command=self.capture_point)
        self.btn_capture.pack(pady=10, fill="x", padx=40)
        
        # --- NEW CONTROL SECTION ---
        ctrl = ctk.CTkFrame(frame, fg_color="transparent")
        ctrl.pack(pady=5)
        
        # Input 1: Motor Speed
        ctk.CTkLabel(ctrl, text="Motor Speed (0-3000):").pack(side="left", padx=5)
        self.ent_motor_speed = ctk.CTkEntry(ctrl, width=60)
        self.ent_motor_speed.insert(0, "600") # Default Speed
        self.ent_motor_speed.pack(side="left", padx=5)

        # Input 2: Delay between points
        ctk.CTkLabel(ctrl, text="Delay (sec):").pack(side="left", padx=5)
        self.ent_delay = ctk.CTkEntry(ctrl, width=60)
        self.ent_delay.insert(0, "1.0") # Default Wait
        self.ent_delay.pack(side="left", padx=5)
        
        self.btn_play = ctk.CTkButton(ctrl, text="▶ PLAY SEQUENCE", fg_color="#1F6AA5",
                                      command=self.start_playback)
        self.btn_play.pack(side="left", padx=20)

    def create_list_panel(self):
        frame = ctk.CTkFrame(self)
        frame.pack(pady=10, padx=20, fill="both", expand=True)
        
        io = ctk.CTkFrame(frame, fg_color="transparent")
        io.pack(pady=5, fill="x")
        
        self.ent_filename = ctk.CTkEntry(io, placeholder_text="filename.csv")
        self.ent_filename.pack(side="left", padx=5, expand=True, fill="x")
        
        ctk.CTkButton(io, text="Save CSV", width=80, command=self.save_csv).pack(side="left", padx=2)
        ctk.CTkButton(io, text="Load CSV", width=80, command=self.load_csv).pack(side="left", padx=2)
        ctk.CTkButton(io, text="Clear List", width=80, fg_color="red", command=self.clear_list).pack(side="left", padx=2)

        self.txt_list = ctk.CTkTextbox(frame)
        self.txt_list.pack(pady=5, padx=5, fill="both", expand=True)

    # --- LOGGING ---
    def log(self, msg):
        self.after(0, lambda: self._log_internal(msg))

    def _log_internal(self, msg):
        self.txt_list.insert("end", f"{msg}\n")
        self.txt_list.see("end")

    # --- LOGIC: CAPTURE ---
    def capture_point(self):
        self.btn_capture.configure(text="Reading...", state="disabled")
        threading.Thread(target=self._capture_thread).start()

    def _capture_thread(self):
        read_ids = [ID_BASE, ID_SHOULDER_L, ID_ELBOW, ID_PITCH, ID_ROLL, ID_GRIPPER]
        point = []
        
        for mid in read_ids:
            raw = self.read_position(mid)
            if raw is not None:
                deg = round(raw / STEPS_PER_DEG, 1)
                point.append(deg)
            else:
                self.log(f"Error reading ID {mid}. Try again.")
                self.after(0, lambda: self.btn_capture.configure(text="📸 CAPTURE CURRENT POSITION", state="normal"))
                return

        self.waypoints.append(point)
        idx = len(self.waypoints)
        self.log(f"Point {idx} Captured: {point}")
        self.after(0, lambda: self.btn_capture.configure(text="📸 CAPTURE CURRENT POSITION", state="normal"))

    def read_position(self, mid):
        cksm = (~(mid + 64)) & 0xFF
        cmd = bytearray([0xFF, 0xFF, mid, 4, 2, 56, 2, cksm])
        try:
            self.ser.reset_input_buffer()
            self.ser.write(cmd)
            t = time.time()
            while time.time() - t < 0.1: 
                if self.ser.in_waiting >= 8:
                    if self.ser.read(1) == b'\xff':
                        if self.ser.read(1) == b'\xff':
                            pkt = self.ser.read(6)
                            if pkt[0] == mid:
                                return pkt[3] + (pkt[4] << 8)
        except: pass
        return None

    # --- LOGIC: PLAYBACK (FIXED SPEED & DELAY) ---
    def start_playback(self):
        if not self.waypoints:
            self.log("No waypoints to play!")
            return
        
        try:
            motor_speed = int(self.ent_motor_speed.get()) # e.g. 600
            delay_sec = float(self.ent_delay.get())       # e.g. 1.0
        except:
            self.log("Invalid speed or delay.")
            return

        self.log(f"Playing {len(self.waypoints)} points... Speed={motor_speed}, Delay={delay_sec}s")
        threading.Thread(target=self._play_sequence, args=(motor_speed, delay_sec)).start()

    def _play_sequence(self, motor_speed, delay_sec):
        self.set_torque_logic(True) # Ensure locked
        time.sleep(0.5)
        
        self.is_playing = True
        self.after(0, lambda: self.btn_play.configure(state="disabled"))
        
        ids = [ID_BASE, ID_SHOULDER_L, ID_ELBOW, ID_PITCH, ID_ROLL, ID_GRIPPER]
        
        for i, point in enumerate(self.waypoints):
            if not self.is_playing: break
            self.log(f"Moving to Point {i+1}...")
            
            for j, deg in enumerate(point):
                mid = ids[j]
                steps = int(deg * STEPS_PER_DEG)
                if mid == ID_SHOULDER_L:
                    self.send_motor(ID_SHOULDER_L, steps, motor_speed)
                    self.send_motor(ID_SHOULDER_R, 4096 - steps, motor_speed)
                else:
                    self.send_motor(mid, steps, motor_speed)
            
            # Wait for the specified delay before next point
            time.sleep(delay_sec)
            
        self.log("Sequence Complete.")
        self.after(0, lambda: self.btn_play.configure(state="normal"))
        self.is_playing = False

    def send_motor(self, mid, steps, speed):
        # SPEED-BASED MOVE COMMAND
        # Steps: Position (0-4095)
        # Speed: Velocity (0-3000)
        
        steps = max(0, min(4095, steps))
        
        pl = steps & 0xFF
        ph = (steps >> 8) & 0xFF
        
        # Time Bytes must be 0 for Speed control to work
        tl = 0
        th = 0
        
        sl = speed & 0xFF
        sh = (speed >> 8) & 0xFF
        
        # Checksum: ID + Len(9) + Instr(3) + Addr(42) + Data...
        cksm = (~(mid + 9 + 3 + 42 + pl + ph + tl + th + sl + sh)) & 0xFF
        
        # Packet: Header(2), ID(1), Len(1), Instr(1), Addr(1), Data(6), Cksm(1)
        msg = bytearray([0xFF, 0xFF, mid, 9, 3, 42, pl, ph, tl, th, sl, sh, cksm])
        self.ser.write(msg)

    # --- SERIAL & TORQUE ---
    def refresh_ports(self):
        ports = serial.tools.list_ports.comports()
        self.cbox_ports.configure(values=[p.device for p in ports] or ["No Ports"])

    def run_connection(self):
        p = self.cbox_ports.get()
        if p == "No Ports": return
        self.btn_connect.configure(state="disabled", text="Connecting...")
        threading.Thread(target=self.connect, args=(p,)).start()

    def connect(self, port):
        try:
            self.ser = serial.Serial()
            self.ser.port = port
            self.ser.baudrate = BAUD_RATE
            self.ser.dtr = False; self.ser.rts = False
            self.ser.open()
            self.ser.dtr = False; self.ser.rts = False
            time.sleep(1.5)
            self.ser.reset_input_buffer()
            
            self.after(0, lambda: self.toggle_ui_state("normal"))
            self.after(0, lambda: self.lbl_conn.configure(text="Connected", text_color="green"))
            self.log("Connected. Locking Motors...")
            self.set_torque_logic(True)
        except Exception as e:
            self.log(f"Conn Error: {e}")
            self.after(0, lambda: self.btn_connect.configure(state="normal", text="CONNECT"))

    def set_torque_thread(self, enable):
        threading.Thread(target=self.set_torque_logic, args=(enable,)).start()

    def set_torque_logic(self, enable):
        if not self.ser: return
        val = 1 if enable else 0
        state = "LOCKED" if enable else "FREE"
        
        color = "green" if enable else "red"
        self.after(0, lambda: self.lbl_torque_status.configure(text=f"Status: {state}", text_color=color))
        self.log(f"Setting Torque -> {state}...")
        
        for i in range(3):
            for mid in ALL_MOTORS:
                cksm = (~(mid + 4 + 3 + 40 + val)) & 0xFF
                msg = bytearray([0xFF, 0xFF, mid, 4, 3, 40, val, cksm])
                self.ser.write(msg)
                time.sleep(0.002) 

    # --- FILE IO ---
    def save_csv(self):
        fname = self.ent_filename.get() or DEFAULT_FILENAME
        if not fname.endswith(".csv"): fname += ".csv"
        try:
            with open(fname, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Base", "Shoulder", "Elbow", "Pitch", "Roll", "Gripper"])
                writer.writerows(self.waypoints)
            self.log(f"Saved {len(self.waypoints)} points.")
        except Exception as e: self.log(str(e))

    def load_csv(self):
        fname = self.ent_filename.get()
        if not os.path.exists(fname): return
        try:
            with open(fname, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                self.waypoints = [[float(x) for x in row] for row in reader]
            self.log(f"Loaded {len(self.waypoints)} points.")
            self.refresh_list_display()
        except: pass

    def clear_list(self):
        self.waypoints = []
        self.refresh_list_display()

    def refresh_list_display(self):
        self.txt_list.delete("1.0", "end")
        for i, pt in enumerate(self.waypoints):
            self.txt_list.insert("end", f"Point {i+1}: {pt}\n")

    def toggle_ui_state(self, state):
        self.btn_torque_off.configure(state=state)
        self.btn_torque_on.configure(state=state)
        self.btn_capture.configure(state=state)
        self.btn_play.configure(state=state)

if __name__ == "__main__":
    app = RobotWaypointApp()
    app.mainloop()