import customtkinter as ctk
import serial
import serial.tools.list_ports
import time
import threading
import cv2
import numpy as np
from PIL import Image, ImageTk

# --- CONFIGURATION ---
BAUD_RATE = 115200       
DEFAULT_CAM_ID = 0
ID_SHOULDER_L = 12      
ALL_MOTORS = [11, 12, 14, 15, 16, 17] 
STEPS_PER_DEG = 4096.0 / 360.0

# Define the 3 Search Zones (x, y, w, h) based on 640x480 resolution
# Adjust these numbers to match your camera alignment!
ZONES = {
    1: (50,  200, 100, 100),  # Channel 1 Zone (Left)
    2: (270, 200, 100, 100),  # Channel 2 Zone (Center)
    3: (490, 200, 100, 100)   # Channel 3 Zone (Right)
}

class ColorSorterV3(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("BOSET Color Sorter V3 (Interactive Vision)")
        self.geometry("1300x850")
        ctk.set_appearance_mode("Dark")

        # State
        self.ser = None
        self.is_running = False
        self.joints = {mid: 90 for mid in ALL_MOTORS}
        self.sliders = {}
        self.waypoints = {} 
        
        # Color Learning State
        self.picking_mode = None  # Stores which bin (1, 2, or 3) we are teaching
        # Stores HSV ranges: {1: (lower, upper), 2: ...}
        self.color_defs = {1: None, 2: None, 3: None} 
        self.detected_in_zone = {1: "NONE", 2: "NONE", 3: "NONE"}

        self.setup_ui()
        self.refresh_ports()
        self.start_camera()

    
    def setup_ui(self):
        # === LEFT: INTERACTIVE VISION ===
        f_left = ctk.CTkFrame(self, width=660)
        f_left.pack(side="left", fill="y", padx=10, pady=10)
        
        ctk.CTkLabel(f_left, text="Interactive Vision Feed", font=("Arial", 18, "bold")).pack(pady=10)
        
        # Video Label with Click Event
        self.lbl_video = ctk.CTkLabel(f_left, text="Loading Camera...", width=640, height=480, fg_color="black")
        self.lbl_video.pack(pady=5)
        # Bind Left Click to the function
        self.lbl_video.bind("<Button-1>", self.on_video_click)
        
        ctk.CTkLabel(f_left, text="Click a 'Teach' button below, then click the object in the video.").pack(pady=5)

        # Color Teaching Buttons
        f_colors = ctk.CTkFrame(f_left)
        f_colors.pack(fill="x", pady=10)
        
        self.btn_c1 = ctk.CTkButton(f_colors, text="Teach Color 1 (Bin 1)", command=lambda: self.set_pick_mode(1), fg_color="gray")
        self.btn_c1.pack(side="left", padx=10, expand=True)
        
        self.btn_c2 = ctk.CTkButton(f_colors, text="Teach Color 2 (Bin 2)", command=lambda: self.set_pick_mode(2), fg_color="gray")
        self.btn_c2.pack(side="left", padx=10, expand=True)
        
        self.btn_c3 = ctk.CTkButton(f_colors, text="Teach Color 3 (Bin 3)", command=lambda: self.set_pick_mode(3), fg_color="gray")
        self.btn_c3.pack(side="left", padx=10, expand=True)

        # === RIGHT: CONTROL & LOGIC ===
        f_right = ctk.CTkFrame(self)
        f_right.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        # Connection
        f_conn = ctk.CTkFrame(f_right)
        f_conn.pack(fill="x", pady=5)
        self.cbox_ports = ctk.CTkComboBox(f_conn, values=["Scanning..."]); self.cbox_ports.pack(side="left", padx=5)
        self.btn_conn = ctk.CTkButton(f_conn, text="CONNECT", command=self.connect_serial, fg_color="green", width=80); self.btn_conn.pack(side="left", padx=5)

        # Sliders
        f_sliders = ctk.CTkFrame(f_right)
        f_sliders.pack(fill="x", pady=10)
        for mid in ALL_MOTORS:
            r = ctk.CTkFrame(f_sliders, fg_color="transparent")
            r.pack(fill="x", pady=1)
            ctk.CTkLabel(r, text=f"J{mid}", width=30).pack(side="left")
            s = ctk.CTkSlider(r, from_=0, to=180, command=lambda v, m=mid: self.move_motor(m, v))
            s.set(90); s.pack(side="left", fill="x", expand=True)
            self.sliders[mid] = s

        # Teaching Matrix
        ctk.CTkLabel(f_right, text="Teaching Matrix", font=("Arial", 16, "bold")).pack(pady=10)
        
        # Channel Inputs
        for i in range(1, 4):
            ctk.CTkButton(f_right, text=f"Set Pick Pose (Channel {i})", 
                          command=lambda x=i: self.save_waypoint(f"CHAN_{x}"), 
                          fg_color="#444").pack(fill="x", pady=2, padx=20)
        
        ctk.CTkLabel(f_right, text="--- Bins ---").pack(pady=5)
        
        # Bin Outputs
        for i in range(1, 4):
            ctk.CTkButton(f_right, text=f"Set Place Pose (Bin {i})", 
                          command=lambda x=i: self.save_waypoint(f"BIN_{x}"), 
                          fg_color="#666").pack(fill="x", pady=2, padx=20)

        # Auto Button
        self.txt_log = ctk.CTkTextbox(f_right, height=100)
        self.txt_log.pack(fill="x", side="bottom", padx=5, pady=5)
        
        self.btn_auto = ctk.CTkButton(f_right, text="START AUTO SORT", height=50, 
                                      fg_color="gray", state="disabled", command=self.toggle_auto)
        self.btn_auto.pack(fill="x", side="bottom", padx=20, pady=10)

    # --- VISION LOGIC ---
    def set_pick_mode(self, idx):
        self.picking_mode = idx
        self.log(f"CLICK ON THE VIDEO to set Color {idx}...")
        
        # Reset Button Colors
        self.btn_c1.configure(fg_color="gray")
        self.btn_c2.configure(fg_color="gray")
        self.btn_c3.configure(fg_color="gray")
        
        # Highlight active
        if idx == 1: self.btn_c1.configure(fg_color="yellow", text_color="black")
        if idx == 2: self.btn_c2.configure(fg_color="yellow", text_color="black")
        if idx == 3: self.btn_c3.configure(fg_color="yellow", text_color="black")

    def on_video_click(self, event):
        if self.picking_mode is None or self.current_frame is None: return
        
        # Get coordinates relative to the image
        x, y = event.x, event.y
        
        # Safety Check
        if 0 <= x < 640 and 0 <= y < 480:
            # Convert frame to HSV
            hsv_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2HSV)
            pixel = hsv_frame[y, x]
            
            # --- FIX STARTS HERE ---
            # 1. Cast to standard Python 'int' to allow negative math (prevents overflow)
            h = int(pixel[0])
            s = int(pixel[1])
            v = int(pixel[2])
            
            # 2. Define Range (+- 10 Hue, +- 50 Sat/Val)
            # 3. Explicitly set dtype="uint8" for OpenCV compatibility
            lower = np.array([max(0, h-10), max(50, s-50), max(50, v-50)], dtype="uint8")
            upper = np.array([min(180, h+10), 255, 255], dtype="uint8")
            # --- FIX ENDS HERE ---
            
            # Save
            self.color_defs[self.picking_mode] = (lower, upper)
            
            # Update UI Button Color to match
            rgb_pixel = self.current_frame[y, x]
            hex_color = '#%02x%02x%02x' % (rgb_pixel[2], rgb_pixel[1], rgb_pixel[0]) # BGR to Hex
            
            if self.picking_mode == 1: self.btn_c1.configure(fg_color=hex_color, text=f"Color 1 Set", text_color="white")
            if self.picking_mode == 2: self.btn_c2.configure(fg_color=hex_color, text=f"Color 2 Set", text_color="white")
            if self.picking_mode == 3: self.btn_c3.configure(fg_color=hex_color, text=f"Color 3 Set", text_color="white")
            
            self.log(f"Color {self.picking_mode} set! HSV: {h},{s},{v}")
            self.picking_mode = None # Exit mode

    def start_camera(self):
        self.cap = cv2.VideoCapture(DEFAULT_CAM_ID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        threading.Thread(target=self.cam_loop, daemon=True).start()

    def cam_loop(self):
        while True:
            ret, frame = self.cap.read()
            if not ret: continue
            
            # Flip for mirror effect if needed
            # frame = cv2.flip(frame, 1)
            self.current_frame = frame.copy()
            
            # --- PROCESS ZONES ---
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            for zone_id, (zx, zy, zw, zh) in ZONES.items():
                # Draw Zone Box
                cv2.rectangle(frame, (zx, zy), (zx+zw, zy+zh), (0, 255, 255), 2)
                cv2.putText(frame, f"CH {zone_id}", (zx, zy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Crop ROI
                roi = hsv[zy:zy+zh, zx:zx+zw]
                found_color = "NONE"
                max_roi_area = 0
                
                # Check against taught colors
                for color_id, bounds in self.color_defs.items():
                    if bounds is None: continue
                    
                    lower, upper = bounds
                    mask = cv2.inRange(roi, lower, upper)
                    
                    # Clean noise
                    mask = cv2.erode(mask, None, iterations=1)
                    mask = cv2.dilate(mask, None, iterations=1)
                    
                    # Count pixels
                    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        c = max(cnts, key=cv2.contourArea)
                        area = cv2.contourArea(c)
                        if area > 500: # Threshold
                            if area > max_roi_area:
                                max_roi_area = area
                                found_color = color_id
                                
                                # Visual feedback inside zone
                                x, y, w, h = cv2.boundingRect(c)
                                cv2.rectangle(frame, (zx+x, zy+y), (zx+x+w, zy+y+h), (0, 255, 0), 2)
                                cv2.putText(frame, f"C{color_id}", (zx+x, zy+y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                self.detected_in_zone[zone_id] = found_color

            # --- DISPLAY ---
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            ctk_img = ctk.CTkImage(img, size=(640, 480))
            
            # Update GUI safely
            try:
                self.lbl_video.configure(image=ctk_img, text="")
            except: pass
            
            time.sleep(0.03)

    # --- AUTO SORTING ---
    def toggle_auto(self):
        if self.is_running:
            self.is_running = False
            self.btn_auto.configure(text="START AUTO SORT", fg_color="#D97C23")
        else:
            self.is_running = True
            self.btn_auto.configure(text="STOP", fg_color="red")
            threading.Thread(target=self.sorting_loop, daemon=True).start()

    def sorting_loop(self):
        self.log("Auto Sort Started")
        self.execute_pose([90]*6, 2.0)
        
        while self.is_running:
            # Check Zone 1 -> 2 -> 3
            for i in range(1, 4):
                if not self.is_running: break
                
                detected_id = self.detected_in_zone[i]
                
                # If we see a taught color (1, 2, or 3)
                if detected_id != "NONE":
                    self.log(f"Zone {i}: Found Color {detected_id}")
                    
                    chan_key = f"CHAN_{i}"
                    bin_key = f"BIN_{detected_id}"
                    
                    if chan_key in self.waypoints and bin_key in self.waypoints:
                        # 1. Approach Channel
                        hover = self.waypoints[chan_key].copy()
                        hover[5] = 0 # Open
                        hover[1] -= 20 # Lift
                        self.execute_pose(hover, 1.5)
                        
                        # 2. Pick
                        pick = self.waypoints[chan_key].copy()
                        pick[5] = 0
                        self.execute_pose(pick, 1.0)
                        
                        pick[5] = 75 # Close
                        self.execute_pose(pick, 1.0)
                        
                        # 3. Lift
                        hover[5] = 75
                        self.execute_pose(hover, 1.0)
                        
                        # 4. Place in Bin
                        place = self.waypoints[bin_key].copy()
                        place[5] = 75
                        self.execute_pose(place, 3.0)
                        
                        place[5] = 0 # Open
                        self.execute_pose(place, 0.5)
                        
                        # 5. Home
                        self.execute_pose([90,145,12,90,7,0], 2.0)
                        time.sleep(1.0)
                    else:
                        self.log(f"Missing Waypoints for Ch{i} or Bin{detected_id}")
            
            time.sleep(0.5)

    # --- HELPERS (Same as before) ---
    def log(self, msg):
        self.txt_log.insert("end", f"> {msg}\n")
        self.txt_log.see("end")

    def refresh_ports(self):
        """Scans for available serial ports and updates the dropdown."""
        ports = serial.tools.list_ports.comports()
        port_list = [p.device for p in ports]
        if not port_list:
            port_list = ["No Ports Found"]
        self.cbox_ports.configure(values=port_list)
        # Select the first port automatically if available
        if len(port_list) > 0 and port_list[0] != "No Ports Found":
            self.cbox_ports.set(port_list[0])

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

    def move_motor(self, mid, val):
        self.joints[mid] = int(val)
        if self.ser and self.ser.is_open:
            self.send_packet(mid, int(val))

    def send_packet(self, mid, angle):
        steps = int(angle * STEPS_PER_DEG)
        steps = max(0, min(4095, steps))
        sl, sh = 600 & 0xFF, (600 >> 8) & 0xFF
        pl, ph = steps & 0xFF, (steps >> 8) & 0xFF
        cksm = (~(mid + 9 + 3 + 42 + pl + ph + 0 + 0 + sl + sh)) & 0xFF
        msg = bytearray([0xFF, 0xFF, mid, 9, 3, 42, pl, ph, 0, 0, sl, sh, cksm])
        try: 
            self.ser.write(msg)
            if mid == ID_SHOULDER_L: 
                inv = 4096 - steps
                pl2, ph2 = inv & 0xFF, (inv >> 8) & 0xFF
                cksm2 = (~(13 + 9 + 3 + 42 + pl2 + ph2 + 0 + 0 + sl + sh)) & 0xFF
                self.ser.write(bytearray([0xFF, 0xFF, 13, 9, 3, 42, pl2, ph2, 0, 0, sl, sh, cksm2]))
        except: pass

    def save_waypoint(self, name):
        self.waypoints[name] = [int(self.sliders[m].get()) for m in ALL_MOTORS]
        self.log(f"Saved: {name}")

    def execute_pose(self, pose, delay):
        if not self.is_running: return
        for i, mid in enumerate(ALL_MOTORS):
            self.move_motor(mid, pose[i])
            self.sliders[mid].set(pose[i])
        time.sleep(delay)

if __name__ == "__main__":
    app = ColorSorterV3()
    app.mainloop()