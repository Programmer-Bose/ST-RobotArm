import customtkinter as ctk
import serial
import serial.tools.list_ports
import csv
import cv2
import threading
import math
import numpy as np
from PIL import Image

# --- CONFIGURATION ---
BAUD_RATE = 115200
DEFAULT_CAM_ID = 0  
ALL_MOTORS = [11, 12, 14, 15, 16, 17]
STEPS_PER_DEG = 4096.0 / 360.0
CSV_FILE = "ik_data_orientation.csv" # New file for V2

# HSV Range for RED Object (Adjust if needed)
RED_LOWER1 = np.array([0, 120, 70])
RED_UPPER1 = np.array([10, 255, 255])
RED_LOWER2 = np.array([170, 120, 70])
RED_UPPER2 = np.array([180, 255, 255])

class IKDataCollectorV2(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("ML Data Collector V2 (With Orientation)")
        self.geometry("1100x700")
        
        self.ser = None
        self.sliders = {}
        self.joints = {m: 90 for m in ALL_MOTORS}
        self.current_frame = None
        self.detected_info = None # Stores (x, y, angle)
        
        # Init CSV with NEW Column: object_angle
        with open(CSV_FILE, 'w', newline='') as f:
            csv.writer(f).writerow(["pixel_x", "pixel_y", "obj_angle", "j11", "j12", "j14", "j15", "j16", "j17"])

        self.setup_ui()
        self.start_camera()
        self.refresh_ports()

    def setup_ui(self):
        # Camera Feed
        self.lbl_video = ctk.CTkLabel(self, text="Camera Loading...", width=640, height=480)
        self.lbl_video.pack(side="left", padx=10, pady=10)
        self.lbl_video.bind("<Button-1>", self.on_click) 

        # Controls
        f_ctrl = ctk.CTkFrame(self)
        f_ctrl.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.cbox_ports = ctk.CTkComboBox(f_ctrl, values=["Scanning..."])
        self.cbox_ports.pack(pady=5)
        ctk.CTkButton(f_ctrl, text="CONNECT", command=self.connect, fg_color="green").pack(pady=5)

        # Sliders
        for mid in ALL_MOTORS:
            f = ctk.CTkFrame(f_ctrl, fg_color="transparent")
            f.pack(fill="x", pady=2)
            ctk.CTkLabel(f, text=f"J{mid}", width=30).pack(side="left")
            s = ctk.CTkSlider(f, from_=0, to=180, command=lambda v, m=mid: self.move(m, v))
            s.set(90); s.pack(side="left", fill="x")
            self.sliders[mid] = s

        self.lbl_log = ctk.CTkLabel(f_ctrl, text="Align Gripper manually -> Click Object to Save", text_color="yellow")
        self.lbl_log.pack(pady=20)

    def on_click(self, event):
        if self.detected_info is None:
            print("No object detected! Cannot save.")
            return
            
        # Use detected data, NOT click coordinates (for precision)
        x, y, angle = self.detected_info
        
        # Save Data
        angles = [int(self.sliders[m].get()) for m in ALL_MOTORS]
        
        # Row: [Input X, Input Y, Input Angle, Output J11...J17]
        row = [x, y, angle] + angles
        
        with open(CSV_FILE, 'a', newline='') as f:
            csv.writer(f).writerow(row)
            
        self.lbl_log.configure(text=f"Saved: Pos({x},{y}) Ang({angle:.1f}) -> J16({angles[4]})")
        print(f"Saved Row: {row}")

    # --- VISION LOGIC ---
    def start_camera(self):
        self.cap = cv2.VideoCapture(DEFAULT_CAM_ID)
        threading.Thread(target=self.cam_loop, daemon=True).start()

    def cam_loop(self):
        while True:
            ret, frame = self.cap.read()
            if not ret: continue
            
            # 1. Color Detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.bitwise_or(cv2.inRange(hsv, RED_LOWER1, RED_UPPER1), 
                                  cv2.inRange(hsv, RED_LOWER2, RED_UPPER2))
            
            # 2. Find Contours
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.detected_info = None # Reset
            
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                if cv2.contourArea(c) > 500:
                    # 3. Get Oriented Bounding Box
                    rect = cv2.minAreaRect(c)
                    (cx, cy), (w, h), angle = rect
                    
                    # Normalize Angle (-90 to 90 standard)
                    if w < h:
                        angle = angle + 90
                        
                    self.detected_info = (int(cx), int(cy), angle)

                    # 4. Visualization
                    box = cv2.boxPoints(rect)
                    # --- FIX IS HERE ---
                    box = np.int32(box) # Replaced np.int0 with np.int32
                    # -------------------
                    cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
                    
                    # Draw text
                    cv2.putText(frame, f"Ang: {int(angle)}", (int(cx), int(cy)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            self.current_frame = frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ctk.CTkImage(Image.fromarray(rgb), size=(640, 480))
            try: self.lbl_video.configure(image=img, text="")
            except: pass
    # --- SERIAL HELPERS ---
    def connect(self):
        try:
            self.ser = serial.Serial(self.cbox_ports.get(), BAUD_RATE)
            self.ser.dtr = False; self.ser.rts = False
        except: pass
    
    def refresh_ports(self):
        p = [x.device for x in serial.tools.list_ports.comports()]
        self.cbox_ports.configure(values=p if p else ["No Ports"])

    def move(self, mid, val):
        val = int(val)
        self.joints[mid] = val
        if self.ser and self.ser.is_open:
            steps = int(val * STEPS_PER_DEG)
            msg = self.make_packet(mid, steps)
            self.ser.write(msg)
            if mid == 12: self.ser.write(self.make_packet(13, 4096-steps))

    def make_packet(self, mid, steps):
        steps = max(0, min(4095, steps))
        cksm = (~(mid + 54 + (steps&0xFF) + (steps>>8))) & 0xFF
        return bytearray([0xFF, 0xFF, mid, 9, 3, 42, steps&0xFF, steps>>8, 0, 0, 88, 2, cksm])

if __name__ == "__main__":
    IKDataCollectorV2().mainloop()