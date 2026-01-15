import customtkinter as ctk
import serial
import serial.tools.list_ports
import csv
import cv2
import threading
import numpy as np
from PIL import Image

# --- CONFIGURATION ---
BAUD_RATE = 115200
DEFAULT_CAM_ID = 0  # Change if needed
ALL_MOTORS = [11, 12, 14, 15, 16, 17]
STEPS_PER_DEG = 4096.0 / 360.0
CSV_FILE = "ik_data.csv"

class IKDataCollector(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("ML-IK Data Collector")
        self.geometry("1100x700")
        
        self.ser = None
        self.sliders = {}
        self.joints = {m: 90 for m in ALL_MOTORS}
        self.current_frame = None
        
        # Init CSV
        with open(CSV_FILE, 'w', newline='') as f:
            csv.writer(f).writerow(["pixel_x", "pixel_y", "j11", "j12", "j14", "j15", "j16", "j17"])

        self.setup_ui()
        self.start_camera()
        self.refresh_ports()

    def setup_ui(self):
        # Camera Feed (Left)
        self.lbl_video = ctk.CTkLabel(self, text="Camera Loading...", width=640, height=480)
        self.lbl_video.pack(side="left", padx=10, pady=10)
        self.lbl_video.bind("<Button-1>", self.on_click) # CLICK TO SAVE

        # Controls (Right)
        f_ctrl = ctk.CTkFrame(self)
        f_ctrl.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # Connection
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

        self.lbl_log = ctk.CTkLabel(f_ctrl, text="Ready. Click object to save.", text_color="yellow")
        self.lbl_log.pack(pady=20)

    def on_click(self, event):
        if self.current_frame is None: return
        x, y = event.x, event.y
        
        # Save Data
        angles = [int(self.sliders[m].get()) for m in ALL_MOTORS]
        row = [x, y] + angles
        
        with open(CSV_FILE, 'a', newline='') as f:
            csv.writer(f).writerow(row)
            
        # Draw feedback
        cv2.circle(self.current_frame, (x, y), 5, (0, 255, 0), -1)
        self.lbl_log.configure(text=f"Saved: Pixel({x},{y}) -> Joints{angles}")
        print(f"Saved: {row}")

    # --- STANDARD HELPERS ---
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

    def start_camera(self):
        self.cap = cv2.VideoCapture(DEFAULT_CAM_ID)
        threading.Thread(target=self.cam_loop, daemon=True).start()

    def cam_loop(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = ctk.CTkImage(Image.fromarray(rgb), size=(640, 480))
                try: self.lbl_video.configure(image=img, text="")
                except: pass

if __name__ == "__main__":
    IKDataCollector().mainloop()