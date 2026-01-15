import customtkinter as ctk
import serial
import serial.tools.list_ports
import time
import threading
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

# --- CONFIGURATION ---
MODEL_PATH = "robot_model_v1.pth" 
BAUD_RATE = 115200
DEFAULT_CAM_ID = 1  
DEFAULT_SPEED = 600

# Device Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running Inference on: {DEVICE}")

# Robot IDs
ID_BASE = 11
ID_SHOULDER_L = 12
ID_ELBOW = 14
ID_PITCH = 15
ID_ROLL = 16
ID_GRIPPER = 17
ALL_MOTORS = [ID_BASE, ID_SHOULDER_L, ID_ELBOW, ID_PITCH, ID_ROLL, ID_GRIPPER]
STEPS_PER_DEG = 4096.0 / 360.0

# --- 1. MODEL DEFINITION ---
class RobotPolicyNetwork(nn.Module):
    def __init__(self):
        super(RobotPolicyNetwork, self).__init__()
        self.resnet = models.resnet18(weights=None) 
        self.resnet.fc = nn.Identity() 
        self.joint_encoder = nn.Linear(6, 64) 
        self.action_head = nn.Sequential(
            nn.Linear(576, 256),
            nn.ReLU(),
            nn.Dropout(0.2),    
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 6) 
        )
        
    def forward(self, img, joints):
        x_img = self.resnet(img)
        x_joints = torch.relu(self.joint_encoder(joints))
        x_cat = torch.cat((x_img, x_joints), dim=1)
        return self.action_head(x_cat)

# --- 2. INFERENCE GUI APPLICATION ---
class InferenceController(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI Inference Controller V3 (Fixed UI)")
        self.geometry("1200x850")
        ctk.set_appearance_mode("Dark")
        
        # State
        self.ser = None
        self.current_frame = None
        self.display_frame = None
        self.lock = threading.Lock()
        self.live_monitoring = False 
        
        self.current_joints = {mid: 90 for mid in ALL_MOTORS}
        self.predicted_targets = {mid: 90 for mid in ALL_MOTORS}
        
        # GUI Elements Storage
        self.cbox_ports = None # Init to None for safety
        self.labels_curr = {}
        self.labels_delta = {}
        self.labels_target = {}
        self.lbl_raw_monitor = {} 
        
        # Load Model
        self.load_ai_model()
        
        # UI Setup
        self.create_ui()
        self.refresh_ports()
        
        # Start Systems
        self.start_camera_thread()
        self.update_gui_image()

    def load_ai_model(self):
        try:
            self.model = RobotPolicyNetwork().to(DEVICE)
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
            self.model.eval()
            
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            print("Model Loaded Successfully!")
        except Exception as e:
            print(f"CRITICAL ERROR LOADING MODEL: {e}")
            self.model = None

    def create_ui(self):
        # === LEFT: VISION ===
        f_view = ctk.CTkFrame(self, width=400)
        f_view.pack(side="left", fill="y", padx=10, pady=10)
        
        ctk.CTkLabel(f_view, text="ROBOT VISION", font=("Arial", 14, "bold")).pack(pady=5)
        self.lbl_cam = ctk.CTkLabel(f_view, text="Waiting...", width=320, height=240, fg_color="#111")
        self.lbl_cam.pack(pady=10)
        
        self.lbl_status = ctk.CTkLabel(f_view, text="Status: Idle", text_color="gray", font=("Arial", 16))
        self.lbl_status.pack(pady=20)
        
        # Log Box
        self.txt_log = ctk.CTkTextbox(f_view, height=200)
        self.txt_log.pack(fill="x", pady=10)

        # === RIGHT: CONTROLS ===
        # (This block was incorrectly indented in the previous version)
        f_ctrl = ctk.CTkFrame(self)
        f_ctrl.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        # 1. Connection
        f_conn = ctk.CTkFrame(f_ctrl)
        f_conn.pack(fill="x", pady=5)
        self.cbox_ports = ctk.CTkComboBox(f_conn, width=120); self.cbox_ports.pack(side="left", padx=5)
        self.btn_connect = ctk.CTkButton(f_conn, text="CONNECT", command=self.run_connect); self.btn_connect.pack(side="left", padx=5)
        
        # 2. Data Monitor Table
        f_table = ctk.CTkFrame(f_ctrl)
        f_table.pack(fill="both", expand=True, pady=10)
        
        # Headers
        h_frame = ctk.CTkFrame(f_table, fg_color="transparent")
        h_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(h_frame, text="JOINT", width=50, font=("Arial", 11, "bold")).pack(side="left")
        ctk.CTkLabel(h_frame, text="RAW", width=80, font=("Arial", 11, "bold")).pack(side="left")
        ctk.CTkLabel(h_frame, text="CURR(Deg)", width=80, font=("Arial", 11, "bold")).pack(side="left")
        ctk.CTkLabel(h_frame, text="DELTA", width=80, font=("Arial", 11, "bold")).pack(side="left")
        ctk.CTkLabel(h_frame, text="TARGET", width=80, font=("Arial", 11, "bold")).pack(side="left")

        joint_names = ["Base", "Shldr", "Elbow", "Pitch", "Roll", "Grip"]
        for i, mid in enumerate(ALL_MOTORS):
            row = ctk.CTkFrame(f_table, fg_color="#2A2A2A")
            row.pack(fill="x", pady=2, padx=5)
            
            ctk.CTkLabel(row, text=joint_names[i], width=50).pack(side="left")
            
            # Raw Data Monitor
            l_raw = ctk.CTkLabel(row, text="----", width=80, text_color="#AAA")
            l_raw.pack(side="left")
            self.lbl_raw_monitor[mid] = l_raw
            
            # Current Angle
            l_curr = ctk.CTkLabel(row, text="--", width=80, text_color="cyan")
            l_curr.pack(side="left")
            self.labels_curr[mid] = l_curr
            
            # Delta
            l_delta = ctk.CTkLabel(row, text="--", width=80, text_color="yellow")
            l_delta.pack(side="left")
            self.labels_delta[mid] = l_delta
            
            # Target
            l_targ = ctk.CTkLabel(row, text="--", width=80, text_color="orange")
            l_targ.pack(side="left")
            self.labels_target[mid] = l_targ

        # 3. Buttons
        f_btns = ctk.CTkFrame(f_ctrl, fg_color="transparent")
        f_btns.pack(fill="x", pady=20)
        self.btn_predict = ctk.CTkButton(f_btns, text="🧠 PREDICT MOVE", height=50, 
                                         fg_color="#1F6AA5", command=self.run_inference)
        self.btn_predict.pack(fill="x", pady=5)
        self.btn_execute = ctk.CTkButton(f_btns, text="✅ EXECUTE MOVE", height=50, state="disabled",
                                         fg_color="green", command=self.execute_move)
        self.btn_execute.pack(fill="x", pady=5)

    def log(self, msg):
        self.txt_log.insert("end", f"{msg}\n")
        self.txt_log.see("end")

    # --- LIVE MONITORING THREAD ---
    def start_monitoring_thread(self):
        self.live_monitoring = True
        threading.Thread(target=self.monitor_loop, daemon=True).start()
        
    def monitor_loop(self):
        while self.live_monitoring and self.ser and self.ser.is_open:
            for mid in ALL_MOTORS:
                raw = self.read_motor_pos_safe(mid)
                if raw is not None:
                    deg = int(raw / STEPS_PER_DEG)
                    self.current_joints[mid] = deg
                    self.lbl_raw_monitor[mid].configure(text=str(raw))
                    self.labels_curr[mid].configure(text=str(deg))
                time.sleep(0.01)
            time.sleep(0.1) 

    # --- INFERENCE ---
    def run_inference(self):
        if self.model is None or self.current_frame is None:
            self.log("Error: Model or Camera missing")
            return

        with self.lock:
            img_pil = Image.fromarray(cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB))
        input_img = self.transform(img_pil).unsqueeze(0).to(DEVICE)
        
        raw_joints = [self.current_joints[mid] for mid in ALL_MOTORS]
        norm_joints = [j / 180.0 for j in raw_joints]
        input_joints = torch.tensor([norm_joints], dtype=torch.float32).to(DEVICE)
        
        with torch.no_grad():
            output_deltas = self.model(input_img, input_joints).cpu().numpy()[0]
            
        self.log("Proposal Generated.")
        
        for i, mid in enumerate(ALL_MOTORS):
            norm_d = output_deltas[i]
            raw_delta = (norm_d * 360.0) - 180.0
            
            current_val = self.current_joints[mid]
            target_val = current_val + raw_delta
            target_val = max(0, min(180, int(target_val)))
            
            self.predicted_targets[mid] = target_val
            self.labels_delta[mid].configure(text=f"{raw_delta:+.1f}")
            self.labels_target[mid].configure(text=str(target_val))

        self.btn_execute.configure(state="normal")

    def execute_move(self):
        self.log("Executing Move...")
        self.btn_execute.configure(state="disabled")
        
        for mid in ALL_MOTORS:
            target = self.predicted_targets[mid]
            self.send_packet(mid, target)
            self.labels_delta[mid].configure(text="--")
            self.labels_target[mid].configure(text="--")
        self.log("Done.")

    # --- CONNECTION ---
    def refresh_ports(self):
        ports = serial.tools.list_ports.comports()
        if self.cbox_ports:
            self.cbox_ports.configure(values=[p.device for p in ports] or ["No Ports"])

    def run_connect(self):
        p = self.cbox_ports.get()
        if p == "No Ports": return
        threading.Thread(target=self.connect_thread, args=(p,)).start()

    def connect_thread(self, port):
        self.btn_connect.configure(state="disabled", text="Connecting...")
        try:
            self.ser = serial.Serial()
            self.ser.port = port
            self.ser.baudrate = BAUD_RATE
            self.ser.dtr = False
            self.ser.rts = False
            self.ser.open()
            
            self.log("Port Opened. Waiting for board...")
            time.sleep(2.0)
            self.ser.reset_input_buffer()
            
            self.btn_connect.configure(text="CONNECTED", fg_color="green")
            self.lbl_status.configure(text="Online", text_color="green")
            
            for mid in ALL_MOTORS:
                cksm = (~(mid + 48)) & 0xFF 
                self.ser.write(bytearray([0xFF, 0xFF, mid, 4, 3, 40, 1, cksm]))
                time.sleep(0.02)
                
            self.start_monitoring_thread()
            self.log("Live Monitoring Started.")

        except Exception as e:
            self.btn_connect.configure(state="normal", text="CONN")
            self.log(f"Connection Error: {e}")

    # --- HARDWARE IO ---
    def read_motor_pos_safe(self, mid):
        cksm = (~(mid + 64)) & 0xFF
        try:
            self.ser.reset_input_buffer()
            self.ser.write(bytearray([0xFF, 0xFF, mid, 4, 2, 56, 2, cksm]))
            t = time.time()
            while time.time() - t < 0.05:
                if self.ser.in_waiting >= 8:
                    if self.ser.read(1) == b'\xff':
                        pkt = self.ser.read(7)
                        if pkt[1] == mid: 
                            raw_val = pkt[4] + (pkt[5] << 8)
                            if raw_val > 4095: return None
                            return raw_val
        except: pass
        return None

    def send_packet(self, mid, deg):
        steps = int(deg * STEPS_PER_DEG)
        steps = max(0, min(4095, steps))
        if mid == ID_SHOULDER_L:
            self._write_pos(ID_SHOULDER_L, steps)
            self._write_pos(13, 4096 - steps)
        else:
            self._write_pos(mid, steps)

    def _write_pos(self, mid, steps):
        spd = DEFAULT_SPEED
        sl, sh = spd & 0xFF, (spd >> 8) & 0xFF
        pl, ph = steps & 0xFF, (steps >> 8) & 0xFF
        cksm = (~(mid + 54 + pl + ph + sl + sh)) & 0xFF
        try:
            self.ser.write(bytearray([0xFF, 0xFF, mid, 9, 3, 42, pl, ph, 0, 0, sl, sh, cksm]))
        except: pass

    # --- CAMERA ---
    def start_camera_thread(self):
        self.cap = cv2.VideoCapture(DEFAULT_CAM_ID, cv2.CAP_DSHOW)
        self.cap.set(3, 640); self.cap.set(4, 480)
        threading.Thread(target=self.camera_loop, daemon=True).start()

    def camera_loop(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.current_frame = frame
                    self.display_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            time.sleep(0.03)

    def update_gui_image(self):
        with self.lock:
            if self.display_frame:
                img = ctk.CTkImage(self.display_frame, size=(320, 240))
                self.lbl_cam.configure(image=img, text="")
        self.after(30, self.update_gui_image)

if __name__ == "__main__":
    app = InferenceController()
    app.mainloop()