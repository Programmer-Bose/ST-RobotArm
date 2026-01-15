import customtkinter as ctk
import cv2
import mediapipe as mp
import websocket
import threading
import math
from PIL import Image, ImageTk

# Configuration
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class MecanumGestureController(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window Setup
        self.title("Mecanum Gesture Controller")
        self.geometry("1000x750")
        
        # Variables
        self.ws = None
        self.is_connected = False
        self.last_cmd = "S"
        self.cap = cv2.VideoCapture(0)
        
        # MediaPipe Setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # --- LAYOUT ---
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # 1. Top Bar (Connection Settings)
        self.top_frame = ctk.CTkFrame(self)
        self.top_frame.grid(row=0, column=0, padx=20, pady=10, sticky="ew")

        self.lbl_ip = ctk.CTkLabel(self.top_frame, text="IP Address:")
        self.lbl_ip.pack(side="left", padx=10)
        
        self.entry_ip = ctk.CTkEntry(self.top_frame, width=150)
        self.entry_ip.insert(0, "192.168.1.168")
        self.entry_ip.pack(side="left", padx=5)

        self.lbl_port = ctk.CTkLabel(self.top_frame, text="Port:")
        self.lbl_port.pack(side="left", padx=10)
        
        self.entry_port = ctk.CTkEntry(self.top_frame, width=60)
        self.entry_port.insert(0, "81")
        self.entry_port.pack(side="left", padx=5)

        self.btn_connect = ctk.CTkButton(self.top_frame, text="Connect", command=self.toggle_connection, fg_color="green")
        self.btn_connect.pack(side="left", padx=20)

        self.lbl_status = ctk.CTkLabel(self.top_frame, text="Disconnected", text_color="red", font=("Arial", 14, "bold"))
        self.lbl_status.pack(side="right", padx=20)

        # 2. Main Video Feed Area
        self.video_frame = ctk.CTkFrame(self)
        self.video_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        
        self.lbl_video = ctk.CTkLabel(self.video_frame, text="")
        self.lbl_video.pack(expand=True, fill="both", padx=10, pady=10)

        # 3. Bottom Status Bar
        self.bottom_frame = ctk.CTkFrame(self, height=50)
        self.bottom_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        
        self.lbl_cmd = ctk.CTkLabel(self.bottom_frame, text="CMD: STOP", font=("Arial", 20, "bold"))
        self.lbl_cmd.pack(pady=10)

        # Start Video Loop
        self.update_video_feed()

    def toggle_connection(self):
        if not self.is_connected:
            # Connect
            ip = self.entry_ip.get()
            port = self.entry_port.get()
            url = f"ws://{ip}:{port}/"
            
            try:
                self.ws = websocket.create_connection(url, timeout=2)
                self.is_connected = True
                self.btn_connect.configure(text="Disconnect", fg_color="red")
                self.lbl_status.configure(text="CONNECTED", text_color="green")
                self.entry_ip.configure(state="disabled")
                self.entry_port.configure(state="disabled")
            except Exception as e:
                self.lbl_status.configure(text=f"Error: {e}", text_color="orange")
        else:
            # Disconnect
            if self.ws:
                self.ws.close()
            self.is_connected = False
            self.btn_connect.configure(text="Connect", fg_color="green")
            self.lbl_status.configure(text="Disconnected", text_color="red")
            self.entry_ip.configure(state="normal")
            self.entry_port.configure(state="normal")

    def send_command(self, cmd):
        # SECURITY CHECK: Only send if connected
        if self.is_connected and self.ws:
            if cmd != self.last_cmd:
                try:
                    self.ws.send(cmd)
                    print(f"Sent: {cmd}")
                    self.last_cmd = cmd
                except Exception as e:
                    print(f"Send Error: {e}")
                    self.toggle_connection() # Auto disconnect on error

    def check_fist(self, landmarks):
        wrist = landmarks[0]
        fingers_folded = 0
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        
        for tip, pip in zip(tips, pips):
            dist_tip = math.hypot(landmarks[tip].x - wrist.x, landmarks[tip].y - wrist.y)
            dist_pip = math.hypot(landmarks[pip].x - wrist.x, landmarks[pip].y - wrist.y)
            if dist_tip < dist_pip:
                fingers_folded += 1
        return fingers_folded >= 3

    def get_hand_tilt(self, landmarks):
        """Calculates tilt relative to 90 degrees (Straight Up)"""
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        
        # Calculate angle (inverted Y because screen coordinates go down)
        delta_x = middle_mcp.x - wrist.x
        delta_y = wrist.y - middle_mcp.y  # Positive Y is UP
        
        # Atan2 returns radians, convert to degrees
        angle_rad = math.atan2(delta_y, delta_x)
        angle_deg = math.degrees(angle_rad)
        
        # Calculate Difference from 90
        # Positive = Left Tilt
        # Negative = Right Tilt
        tilt = angle_deg - 90
        return tilt

    def get_grid_command(self, x, y, tilt):
        # 3x3 Grid
        if x < 0.33: col = 0
        elif x < 0.66: col = 1
        else: col = 2
        
        if y < 0.33: row = 0
        elif y < 0.66: row = 1
        else: row = 2

        # Grid Map
        grid = [
            ["DFL", "F", "DFR"],
            ["SL",  "CENTER", "SR" ],
            ["DBL", "B", "DBR"]
        ]
        
        cmd = grid[row][col]
        
        # Special Logic for CENTER region: Check Tilt Difference
        if cmd == "CENTER":
            if tilt > 15: # Tilted Left > 15 deg from straight
                return "L"
            elif tilt < -15: # Tilted Right > 15 deg from straight
                return "R"
            else:
                return "S"
        
        return cmd

    def update_video_feed(self):
        ret, frame = self.cap.read()
        if ret:
            # 1. Pre-processing
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 2. MediaPipe Detection
            results = self.hands.process(rgb_frame)
            current_cmd = "S"
            is_fist = False
            tilt_val = 0
            
            # 3. Logic
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                    
                    cx, cy = hand_lms.landmark[9].x, hand_lms.landmark[9].y
                    is_fist = self.check_fist(hand_lms.landmark)
                    tilt_val = self.get_hand_tilt(hand_lms.landmark)
                    
                    if is_fist:
                        current_cmd = "S"
                    else:
                        current_cmd = self.get_grid_command(cx, cy, tilt_val)

                    # Highlight Hand Center
                    color = (0, 0, 255) if is_fist else (0, 255, 0)
                    cv2.circle(frame, (int(cx*w), int(cy*h)), 15, color, -1)
                    
                    # Display Tilt Difference for debugging
                    cv2.putText(frame, f"Tilt: {int(tilt_val)}", (int(cx*w)+20, int(cy*h)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # 4. Draw Grid Overlay
            grid_color = (0, 0, 255) if is_fist else (255, 255, 255)
            cv2.line(frame, (int(w*0.33), 0), (int(w*0.33), h), grid_color, 2)
            cv2.line(frame, (int(w*0.66), 0), (int(w*0.66), h), grid_color, 2)
            cv2.line(frame, (0, int(h*0.33)), (w, int(h*0.33)), grid_color, 2)
            cv2.line(frame, (0, int(h*0.66)), (w, int(h*0.66)), grid_color, 2)

            # Add Text Labels
            labels = [["DFL", "FWD", "DFR"], ["LEFT", "ROTATE", "RIGHT"], ["DBL", "BACK", "DBR"]]
            for r in range(3):
                for c in range(3):
                    text_x = int(c * (w/3) + (w/3)/2 - 30)
                    text_y = int(r * (h/3) + (h/3)/2)
                    cv2.putText(frame, labels[r][c], (text_x, text_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

            # 5. Send Command (Only if Connected)
            if self.is_connected:
                self.send_command(current_cmd)
                status_text = f"CMD: {current_cmd} {'(SAFETY STOP)' if is_fist else ''}"
                color_text = "red" if is_fist else "green"
            else:
                status_text = "CMD: S (Offline)"
                color_text = "gray"

            self.lbl_cmd.configure(text=status_text, text_color=color_text)

            # 6. Convert to TK Image
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Resize logic to fit window
            img_width, img_height = img.size
            ratio = min(800/img_width, 600/img_height) # fit inside frame
            new_size = (int(img_width*ratio), int(img_height*ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            tk_img = ctk.CTkImage(light_image=img, dark_image=img, size=new_size)
            self.lbl_video.configure(image=tk_img)
            self.lbl_video.image = tk_img # Keep reference

        # Loop
        self.after(20, self.update_video_feed)

    def close(self):
        if self.ws:
            self.ws.close()
        self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = MecanumGestureController()
    app.protocol("WM_DELETE_WINDOW", app.close)
    app.mainloop()