import serial
import time
import sys

# --- CONFIGURATION ---
SERIAL_PORT = 'COM3'    # Change to your port
BAUD_RATE = 115200      # Match your ESP32 setup
ID_LEFT = 12
ID_RIGHT = 13
STEPS_PER_DEGREE = 4096 / 360.0

def init_serial():
    """Opens serial port safely to prevent ESP32 reset."""
    ser = serial.Serial()
    ser.port = SERIAL_PORT
    ser.baudrate = BAUD_RATE
    ser.timeout = 0.1
    ser.dtr = False
    ser.rts = False
    ser.open()
    ser.reset_input_buffer()
    return ser

def get_current_position(ser, servo_id):
    """
    Reads current position. Returns steps (0-4096).
    """
    # Packet: [FF, FF, ID, Len, Instr(Read), Addr(56), Len(2), Cksm]
    checksum = (~(servo_id + 4 + 2 + 56 + 2)) & 0xFF
    cmd = bytearray([0xFF, 0xFF, servo_id, 4, 2, 56, 2, checksum])
    
    ser.reset_input_buffer()
    ser.write(cmd)
    
    start = time.time()
    while (time.time() - start) < 0.2:
        if ser.in_waiting >= 8:
            # Simple header check
            header = ser.read(2)
            if header == b'\xff\xff':
                packet = ser.read(6) # Read rest
                # ID is at index 0 of the rest
                if packet[0] == servo_id:
                    # Pos is at index 3 (Low) and 4 (High) of the 'rest'
                    pos = packet[3] + (packet[4] << 8)
                    return pos
    return None # Failed to read

def move_servo(ser, servo_id, steps, time_ms=0, speed=0):
    """
    Moves servo. 
    If time_ms > 0, the servo calculates speed to arrive exactly then.
    Otherwise uses 'speed' (steps/sec).
    """
    if steps < 0: steps = 0
    if steps > 4095: steps = 4095
    
    p_low = int(steps) & 0xFF
    p_high = (int(steps) >> 8) & 0xFF
    
    # We can use the Time/Speed registers. 
    # Addr 42 = Goal Pos. Addr 44 = Goal Time. Addr 46 = Goal Speed.
    # We will write 6 bytes starting at 42.
    
    t_low = time_ms & 0xFF
    t_high = (time_ms >> 8) & 0xFF
    
    s_low = speed & 0xFF
    s_high = (speed >> 8) & 0xFF
    
    # Length = 9 (Instr(1)+Addr(1)+Data(6)+Cksm(1))
    checksum_sum = servo_id + 9 + 3 + 42 + p_low + p_high + t_low + t_high + s_low + s_high
    checksum = (~checksum_sum) & 0xFF
    
    cmd = bytearray([0xFF, 0xFF, servo_id, 9, 3, 42, 
                     p_low, p_high, t_low, t_high, s_low, s_high, checksum])
    ser.write(cmd)

def main():
    ser = None
    try:
        ser = init_serial()
        print(f"Connected to {SERIAL_PORT}.")
        
        # --- 1. GET USER INPUTS ---
        try:
            min_deg = float(input("Enter MIN Angle (e.g., 0): "))
            max_deg = float(input("Enter MAX Angle (e.g., 180): "))
            wait_time = float(input("Enter Delay between sweeps (seconds): "))
        except ValueError:
            print("Invalid input.")
            return

        # --- 2. SOFT START ---
        print("\n[Soft Start] Reading current positions...")
        
        # We read ID_LEFT to know where we are starting
        current_steps = get_current_position(ser, ID_LEFT)
        
        if current_steps is None:
            print("Warning: Could not read position. Assuming 0.")
            current_steps = 0
        
        print(f"Current Pos (Left): {current_steps} steps")
        
        target_start_steps = int(min_deg * STEPS_PER_DEGREE)
        
        # Move gently to start position (take 2 seconds to get there)
        print("Moving slowly to Start Position...")
        
        # Left moves to Target
        move_servo(ser, ID_LEFT, target_start_steps, time_ms=2000)
        
        # Right moves to Mirror Target (4096 - Target)
        move_servo(ser, ID_RIGHT, 4096 - target_start_steps, time_ms=2000)
        
        time.sleep(2.5) # Wait for move to finish

        # --- 3. CONTINUOUS SWEEP ---
        print(f"\n[Running] Sweeping {min_deg}° <-> {max_deg}°... (Ctrl+C to stop)")
        
        min_steps = int(min_deg * STEPS_PER_DEGREE)
        max_steps = int(max_deg * STEPS_PER_DEGREE)
        
        # Speed for the sweep (steps per second)
        SWEEP_SPEED = 700 
        
        while True:
            # --- SWEEP UP ---
            print(f" -> Moving to {max_deg}°")
            move_servo(ser, ID_LEFT, max_steps, speed=SWEEP_SPEED)
            move_servo(ser, ID_RIGHT, 4096 - max_steps, speed=SWEEP_SPEED)
            
            # Wait for movement + User Delay
            time.sleep(1.5 + wait_time) 

            # --- SWEEP DOWN ---
            print(f" <- Moving to {min_deg}°")
            move_servo(ser, ID_LEFT, min_steps, speed=SWEEP_SPEED)
            move_servo(ser, ID_RIGHT, 4096 - min_steps, speed=SWEEP_SPEED)
            
            # Wait for movement + User Delay
            time.sleep(1.5 + wait_time)

    except KeyboardInterrupt:
        print("\nStopping...")
        # Optional: Stop command on exit
        # move_servo(ser, ID_LEFT, 0, speed=0) 
        # move_servo(ser, ID_RIGHT, 0, speed=0) # Careful, this might jerk 
        
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if ser and ser.is_open:
            ser.close()

if __name__ == "__main__":
    main()