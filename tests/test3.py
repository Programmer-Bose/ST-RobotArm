import serial
import time

# --- CONFIGURATION ---
SERIAL_PORT = 'COM3'    # Change to your port
BAUD_RATE = 115200      # 115200 for forwarding mode, 1000000 for direct
SERVO_ID = 3            # The ID of your servo

# ST3215 Resolution: 0-4095 = 0-360 degrees
# Therefore, 180 degrees = 2048 steps
MIN_POS = 0
MAX_POS = 2048          # ~180 Degrees
SPEED = 1500            # Movement speed (steps per second usually)

def set_servo_position(ser, servo_id, position, speed):
    """
    Moves the servo to a specific position.
    Protocol: WRITE (0x03) to Address 0x2A (Goal Position).
    We write 4 bytes: [Position_Low, Position_High, Speed_Low, Speed_High]
    """
    
    # 1. Clamp values to safe ranges
    if position < 0: position = 0
    if position > 4095: position = 4095
    
    # 2. Split Integer into Low/High Bytes
    p_low = position & 0xFF
    p_high = (position >> 8) & 0xFF
    
    s_low = speed & 0xFF
    s_high = (speed >> 8) & 0xFF
    
    # 3. Construct Frame
    header = [0xFF, 0xFF]
    instruction = 0x03       # WRITE
    start_addr = 0x2A        # Address 42 (Goal Position)
    
    # Data = Pos_L, Pos_H, Speed_L, Speed_H
    data = [p_low, p_high, s_low, s_high]
    
    # Length = Instr(1) + Addr(1) + Data(4) + Checksum(1)
    length = 0x07
    
    # Calculate Checksum
    checksum_sum = servo_id + length + instruction + start_addr + sum(data)
    checksum = (~checksum_sum) & 0xFF
    
    frame = bytearray(header + [servo_id, length, instruction, start_addr] + data + [checksum])
    
    # 4. Send Command
    try:
        ser.write(frame)
        # No need to read response for WRITE commands usually, 
        # unless you programmed the servo to reply to everything.
    except Exception as e:
        print(f"Write Error: {e}")

def main():
    ser = None
    try:
        # --- INIT SERIAL (Anti-Reset) ---
        ser = serial.Serial()
        ser.port = SERIAL_PORT
        ser.baudrate = BAUD_RATE
        ser.timeout = 1
        
        # KEY: Disable DTR/RTS to prevent ESP32 Reset
        ser.dtr = False
        ser.rts = False
        
        ser.open()
        print(f"Connected to {SERIAL_PORT}. Starting Sweep...")
        print("Press Ctrl+C to stop.")

        while True:
            # --- MOVE TO 0 ---
            print(f"Moving to 0 (0 degrees)...")
            set_servo_position(ser, SERVO_ID, MIN_POS, SPEED)
            
            # Wait for servo to get there (approx calculation or fixed delay)
            time.sleep(2) 

            # --- MOVE TO 2048 ---
            print(f"Moving to {MAX_POS} (180 degrees)...")
            set_servo_position(ser, SERVO_ID, MAX_POS, SPEED)
            
            time.sleep(2)

    except serial.SerialException as e:
        print(f"Serial Error: {e}")
    except KeyboardInterrupt:
        print("\nStopping...")
        # Optional: Disable torque on exit?
    finally:
        if ser and ser.is_open:
            ser.close()
            print("Port Closed.")

if __name__ == "__main__":
    main()