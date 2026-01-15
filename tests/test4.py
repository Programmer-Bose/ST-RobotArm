import serial
import time

# --- CONFIGURATION ---
SERIAL_PORT = 'COM3'   # Check your port
BAUD_RATE = 115200     # 115200 (Forwarding)

# List your Servo IDs here
SERVO_IDS = [16] 

def construct_move_packet(servo_id, position, speed):
    """
    Creates the raw bytes for a single motor move command.
    """
    # Clamp Position (0-4095)
    if position < 0: position = 0
    if position > 4095: position = 4095
    
    # Split Data
    p_low = position & 0xFF
    p_high = (position >> 8) & 0xFF
    s_low = speed & 0xFF
    s_high = (speed >> 8) & 0xFF
    
    # Checksum Calc: ~(ID + Len + Instr + Addr + P_L + P_H + S_L + S_H)
    # Length is 7
    checksum_sum = servo_id + 0x07 + 0x03 + 0x2A + p_low + p_high + s_low + s_high
    checksum = (~checksum_sum) & 0xFF
    
    return bytearray([0xFF, 0xFF, servo_id, 0x07, 0x03, 0x2A, p_low, p_high, s_low, s_high, checksum])

def main():
    ser = None
    try:
        # --- 1. SAFE SERIAL OPENING ---
        # We configure the object BEFORE opening to prevent DTR spikes
        ser = serial.Serial()
        ser.port = SERIAL_PORT
        ser.baudrate = BAUD_RATE
        ser.timeout = 0.1
        
        # CRITICAL: Force pins LOW to stop ESP32 Reset
        ser.dtr = False
        ser.rts = False
        
        ser.open()
        
        # Double check pins are low after opening
        ser.dtr = False
        ser.rts = False
        
        print(f"Connected to {SERIAL_PORT}. Starting Loop (Ctrl+C to stop)...")
        time.sleep(1) # Give the connection a moment to stabilize

        while True:
            # --- POSE 1: Zero ---
            print("Moving to 0...")
            for sid in SERVO_IDS:
                packet = construct_move_packet(sid, 0, 1500)
                ser.write(packet)
                time.sleep(0.01) # Tiny delay between sends helps data integrity
            
            time.sleep(5)

            # --- POSE 2: 180 Degrees (2048) ---
            print("Moving to 2048...")
            for sid in SERVO_IDS:
                packet = construct_move_packet(sid, 2048, 1500)
                ser.write(packet)
                time.sleep(0.01)
            
            time.sleep(2)

    except serial.SerialException as e:
        print(f"\n[Error] Serial Port Issue: {e}")
        print("Check if the cable came loose or power dropped.")
    
    except KeyboardInterrupt:
        print("\n[Stop] User interrupted.")
    
    finally:
        # This block ALWAYS runs, ensuring the port closes properly
        if ser and ser.is_open:
            ser.close()
            print("Serial Port Closed Cleanly.")

if __name__ == "__main__":
    main()