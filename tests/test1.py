import serial
import time
import sys

# --- CONFIGURATION ---
# UPDATE THIS to match your computer's port
# Windows: 'COM3', 'COM4'
# Mac/Linux: '/dev/ttyUSB0' or '/dev/tty.usbserial-...'
SERIAL_PORT = 'COM3' 

# Baud Rate: 115200 (if using ESP32 forwarding) or 1000000 (if using direct USB-TTL)
BAUD_RATE = 115200 

def calculate_checksum(servo_id, length, instruction):
    # Checksum = ~(ID + Length + Instruction)
    checksum_sum = servo_id + length + instruction
    return (~checksum_sum) & 0xFF

def scan_servos():
    print(f"Opening {SERIAL_PORT} at {BAUD_RATE} baud...")
    
    try:
        # Initialize Serial
        ser = serial.Serial()
        ser.port = SERIAL_PORT
        ser.baudrate = BAUD_RATE
        ser.timeout = 0.05  # Fast timeout (50ms) to scan quickly
        
        # IMPORTANT: Prevent ESP32 Reset on Connect
        ser.dtr = False
        ser.rts = False
        
        ser.open()
        
    except serial.SerialException as e:
        print(f"Error opening port: {e}")
        return

    print("--- Starting Servo Scan (IDs 0-253) ---")

    # Loop through all possible IDs
    for servo_id in range(20):
        # 1. Build the PING Command
        header = [0xFF, 0xFF]
        length = 0x02
        instruction = 0x01 # PING
        
        # 2. Calculate Checksum automatically
        checksum = calculate_checksum(servo_id, length, instruction)
        
        # 3. Create the data packet
        packet = header + [servo_id, length, instruction, checksum]
        data_to_send = bytearray(packet)
        
        # 4. Clear old data and Send
        ser.reset_input_buffer()
        ser.write(data_to_send)
        
        # 5. Wait slightly for response
        # (Too fast might miss it, too slow makes scanning long)
        time.sleep(0.02) 
        
        # 6. Check if we received data back
        if ser.in_waiting > 0:
            response = ser.read(ser.in_waiting)
            
            # A valid response usually starts with FF FF
            if len(response) >= 6 and response[0] == 0xFF and response[1] == 0xFF:
                found_id = response[2]
                print(f"[FOUND] Servo detected at ID: {found_id} (Hex: {hex(found_id)})")
            else:
                # Detected noise or partial data
                print(f"[?] Data received at ID {servo_id}: {[hex(x) for x in response]}")
        
        # Optional: Print progress every 10 IDs so you know it's working
        if servo_id % 10 == 0:
            sys.stdout.write(f"\rScanning... \n{servo_id}/253")
            sys.stdout.flush()

    print("\n--- Scan Complete ---")
    ser.close()

if __name__ == "__main__":
    scan_servos()