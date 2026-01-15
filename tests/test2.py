import serial
import time

# --- CONFIGURATION ---
SERIAL_PORT = 'COM3'   # Change to your port
BAUD_RATE = 115200     # Change if needed

def get_servo_snapshot(servo_id):
    ser = None
    try:
        # 1. OPEN PORT
        ser = serial.Serial()
        ser.port = SERIAL_PORT
        ser.baudrate = BAUD_RATE
        ser.timeout = 0.5  # Give it 0.5 seconds max to reply
        
        # Prevent Reset (Try to keep DTR/RTS low)
        ser.dtr = False
        ser.rts = False
        
        ser.open()
        
        # 2. FLUSH TRASH
        # Clear any old data sitting in the hardware buffer
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        # 3. SEND COMMAND (Read Position from ID 1)
        # Checksum = ~(ID + Len + Instr + P1 + P2) = ~(1 + 4 + 2 + 38 + 2) = ~(47) = B8
        # Note: Recalculate if ID changes. Below is hardcoded for ID 1.
        if servo_id == 1:
            cmd = bytearray([0xFF, 0xFF, 0x01, 0x04, 0x02, 0x38, 0x02, 0xBE])
        else:
            # Dynamic calculation for other IDs
            checksum = (~(servo_id + 0x04 + 0x02 + 0x38 + 0x02)) & 0xFF
            cmd = bytearray([0xFF, 0xFF, servo_id, 0x04, 0x02, 0x38, 0x02, checksum])

        ser.write(cmd)

        # 4. READ RESPONSE (Snapshot)
        # We need 8 bytes. We wait slightly to ensure they arrived.
        time.sleep(0.05) 
        
        response = ser.read(ser.in_waiting or 8)

        # 5. CLOSE PORT IMMEDIATELY
        ser.close()
        
        # 6. PARSE DATA (Offline)
        # We are now disconnected, so we can take our time to check the data.
        print(f"Raw Hex Received: {[hex(x) for x in response]}")

        # Validate Header
        if len(response) >= 8 and response[0] == 0xFF and response[1] == 0xFF:
            # Position is Low Byte (idx 5) + High Byte (idx 6)
            position = response[5] + (response[6] << 8)
            return position
        else:
            print("Error: Invalid packet structure.")
            return None

    except Exception as e:
        if ser and ser.is_open:
            ser.close()
        print(f"Connection Error: {e}")
        return None

# --- RUN ONCE ---
if __name__ == "__main__":
    print("Taking snapshot...")
    for i in range(11,18):
        pos = get_servo_snapshot(i)
        if pos is not None:
            print(f"--- SNAPSHOT SUCCESS ---")
            print(f"Servo Position with ID {i}: {pos}")
        else:
            print("--- SNAPSHOT FAILED ---")