import serial
import time

# --- CONFIGURATION ---
SERIAL_PORT = 'COM3'   # Check your port
BAUD_RATE = 115200     # 115200 (Forwarding)

def calculate_checksum(payload):
    return (~sum(payload)) & 0xFF

def send_write_packet(ser, servo_id, address, value):
    # Header + ID + Len + Instr + Addr + Data + Cksm
    # Len = 4 (Instr+Addr+Data+Cksm)
    # Instr = 0x03 (WRITE)
    checksum = calculate_checksum([servo_id, 4, 3, address, value])
    packet = bytearray([0xFF, 0xFF, servo_id, 4, 3, address, value, checksum])
    
    ser.write(packet)
    time.sleep(0.05)

def force_permanent_id_change(ser, old_id, new_id):
    print(f"--- ATTEMPTING PERMANENT ID CHANGE: {old_id} -> {new_id} ---")

    # 1. UNLOCK (Try both common Lock addresses)
    # Write 0 to Addr 48 (Old standard) and Addr 55 (New standard)
    print("[Step 1] Unlocking EEPROM...")
    send_write_packet(ser, old_id, 48, 0) # Address 0x30
    send_write_packet(ser, old_id, 55, 0) # Address 0x37
    time.sleep(0.1)

    # 2. WRITE NEW ID
    print(f"[Step 2] Writing New ID {new_id} to Address 5...")
    send_write_packet(ser, old_id, 5, new_id)
    time.sleep(0.1)

    # 3. SAVE / LOCK (This is the critical part)
    # We must command the NEW ID to Lock itself.
    # Writing '1' to these registers triggers the transfer from RAM -> EEPROM
    print(f"[Step 3] Saving to EEPROM (Locking)...")
    send_write_packet(ser, new_id, 48, 1) 
    send_write_packet(ser, new_id, 55, 1)
    
    print("Done. Disconnect Power NOW and wait 5 seconds before reconnecting.")

def main():
    ser = None
    try:
        ser = serial.Serial()
        ser.port = SERIAL_PORT
        ser.baudrate = BAUD_RATE
        ser.timeout = 0.1
        ser.dtr = False; ser.rts = False
        ser.open()
        
        current_id = int(input("Enter CURRENT ID: "))
        target_id = int(input("Enter NEW ID: "))
        
        force_permanent_id_change(ser, current_id, target_id)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if ser and ser.is_open:
            ser.close()

if __name__ == "__main__":
    main()