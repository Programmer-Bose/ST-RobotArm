import numpy as np
import math

# --- ROBOT DIMENSIONS (Meters) ---
# Based on your previous code/images
L_BASE = 0.10       # Height from base to shoulder pivot
L_SHOULDER = 0.245  # Length of shoulder link
L_ELBOW = 0.145     # Length of elbow link
L_PITCH = 0.155     # Length from pitch pivot to gripper tip

# --- CONSTRAINTS ---
# Servo ranges (0 to 180)
LIMITS = {
    'base': (0, 180),
    'shoulder': (0, 180),
    'elbow': (0, 180),
    'pitch': (0, 180),
    'roll': (0, 180)
}

def geometric_ik(x, y, z, target_pitch_rad=0.0):
    """
    Calculates Joint Angles (Degrees) for a given XYZ coordinate.
    
    Args:
        x, y, z: Target coordinates in meters.
        target_pitch_rad: Desired angle of the gripper relative to the ground (0 = Horizontal).
        
    Returns:
        [Base, Shoulder, Elbow, Pitch, Roll] in degrees.
        Returns None if unreachable.
    """
    
    # --- 1. BASE ANGLE (Theta 1) ---
    # Simple atan2 to find the angle in the XY plane
    theta_base_rad = math.atan2(y, x)
    
    # Map to Servo: 90 is Center (Forward/Y+), 0 is Right, 180 is Left
    # If x>0, y=0 -> atan2=0. Servo should be 90?
    # Let's align with your previous sim:
    # If Robot faces Y+, then theta_base=90 is Y+. 
    # Your logic: 90 - degrees.
    base_deg = math.degrees(theta_base_rad)
    # Correction: Assuming 90 is forward (aligned with X axis in calculation frame?)
    # Usually: Base=math.degrees(atan2(y,x)). If Forward is Y, offset by 90.
    # Let's use standard polar: 
    theta1 = math.degrees(theta_base_rad)
    # Final Base Servo Value (Adjust based on your specific assembly zero-point)
    servo_base = theta1 
    if servo_base < 0: servo_base += 360 # Wrap
    
    # --- 2. WRIST POSITION CALCULATION ---
    # We want to solve for the Wrist Center, not the Tip.
    # Work in the 2D plane created by the arm's extension (R-Z plane).
    r_total = math.sqrt(x**2 + y**2)  # Horizontal distance to tip
    z_total = z - L_BASE              # Vertical distance from shoulder pivot
    
    # Calculate Wrist Center (Rw, Zw) by backing up from the tip by L_PITCH
    # We use 'target_pitch_rad' to determine the angle of the gripper.
    rw = r_total - L_PITCH * math.cos(target_pitch_rad)
    zw = z_total - L_PITCH * math.sin(target_pitch_rad)
    
    # Distance from Shoulder Pivot to Wrist Center
    D = math.sqrt(rw**2 + zw**2)
    
    # Check Reachability
    if D > (L_SHOULDER + L_ELBOW):
        print("❌ Target Unreachable (Too far)")
        return None
    if D < abs(L_SHOULDER - L_ELBOW):
        print("❌ Target Unreachable (Too close)")
        return None

    # --- 3. LAW OF COSINES (Shoulder & Elbow) ---
    # Solve the triangle formed by L_SHOULDER, L_ELBOW, and line D
    
    # Alpha: Angle of vector D relative to horizon
    alpha = math.atan2(zw, rw)
    
    # Beta: Internal angle at Shoulder
    # c^2 = a^2 + b^2 - 2ab cos(C)  ->  cos(C) = (a^2 + b^2 - c^2) / 2ab
    cos_beta = (L_SHOULDER**2 + D**2 - L_ELBOW**2) / (2 * L_SHOULDER * D)
    beta = math.acos(max(-1, min(1, cos_beta))) # Clamp for safety
    
    # Gamma: Internal angle at Elbow
    cos_gamma = (L_SHOULDER**2 + L_ELBOW**2 - D**2) / (2 * L_SHOULDER * L_ELBOW)
    gamma = math.acos(max(-1, min(1, cos_gamma)))
    
    # --- 4. JOINT ANGLES (Radians) ---
    # Shoulder Angle (Theta 2): alpha + beta
    theta_shoulder_rad = alpha + beta
    
    # Elbow Angle (Theta 3): 
    # The internal angle is gamma. The deviation from straight is (pi - gamma).
    # Standard "Elbow Up" configuration logic.
    theta_elbow_rad = gamma 
    
    # Pitch Angle (Theta 4):
    # Global Pitch = Shoulder + Elbow_Relative + Pitch_Relative
    # We want Global Pitch to equal target_pitch_rad
    # So: Pitch_Relative = target_pitch_rad - Shoulder - Elbow_Relative
    # Note: Elbow_Relative is usually negative in this frame.
    # Let's simply calculate geometry angle sum:
    # The wrist pitch must compensate for the shoulder and elbow angles to keep the tool at target_pitch.
    # Angle sum logic depends heavily on servo zero definitions.
    # Simplest approach: Pitch_Servo = Target_Global - (Shoulder_Global + Elbow_Global)
    # Here, let's output the raw geometric angle of the pitch link relative to horizon first.
    theta_pitch_rad = target_pitch_rad - (theta_shoulder_rad - (math.pi - theta_elbow_rad))

    # --- 5. CONVERT TO SERVO DEGREES (Applying Offsets) ---
    # Based on your previous code's "Servo Offsets":
    # Base: Direct?
    # Shoulder: 0=Horizontal? 90=Up?
    # Elbow: 180=Straight?
    # Pitch: 90=Straight?
    
    deg_shoulder = math.degrees(theta_shoulder_rad)
    deg_elbow_internal = math.degrees(theta_elbow_rad)
    
    # Mapping Geometry -> Servo Values
    # YOU MAY NEED TO TUNE THESE +/- SIGNS BASED ON YOUR PHYSICAL ASSEMBLY
    servo_shoulder = deg_shoulder 
    
    # Elbow: If 180 is straight line, and gamma is internal angle (180=straight)
    servo_elbow = deg_elbow_internal 
    
    # Pitch: 
    servo_pitch = 90 + math.degrees(theta_pitch_rad)

    # Roll (Passive/User Defined)
    servo_roll = 0 # Default Center

    # --- 6. SAFETY CLAMP ---
    joints = [servo_base, servo_shoulder, servo_elbow, servo_pitch, servo_roll]
    final_joints = []
    
    for i, ang in enumerate(joints):
        # Wrap Base if needed
        if i==0 and ang > 180: ang -= 360
        
        clamped = max(0, min(180, ang))
        final_joints.append(int(clamped))
        
    return final_joints

# --- MAIN TEST LOOP ---
if __name__ == "__main__":
    print("=== GEOMETRIC IK SOLVER ===")
    print("Enter Target XYZ (in Meters). Example: 0.2 0.0 0.15")
    
    while True:
        try:
            user_in = input("\nTarget X Y Z > ")
            if user_in.lower() in ['q', 'exit']: break
            
            parts = user_in.split()
            if len(parts) < 3: continue
            
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            
            # Optional: Ask for gripper pitch (e.g., 0 for horizontal, -90 for pointing down)
            # pitch = float(input("Gripper Pitch (deg, default 0): ") or 0)
            pitch = 0 
            
            angles = geometric_ik(x, y, z, math.radians(pitch))
            
            if angles:
                print(f"✅ SOLUTION FOUND: {angles}")
                print(f"   (Base, Shldr, Elbow, Pitch, Roll)")
            else:
                print("❌ No Solution Found.")
                
        except ValueError:
            print("Invalid Input")