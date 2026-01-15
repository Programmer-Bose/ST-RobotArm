import numpy as np
import sys

# --- 1. CONFIGURATION ---
STEP_XYZ = 0.01   # Meters per key press (1 cm)
STEP_ROLL = 5.0   # Degrees per key press (for Q/R)

# DH Parameters from our verified table [a, alpha, d] (theta is variable)
# Units: Meters, Radians
DH_PARAMS = [
    {'a': 0,     'alpha': np.pi/2,  'd': 0.10}, # J1: Base
    {'a': 0.26,  'alpha': 0,        'd': 0.00}, # J2: Shoulder
    {'a': 0.18,  'alpha': 0,        'd': 0.00}, # J3: Elbow
    {'a': 0.12,  'alpha': -np.pi/2, 'd': 0.00}, # J4: Pitch
    {'a': 0,     'alpha': 0,        'd': 0.05}  # J5: Roll (Includes 5cm Tip)
]

def get_transform(a, alpha, d, theta):
    """Calculates individual link transform matrix."""
    c, s = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [c, -s*ca,  s*sa, a*c],
        [s,  c*ca, -s*sa, a*s],
        [0,  sa,    ca,   d],
        [0,  0,     0,    1]
    ])

def forward_kinematics(joints):
    """Returns the (3,1) position of the End Effector."""
    T = np.eye(4)
    for i, p in enumerate(DH_PARAMS):
        T = T @ get_transform(p['a'], p['alpha'], p['d'], joints[i])
    return T[:3, 3] # Return X, Y, Z

def calculate_jacobian(joints):
    """
    Calculates the 3x5 Jacobian Matrix (Positional only).
    J = [dx/dq1 ... dx/dq5]
        [dy/dq1 ... dy/dq5]
        [dz/dq1 ... dz/dq5]
    """
    n = len(joints)
    J = np.zeros((3, n))
    
    # Calculate global transforms for all frames
    transforms = []
    T_curr = np.eye(4)
    transforms.append(T_curr) # Base frame
    
    for i, p in enumerate(DH_PARAMS):
        T_curr = T_curr @ get_transform(p['a'], p['alpha'], p['d'], joints[i])
        transforms.append(T_curr)
        
    # End Effector Position (Pe)
    Pe = transforms[-1][:3, 3]
    
    # Fill Jacobian columns
    for i in range(n):
        # Position of joint i-1 (P_prev)
        P_prev = transforms[i][:3, 3]
        
        # Z-axis of joint i-1 (Z_prev) -> Extracts 3rd column of rotation matrix
        Z_prev = transforms[i][:3, 2]
        
        # Geometric Jacobian Formula: J_vi = Z_(i-1) x (Pe - P_(i-1))
        # This gives the velocity vector at the tip caused by joint i
        J_col = np.cross(Z_prev, Pe - P_prev)
        
        J[:, i] = J_col
        
    return J

def inverse_differential_kinematics(current_joints, delta_xyz):
    """
    Calculates change in joint angles (delta_theta) required for a delta_xyz move.
    Uses Pseudo-Inverse of Jacobian.
    """
    # 1. Get current Jacobian (3x5 matrix)
    J = calculate_jacobian(current_joints)
    
    # 2. Invert Jacobian (Damped Least Squares or Pseudo-Inverse)
    # Since we have 5 joints controlling 3 XYZ positions, it is redundant.
    # Pseudo-inverse finds the solution with minimum joint movement.
    J_pinv = np.linalg.pinv(J)
    
    # 3. Calculate Delta Thetas: dq = J_pinv * dx
    delta_theta = J_pinv @ delta_xyz
    
    return delta_theta

# --- MAIN LOOP ---
def main():
    print("=== Robot Arm Keyboard Control ===")
    print("Enter Initial Joint Angles (Degrees):")
    try:
        q_init = [
            float(input("J1: ")), float(input("J2: ")), 
            float(input("J3: ")), float(input("J4: ")), 
            float(input("J5: "))
        ]
        q = np.radians(q_init)
    except:
        print("Invalid input, using defaults: 45, 30, -30, -30, 0")
        q = np.radians([45.0, 30.0, -30.0, -30.0, 0.0])

    print("\n--- CONTROLS ---")
    print(" [W/S] Y +/-   | [A/D] X -/+ (Note: A is X-)")
    print(" [SPC] Z +     | [Shift+SPC] Z - (Type 'Z' or 'z' for simplicity)")
    print(" [Q/R] Roll    | [E] Exit")
    print("----------------")
    
    while True:
        # 1. Get Current Status
        curr_pos = forward_kinematics(q)
        print(f"\nCurrent XYZ: [{curr_pos[0]:.4f}, {curr_pos[1]:.4f}, {curr_pos[2]:.4f}]")
        print(f"Joints (deg): {np.degrees(q).astype(int)}")
        
        # 2. Get User Input
        cmd = input("Command > ").lower()
        
        # 3. Determine Delta XYZ (Task Space Vector)
        d_xyz = np.array([0.0, 0.0, 0.0])
        manual_roll = 0.0
        
        if cmd == 'e': break
        elif cmd == 'w': d_xyz[1] = STEP_XYZ  # Y+
        elif cmd == 's': d_xyz[1] = -STEP_XYZ # Y-
        elif cmd == 'd': d_xyz[0] = STEP_XYZ  # X+
        elif cmd == 'a': d_xyz[0] = -STEP_XYZ # X-
        elif cmd == ' ': d_xyz[2] = STEP_XYZ  # Z+ (Space)
        elif cmd == 'z': d_xyz[2] = -STEP_XYZ # Z- (Simulating Shift+Space)
        elif cmd == 'q': manual_roll = np.radians(STEP_ROLL)  # Roll CW
        elif cmd == 'r': manual_roll = np.radians(-STEP_ROLL) # Roll CCW
        else: continue

        # 4. Calculate Angular Displacement (Delta Theta)
        if np.linalg.norm(d_xyz) > 0:
            # Solve Inverse Kinematics for XYZ
            d_theta = inverse_differential_kinematics(q, d_xyz)
            
            # Print the math requested
            print(f"Desired Delta XYZ: {d_xyz}")
            print("Calculated Angular Displacements (Delta Theta):")
            names = ["Base", "Shldr", "Elbow", "Pitch", "Roll"]
            for i in range(5):
                print(f"  D_{names[i]}: {d_theta[i]:.6f} rad ({np.degrees(d_theta[i]):.3f} deg)")
            
            # Update Joints
            q += d_theta
            
        # 5. Apply Manual Roll (Orientation only)
        # Note: Roll joint (J5) does not affect XYZ position, so we add it directly.
        if manual_roll != 0:
            q[4] += manual_roll
            print(f"  D_Roll: {manual_roll:.6f} rad")

if __name__ == "__main__":
    main()