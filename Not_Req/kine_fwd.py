import numpy as np

def get_dh_transform(a, alpha, d, theta):
    """
    Calculates the 4x4 homogenous transformation matrix for a single link
    using the Denavit-Hartenberg (DH) convention.
    """
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    c_alpha = np.cos(alpha)
    s_alpha = np.sin(alpha)

    # Standard DH Matrix
    return np.array([
        [c_theta, -s_theta*c_alpha,  s_theta*s_alpha, a * c_theta],
        [s_theta,  c_theta*c_alpha, -c_theta*s_alpha, a * s_theta],
        [0,        s_alpha,          c_alpha,         d],
        [0,        0,                0,               1]
    ])

def forward_kinematics(joint_angles_deg):
    """
    Computes end-effector pose.
    Includes SERVO OFFSETS so that 90 degrees can be 'Straight'.
    """
    # 1. Define Offsets: What angle makes the joint STRAIGHT (DH = 0)?
    # If your servos are straight at 90, put 90 here.
    # Based on your input, it seems you want J3 and J4 to be straight at 90.
    servo_offsets = np.array([0.0, 0.0, 90.0, 90.0, 0.0]) 
    
    # 2. Apply Offset: DH_Angle = Input_Angle - Offset
    # Example: If Input is 90 and Offset is 90, Math uses 0 (Straight).
    dh_angles_deg = np.array(joint_angles_deg) - servo_offsets
    
    # 3. Convert to radians for calculation
    q = np.radians(dh_angles_deg)

    # --- Rest of the code is the same ---
    dh_table = [
        {'a': 0,     'alpha': np.pi/2,  'd': 0.10, 'theta': q[0]}, # Base
        {'a': 0.245,  'alpha': 0,        'd': 0,    'theta': q[1]}, # Shoulder
        {'a': 0.145,  'alpha': 0,        'd': 0,    'theta': q[2]}, # Elbow
        {'a': 0.155,  'alpha': -np.pi/2, 'd': 0,    'theta': q[3]}, # Pitch
        {'a': 0,     'alpha': 0,        'd': 0,    'theta': q[4]}  # Roll
    ]

    T_total = np.eye(4)
    for row in dh_table:
        T_link = get_dh_transform(row['a'], row['alpha'], row['d'], row['theta'])
        T_total = np.dot(T_total, T_link)

    return T_total

# --- User Input Section ---
if __name__ == "__main__":
    print("=== 5-DOF Robot Forward Kinematics Solver ===")
    print("Please enter the 5 joint angles in degrees.")
    
    try:
        # Taking inputs one by one for clarity
        j1 = float(input("Joint 1 (Base Pan)   [deg]: "))
        j2 = float(input("Joint 2 (Shoulder)   [deg]: "))
        j3 = float(input("Joint 3 (Elbow)      [deg]: "))
        j4 = float(input("Joint 4 (Wrist Pitch)[deg]: "))
        j5 = float(input("Joint 5 (Wrist Roll) [deg]: "))
        
        user_joints = [j1, j2, j3, j4, j5]
        
        # Calculate
        final_transform = forward_kinematics(user_joints)
        
        # Extract components
        position = final_transform[:3, 3]
        rotation_matrix = final_transform[:3, :3]
        
        # --- Output Display ---
        print("\n" + "="*40)
        print(f"RESULTS for Input Angles: {user_joints}")
        print("="*40)
        
        print("\n[End-Effector Position (X, Y, Z) in Meters]")
        print(f"X: {position[0]:.6f}")
        print(f"Y: {position[1]:.6f}")
        print(f"Z: {position[2]:.6f}")
        
        print("\n[Rotation Matrix (3x3)]")
        # Printing purely the matrix, formatted nicely
        print(np.array2string(rotation_matrix, formatter={'float_kind':lambda x: "%.4f" % x}))
        
        print("\n" + "="*40)

    except ValueError:
        print("Error: Please enter valid numbers for the angles.")