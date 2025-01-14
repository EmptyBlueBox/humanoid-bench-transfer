import pickle
import pprint

def read_and_print_trajectory(file_path):
    # Read the pickle file
    with open(file_path, 'rb') as f:
        trajectory_data = pickle.load(f)
    
    # Print available robot names
    robot_names = list(trajectory_data.keys())
    print("Available robots:", robot_names)
    
    if not robot_names:
        print("No trajectories found")
        return
        
    # Get the first robot's trajectories
    first_robot = robot_names[0]
    robot_trajectories = trajectory_data[first_robot]
    if not robot_trajectories:
        print(f"No trajectories found for {first_robot}")
        return
    
    # Print the first trajectory's information
    first_traj = robot_trajectories[0]
    
    print(f"\n=== Readable Trajectory Information for {first_robot} ===")
    print("\nTrajectory Keys:")
    print(first_traj.keys())
    key = 'init_state'
    print(f"traj['{key}']['{first_robot}'] size: {len(first_traj[key][first_robot])}")
    print(f"traj['{key}']['{first_robot}']['dof_pos'] size: {len(first_traj[key][first_robot]['dof_pos'])}")
    
    # Save to text file
    output_file = file_path.rsplit('.', 1)[0] + '_readable.txt'
    with open(output_file, 'w') as f:
        f.write(f"Available robots: {robot_names}\n\n")
        f.write(f"=== Sample Trajectory Information for {first_robot} ===\n\n")
        f.write("Trajectory Keys:\n")
        f.write(str(first_traj.keys()) + "\n\n")
        
        f.write(pprint.pformat(trajectory_data, indent=2))
        # # Print each available key and its content
        # for key in first_traj.keys():
        #     f.write(f"\n{key}:\n")
        #     f.write(f"traj['{key}'] size: {len(first_traj[key])}\n")
        #     if key == 'states' and len(first_traj[key]) > 3:
        #         f.write(f"traj['{key}'][0]['h1_2_without_hand'] size: {len(first_traj[key][0]['h1_2_without_hand'])}\n")
        #         f.write(f"traj['{key}'][0]['h1_2_without_hand']['dof_pos'] size: {len(first_traj[key][0]['h1_2_without_hand']['dof_pos'])}\n")
        #         # Only show first 3 frames for states
        #         truncated_states = first_traj[key][:3]
        #         f.write(pprint.pformat(truncated_states, indent=2) + "\n")
        #         f.write("... (remaining frames omitted)\n")
        #         print(f"\n{key}:")
        #         pprint.pprint(truncated_states, indent=2)
        #         print("... (remaining frames omitted)")
        #     else:
        #         f.write(f"traj['{key}']['h1_2_without_hand'] size: {len(first_traj[key]['h1_2_without_hand'])}\n")
        #         f.write(f"traj['{key}']['h1_2_without_hand']['dof_pos'] size: {len(first_traj[key]['h1_2_without_hand']['dof_pos'])}\n")
        #         # Print other keys normally
        #         f.write(pprint.pformat(first_traj[key], indent=2) + "\n")
        #         print(f"\n{key}:")
        #         pprint.pprint(first_traj[key], indent=2)
    
    print(f"\nData has been saved to: {output_file}")

if __name__ == "__main__":
    trajectory_file = "/home/descfly/humanoid-bench-transfer/Trajectory/humanoidbench_reach_traj_v2.pkl"
    # trajectory_file = "/home/descfly/humanoid-bench-transfer/Trajectory/maobaoguo_traj_v2.pkl"
    read_and_print_trajectory(trajectory_file)
