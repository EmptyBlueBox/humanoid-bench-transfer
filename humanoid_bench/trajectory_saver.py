import os
import pickle

class TrajectorySaver:
    def __init__(self, dof_names):
        self.file_name = 'humanoidbench_reach_traj_v2.pkl'
        self.file_save_folder = '/home/descfly/humanoid-bench-transfer/Trajectory'
        self.max_trajectory_frame = 200
        self.dof_names = dof_names
        self.trajectory = []

    def update_trajectory(self, dof_pos):
        self.trajectory.append(dof_pos)
        if len(self.trajectory) >= self.max_trajectory_frame:
            self.save_trajectory_to_file()
            exit()

    def save_trajectory_to_file(self):
        trajectory_data = {
            'h1': [{
                'init_state': {
                    'h1': {
                        'dof_pos': dict(zip(self.dof_names, self.trajectory[0][7:])),
                        'pos': self.trajectory[0][:3],
                        'rot': self.trajectory[0][3:7]
                    }
                },
                'states': [{
                    'h1': {
                        'dof_pos': dict(zip(self.dof_names, frame[7:])),
                        'pos': frame[:3],
                        'rot': frame[3:7]
                    }
                } for frame in self.trajectory[1:]]
            }]
        }
        
        save_path = os.path.join(self.file_save_folder, self.file_name)
        with open(save_path, 'wb') as f:
            pickle.dump(trajectory_data, f)
        
        print(f"Trajectory saved to {save_path}")
        
