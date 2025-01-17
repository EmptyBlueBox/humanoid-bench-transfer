import os
import pickle

class TrajectorySaver:
    def __init__(self, dof_names):
        self.file_name = "humanoidbench_reach_traj_v2.pkl"
        self.file_save_folder = "/home/descfly/humanoid-bench-transfer/Trajectory"
        self.max_trajectory_frame = 500
        self.dof_names = dof_names
        self.state_trajectory = []
        self.action_trajectory = []

    def update_dof_trajectory(self, dof_pos):
        self.state_trajectory.append(dof_pos)
        if (len(self.state_trajectory) == self.max_trajectory_frame + 1) and (len(self.action_trajectory) == self.max_trajectory_frame):
            self.save_trajectory_to_file()
            exit()

    def update_action_trajectory(self, action):
        self.action_trajectory.append(action)
        if (len(self.action_trajectory) == self.max_trajectory_frame) and (len(self.state_trajectory) == self.max_trajectory_frame + 1):
            self.save_trajectory_to_file()
            exit()

    def save_trajectory_to_file(self):
        trajectory_data = {
            "h1": [{
                "actions": [{
                    "dof_pos_target": {
                        k:v for k,v in zip(self.dof_names, frame)
                    }
                } for frame in self.action_trajectory],
                "init_state": {
                    "h1": {
                        "dof_pos": dict(zip(self.dof_names, self.state_trajectory[0][7:])),
                        "pos": self.state_trajectory[0][:3],
                        "rot": self.state_trajectory[0][3:7]
                    }
                },
                "states": [{
                    "h1": {
                        "dof_pos": dict(zip(self.dof_names, frame[7:])),
                        "pos": frame[:3],
                        "rot": frame[3:7]
                    }
                } for frame in self.state_trajectory[1:]]
            }]
        }
        
        save_path = os.path.join(self.file_save_folder, self.file_name)
        with open(save_path, "wb") as f:
            pickle.dump(trajectory_data, f)
        
        print(f"Trajectory saved to {save_path}")
        
