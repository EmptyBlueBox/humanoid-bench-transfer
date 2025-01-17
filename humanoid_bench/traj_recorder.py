import os
import pickle
import time
from pprint import pp

class TrajRecorder:
    def __init__(self, dof_names, robot_name='h1',save_id="",
                 file_save_folder='./trajectory'):
        self.robot_name = robot_name
        self.file_save_folder = file_save_folder
        self.save_id = save_id
        self.max_trajectory_frame = 250
        self.dof_names = dof_names
        self.actions = []
        self.states = []

    def update(self, action, state, verbose=False):
        self.actions.append(action)
        self.states.append(state)
        if verbose:
            print("\n"*2+f"{'actions':=^30}")
            pp(self.actions[-2:])
            print(f"{'states':-^30}")
            pp(self.states[-2:])

        if len(self.states) >= self.max_trajectory_frame:
            self.save_trajectory_to_file()
            exit()

    def save_trajectory_to_file(self,
                                file_name=None):
        if file_name is None:
            file_name = f"{self.save_id}_{time.time():.0f}_traj_v2.pkl"
        trajectory_data = {
            self.robot_name: [{
                "actions": self.actions,
                # [
                #     "dof_pos_target":{
                #         "joint_name_1": 1.0,
                #         "joint_name_2": 1.0,
                #     },
                #     "ee_pose_target": {
                #         "pos": [1.0, 1.0, 1.0],
                #         "rot": [1.0, 1.0, 1.0, 1.0],
                #         "gripper_joint_pos": 1.0
                #     }
                # ]
                "init_state": self.states[0],
                # {
                #     self.robot_name: {
                #         "pos": self.trajectory[0][:3],
                #         "rot": self.trajectory[0][3:7],
                #         "dof_pos": dict(zip(self.dof_names, self.trajectory[0][7:])),
                #     },
                #     # "object_name1" :{
                #     #     "pos": [1.0, 1.0, 1.0],
                #     #     "rot": [1.0, 1.0, 1.0, 1.0]
                #     # }
                # },
                "states": self.states[1:],
                # [
                #     {self.robot_name: {
                #         "pos": self.trajectory[0][:3],
                #         "rot": self.trajectory[0][3:7],
                #         "dof_pos": dict(zip(self.dof_names, self.trajectory[0][7:])),
                #     }},
                # ],
                "extra":None
            }]
        }
        
        save_path = os.path.join(self.file_save_folder, file_name)
        with open(save_path, 'wb') as f:
            pickle.dump(trajectory_data, f)
        
        print(f"Trajectory saved to {save_path}")