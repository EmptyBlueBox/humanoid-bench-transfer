import os
import pickle
import time
from pprint import pp
import copy

class TrajRecorder:
    def __init__(self, dof_names, robot_name='h1',save_id="",
                 max_trajectory_frame=float('inf'), # default to record all frames
                 file_save_folder='./trajectory'):
        self.robot_name = robot_name
        self.file_save_folder = file_save_folder
        self.save_id = save_id
        self.max_trajectory_frame = max_trajectory_frame
        self.dof_names = dof_names
        self.start_new_record()

    def update(self, action=None, state=None, env=None, 
               verbose=False, auto_format=False, auto_quit=False):
        """
        If formatted flag set True, pass raw action and env
        Else, pass formatted action and formatted state
        """
        
        action = copy.deepcopy(action)

        if auto_format:
            assert env is not None and state is None, "auto_format only works with raw action and env"
            action = {
                "dof_pos_target":{
                    k:v for k,v in zip(self.dof_names[1:],
                                       env.task.unnormalize_action(action))
                }
            } if action is not None else None
            state = {
                env.get_wrapper_attr('robot').name:{
                    "pos": env.get_wrapper_attr('data').qpos[:3],
                    "rot": env.get_wrapper_attr('data').qpos[3:7],
                    "dof_pos": {k:v for k,v in zip(self.dof_names[1:],env.get_wrapper_attr('data').qpos[7:])}
                }
            }
        else:
            assert state is not None, "state must be provided if auto_format is False"
        
        state = copy.deepcopy(state)

        if action is not None:
            self.actions.append(action)
        self.states.append(state)
        if verbose:
            print("\n"*2+f"{'actions':=^30}")
            pp(self.actions[-2:])
            print(f"{'states':-^30}")
            pp(self.states[-2:])

        if len(self.states) >= self.max_trajectory_frame:
            self.save_and_start_new()
            if auto_quit:
                exit()
            return True
        return False
            

    def start_new_record(self):
        self.actions = []
        self.states = []

    def save_and_start_new(self, file_name=None):
        if len(self.states) == 0:
            return
        self.save_trajectory_to_file(file_name)
        self.start_new_record()


    def save_trajectory_to_file(self,
                                file_name=None):
        if file_name is None:
            file_name = f"{self.save_id}_{time.strftime('%m%d_%H%M%S')}_traj_v2.pkl"
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
        
        print(f"Trajectory saved to {save_path}, length: {len(self.states)}")