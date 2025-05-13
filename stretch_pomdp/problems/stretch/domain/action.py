from pomdp_py.framework import basics
import numpy as np

class Action(basics.Action):

    def __init__(self, name, step=0.05):
        """
        name (string): The action name. We discretise our solution into finite steps with step
        length step. 
        """
        self.STEP = step
        # (x, y, theta, lift, extenstion, wrist_yaw, head_pan, head_tilt, gripper) w.r.t base_link (x points forward)
        self.MOTIONS = {"Forward": (self.STEP, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                        "Backward": (-self.STEP, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                        "Left_Face": (0, 0, np.pi/2, 0, 0, 0, 0, 0, 0, 0, 0),
                        "Right_Face": (0, 0, -np.pi/2, 0, 0, 0, 0, 0, 0, 0, 0),
                        "Slight_Left": (0, 0, np.pi/4, 0, 0, 0, 0, 0, 0, 0, 0),
                        "Slight_Right": (0, 0, -np.pi/4, 0, 0, 0, 0, 0, 0, 0, 0),
                        "Lift_Arm": (0, 0, 0, 0.095, 0, 0, 0, 0, 0, 0, 0), # Limit 1.1 meters
                        "Lower_Arm": (0, 0, 0, -0.095, 0, 0, 0, 0, 0, 0, 0), # Limit 0.15 meters
                        "Extend_Arm": (0, 0, 0, 0, 0.05, 0, 0, 0, 0, 0, 0), # Limit 0.5 meters
                        "Retract_Arm": (0, 0, 0, 0, -0.05, 0, 0, 0, 0, 0, 0), # Limit 0.0 meters
                        "Wrist_Clockwise": (0, 0, 0, 0, 0, -0.3, 0, 0, 0, 0, 0), # Limit -1.75 radians
                        "Wrist_CounterClockwise": (0, 0, 0, 0, 0, 0.3, 0, 0, 0, 0, 0), # Limit 4.0 radians
                        "Wrist_Up": (0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0),
                        "Wrist_Down": (0, 0, 0, 0, 0, 0, -0.1, 0, 0, 0, 0),
                        "Wrist_Roll_Left": (0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0),
                        "Wrist_Roll_Right": (0, 0, 0, 0, 0, 0, 0, -0.1, 0, 0, 0),
                        "Head_Clockwise": (0, 0, 0, 0, 0, 0, 0, 0, -0.19, 0, 0), # Limit -2.8 radians
                        "Head_CounterClockwise": (0, 0, 0, 0, 0, 0, 0, 0, 0.19, 0, 0), # Limit 2.9 radians
                        "Head_Up": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0), # Limit 0.4 radians
                        "Head_Down": (0, 0, 0, 0, 0, 0, 0, 0, 0, -0.1, 0), # Limit -1.6 radians
                        "Grip_In": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.1), # Limit -0.35 radians
                        "Grip_Out": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1)} # Limit 0.165 radians

        if name not in self.MOTIONS.keys():
            raise ValueError(f"Invalid action name: {name}")
        self._name = name
        self._motion = self.MOTIONS[name]
        self._hash = hash(self._name)

    @property
    def motion(self):
        return self._motion

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if isinstance(other, Action):
            return self._hash == other._hash
        return False

    def __str__(self):
        return f"<name: {self._name}>"

    def __repr__(self):
        return self.__str__()


class MacroAction(basics.MacroAction):
    def __init__(self, action_sequence):
        super().__init__(action_sequence)
        self.CHECK_FREQ = 1

    def __hash__(self):
        return 1

    def __eq__(self, other):
        if isinstance(other, MacroAction):
            if len(self.action_sequence) != len(other.action_sequence):
                return False
            return all([self.action_sequence[i] == other.action_sequence[i] for i in
                        range(0, len(self.action_sequence), self.CHECK_FREQ)])
        return False

    def __str__(self):
        return f"<action_sequence: {[str(a) for a in self._action_sequence]}>"

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self._action_sequence)