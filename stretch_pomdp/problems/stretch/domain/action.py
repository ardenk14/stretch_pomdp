from pomdp_py.framework import basics
import numpy as np

class Action(basics.Action):

    def __init__(self, name, V=0.3, T=2.0, v_noise = 0.0, w_noise = 0.0, step=1.0):
        """
        name (string): The action name. We discretise our solution into finite steps with step
        length step. 
        """
        self.STEP = step
        self.T = T
        self.V = V * 0.018/0.3 + v_noise #0.047 0.017
        self.W_adjustment = 0.045 + w_noise
        # (x, y, theta, lift, extenstion, wrist_yaw, head_pan, head_tilt, gripper) w.r.t base_link (x points forward)
        self.MOTIONS = {"None": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                        "Forward": (self.T*self.V, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                        "FL1": (self.get_x(0.25), self.get_y(0.25), 0.25*self.T*self.W_adjustment, 0, 0, 0, 0, 0, 0, 0, 0),
                        "FL2": (self.get_x(0.5), self.get_y(0.5), 0.5*self.T*self.W_adjustment, 0, 0, 0, 0, 0, 0, 0, 0),
                        "FL3": (self.get_x(0.75), self.get_y(0.75), 0.75*self.T*self.W_adjustment, 0, 0, 0, 0, 0, 0, 0, 0),
                        "FL4": (self.get_x(1.0), self.get_y(1.0), 1.0*self.T*self.W_adjustment, 0, 0, 0, 0, 0, 0, 0, 0),
                        "FL5": (self.get_x(1.25), self.get_y(1.25), 1.25*self.T*self.W_adjustment, 0, 0, 0, 0, 0, 0, 0, 0),
                        "FR1": (self.get_x(-0.25), self.get_y(-0.25), -0.25*self.T*self.W_adjustment, 0, 0, 0, 0, 0, 0, 0, 0),
                        "FR2": (self.get_x(-0.5), self.get_y(-0.5), -0.5*self.T*self.W_adjustment, 0, 0, 0, 0, 0, 0, 0, 0),
                        "FR3": (self.get_x(-0.75), self.get_y(-0.75), -0.75*self.T*self.W_adjustment, 0, 0, 0, 0, 0, 0, 0, 0),
                        "FR4": (self.get_x(-1.0), self.get_y(-1.0), -1.0*self.T*self.W_adjustment, 0, 0, 0, 0, 0, 0, 0, 0),
                        "FR5": (self.get_x(-1.25), self.get_y(-1.25), -1.25*self.T*self.W_adjustment, 0, 0, 0, 0, 0, 0, 0, 0),
                        #"FL15": (self.get_x_at_angle(np.pi/12), self.get_y_at_angle(np.pi/12), np.pi/12, 0, 0, 0, 0, 0, 0, 0, 0),
                        #"FL7": (self.get_x_at_angle(np.pi/24), self.get_y_at_angle(np.pi/24), np.pi/24, 0, 0, 0, 0, 0, 0, 0, 0),
                        #"FR30": (self.get_x_at_angle(-np.pi/6), self.get_y_at_angle(-np.pi/6), -np.pi/6, 0, 0, 0, 0, 0, 0, 0, 0),
                        #"FR15": (self.get_x_at_angle(-np.pi/12), self.get_y_at_angle(-np.pi/12), -np.pi/12, 0, 0, 0, 0, 0, 0, 0, 0),
                        #"FR7": (self.get_x_at_angle(-np.pi/24), self.get_y_at_angle(-np.pi/24), -np.pi/24, 0, 0, 0, 0, 0, 0, 0, 0),
                        #"FL30z": (self.STEP*np.cos(np.pi/6), self.STEP*np.sin(np.pi/6), 0, 0, 0, 0, 0, 0, 0, 0, 0),
                        #"FL15z": (self.STEP*np.cos(np.pi/12), self.STEP*np.sin(np.pi/12), 0, 0, 0, 0, 0, 0, 0, 0, 0),
                        #"FR30z": (self.STEP*np.cos(-np.pi/6), self.STEP*np.sin(-np.pi/6), 0, 0, 0, 0, 0, 0, 0, 0, 0),
                        #"FR15z": (self.STEP*np.cos(-np.pi/12), self.STEP*np.sin(-np.pi/12), 0, 0, 0, 0, 0, 0, 0, 0, 0),
                        "Backward": (-self.V*self.T, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                        "BL1": (-self.get_x(0.25), self.get_y(0.25), -0.25*self.T*self.W_adjustment, 0, 0, 0, 0, 0, 0, 0, 0),
                        "BL2": (-self.get_x(0.5), self.get_y(0.5), -0.5*self.T*self.W_adjustment, 0, 0, 0, 0, 0, 0, 0, 0),
                        "BL3": (-self.get_x(0.75), self.get_y(0.75), -0.75*self.T*self.W_adjustment, 0, 0, 0, 0, 0, 0, 0, 0),
                        "BL4": (-self.get_x(1.0), self.get_y(1.0), -1.0*self.T*self.W_adjustment, 0, 0, 0, 0, 0, 0, 0, 0),
                        "BL5": (-self.get_x(1.25), self.get_y(1.25), -1.25*self.T*self.W_adjustment, 0, 0, 0, 0, 0, 0, 0, 0),
                        "BR1": (-self.get_x(-0.25), self.get_y(-0.25), 0.25*self.T*self.W_adjustment, 0, 0, 0, 0, 0, 0, 0, 0),
                        "BR2": (-self.get_x(-0.5), self.get_y(-0.5), 0.5*self.T*self.W_adjustment, 0, 0, 0, 0, 0, 0, 0, 0),
                        "BR3": (-self.get_x(-0.75), self.get_y(-0.75), 0.75*self.T*self.W_adjustment, 0, 0, 0, 0, 0, 0, 0, 0),
                        "BR4": (-self.get_x(-1.0), self.get_y(-1.0), 1.0*self.T*self.W_adjustment, 0, 0, 0, 0, 0, 0, 0, 0),
                        "BR5": (-self.get_x(-1.25), self.get_y(-1.25), 1.25*self.T*self.W_adjustment, 0, 0, 0, 0, 0, 0, 0, 0),
                        #"BL30": (-self.get_x_at_angle(-np.pi/6), -self.get_y_at_angle(-np.pi/6), -np.pi/6, 0, 0, 0, 0, 0, 0, 0, 0),
                        #"BL15": (-self.get_x_at_angle(-np.pi/12), -self.get_y_at_angle(-np.pi/12), -np.pi/12, 0, 0, 0, 0, 0, 0, 0, 0),
                        #"BL7": (-self.get_x_at_angle(-np.pi/24), -self.get_y_at_angle(-np.pi/24), -np.pi/24, 0, 0, 0, 0, 0, 0, 0, 0),
                        #"BR30": (-self.get_x_at_angle(np.pi/6), -self.get_y_at_angle(np.pi/6), np.pi/6, 0, 0, 0, 0, 0, 0, 0, 0),
                        #"BR15": (-self.get_x_at_angle(np.pi/12), -self.get_y_at_angle(np.pi/12), np.pi/12, 0, 0, 0, 0, 0, 0, 0, 0),
                        #"BR7": (-self.get_x_at_angle(np.pi/24), -self.get_y_at_angle(np.pi/24), np.pi/24, 0, 0, 0, 0, 0, 0, 0, 0),
                        #"BL30z": (-self.STEP*np.cos(np.pi/6), self.STEP*np.sin(np.pi/6), 0, 0, 0, 0, 0, 0, 0, 0, 0),
                        #"BL15z": (-self.STEP*np.cos(np.pi/12), self.STEP*np.sin(np.pi/12), 0, 0, 0, 0, 0, 0, 0, 0, 0),
                        #"BR30z": (-self.STEP*np.cos(-np.pi/6), self.STEP*np.sin(-np.pi/6), 0, 0, 0, 0, 0, 0, 0, 0, 0),
                        #"BR15z": (-self.STEP*np.cos(-np.pi/12), self.STEP*np.sin(-np.pi/12), 0, 0, 0, 0, 0, 0, 0, 0, 0),
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
        
        """self.MOTIONS = {"None": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                        "Forward": (self.STEP, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                        "FL30": (self.STEP*np.cos(np.pi/6), self.STEP*np.sin(np.pi/6), np.pi/6, 0, 0, 0, 0, 0, 0, 0, 0),
                        "FL15": (self.STEP*np.cos(np.pi/12), self.STEP*np.sin(np.pi/12), np.pi/12, 0, 0, 0, 0, 0, 0, 0, 0),
                        "FR30": (self.STEP*np.cos(-np.pi/6), self.STEP*np.sin(-np.pi/6), -np.pi/6, 0, 0, 0, 0, 0, 0, 0, 0),
                        "FR15": (self.STEP*np.cos(-np.pi/12), self.STEP*np.sin(-np.pi/12), -np.pi/12, 0, 0, 0, 0, 0, 0, 0, 0),
                        "FL30z": (self.STEP*np.cos(np.pi/6), self.STEP*np.sin(np.pi/6), 0, 0, 0, 0, 0, 0, 0, 0, 0),
                        "FL15z": (self.STEP*np.cos(np.pi/12), self.STEP*np.sin(np.pi/12), 0, 0, 0, 0, 0, 0, 0, 0, 0),
                        "FR30z": (self.STEP*np.cos(-np.pi/6), self.STEP*np.sin(-np.pi/6), 0, 0, 0, 0, 0, 0, 0, 0, 0),
                        "FR15z": (self.STEP*np.cos(-np.pi/12), self.STEP*np.sin(-np.pi/12), 0, 0, 0, 0, 0, 0, 0, 0, 0),
                        "Backward": (-self.STEP, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                        "BL30": (-self.STEP*np.cos(np.pi/6), self.STEP*np.sin(np.pi/6), -np.pi/6, 0, 0, 0, 0, 0, 0, 0, 0),
                        "BL15": (-self.STEP*np.cos(np.pi/12), self.STEP*np.sin(np.pi/12), -np.pi/12, 0, 0, 0, 0, 0, 0, 0, 0),
                        "BR30": (-self.STEP*np.cos(-np.pi/6), self.STEP*np.sin(-np.pi/6), np.pi/6, 0, 0, 0, 0, 0, 0, 0, 0),
                        "BR15": (-self.STEP*np.cos(-np.pi/12), self.STEP*np.sin(-np.pi/12), np.pi/12, 0, 0, 0, 0, 0, 0, 0, 0),
                        "BL30z": (-self.STEP*np.cos(np.pi/6), self.STEP*np.sin(np.pi/6), 0, 0, 0, 0, 0, 0, 0, 0, 0),
                        "BL15z": (-self.STEP*np.cos(np.pi/12), self.STEP*np.sin(np.pi/12), 0, 0, 0, 0, 0, 0, 0, 0, 0),
                        "BR30z": (-self.STEP*np.cos(-np.pi/6), self.STEP*np.sin(-np.pi/6), 0, 0, 0, 0, 0, 0, 0, 0, 0),
                        "BR15z": (-self.STEP*np.cos(-np.pi/12), self.STEP*np.sin(-np.pi/12), 0, 0, 0, 0, 0, 0, 0, 0, 0),
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
                        "Grip_Out": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1)} # Limit 0.165 radians"""

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
    
    def get_x(self, w):
        w *= self.W_adjustment
        return (self.V / w)*np.sin(w*self.T)
    
    def get_y(self, w):
        w *= self.W_adjustment
        return (self.V / w)*(1-np.cos(w*self.T))


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