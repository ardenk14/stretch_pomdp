import random
from pomdp_py.framework.basics import TransitionModel
from stretch_pomdp.problems.stretch.domain.action import Action
from stretch_pomdp.problems.stretch.domain.state import State
import numpy as np

np.set_printoptions(precision=3, suppress=True)  # for neat printing of numpy arrays.
np_generator = np.random.default_rng()


class StretchTransitionModel(TransitionModel):

    def __init__(self, vamp_env):
        self._vamp_env = vamp_env
        self.ACTIONS = [Action(i) for i in Action("F").MOTIONS.keys()]

        self.av = 0.12
        self.aw = 0.12

    # First segment: Accelerating motion
    def integrate_motion(self, v0, w0, a_v, a_w, theta0, t):
        # Heading: theta(t) = theta0 + w0*t + 0.5*a_w*t^2
        theta_t = theta0 + w0 * t + 0.5 * a_w * t**2
        # Use linearized closed form
        epsilon = 1e-6
        w0_ = w0 if abs(w0) > epsilon else epsilon

        dx = (v0 / w0_) * (np.sin(theta0 + w0_ * t) - np.sin(theta0)) \
        + (a_v / w0_**2) * (np.cos(theta0 + w0_ * t) - np.cos(theta0)) \
        + (a_v * t / w0_) * np.sin(theta0 + w0_ * t)

        dy = -(v0 / w0_) * (np.cos(theta0 + w0_ * t) - np.cos(theta0)) \
        + (a_v / w0_**2) * (np.sin(theta0 + w0_ * t) - np.sin(theta0)) \
        - (a_v * t / w0_) * np.cos(theta0 + w0_ * t)

        dtheta = w0 * t + 0.5 * a_w * t**2

        return dx, dy, dtheta, v0 + a_v * t, w0 + a_w * t, theta_t

    # Second segment: Constant velocity motion
    def constant_velocity_motion(self, v, w, theta_start, t):
        if abs(w) < 1e-6:
            # Straight line
            dx = v * t * np.cos(theta_start)
            dy = v * t * np.sin(theta_start)
        else:
            dx = (v / w) * (np.sin(theta_start + w * t) - np.sin(theta_start))
            dy = -(v / w) * (np.cos(theta_start + w * t) - np.cos(theta_start))

        dtheta = w * t
        return dx, dy, dtheta

    def move_if_valid_next_position(self, position, action):
        x0, y0, theta0, v0, w0 = position[0], position[1], position[2], position[11], position[12]
        v, w = action._motion[0], action._motion[1]
        T = action.T

        #print("T: ", T)
        #print("V: ", v)
        #print("V0: ", v0)
        #print("W: ", w)
        #print("W0: ", w0)

        if abs(v - v0) > 0.03:
            av = self.av * np.sign(v - v0)
            tv = min(abs((v - v0) / av), T)
        else:
            av = 0.0
            tv = T

        if abs(w - w0) > 0.03:
            aw = self.aw * np.sign(w - w0)
            tw = min(abs((w - w0) / aw), T)
        else:
            aw = 0.0
            tw = T

        # Determine acceleration directions (and if acceleration is necessary)
        #av = self.av * np.sign(v - v0) if abs(v-v0) > 0.03 else 0.0
        #aw = self.aw * np.sign(w - w0) if abs(w-w0) > 0.03 else 0.0

        # Time needed to reach desired v and w
        #tv = min(abs((v - v0) / av) if abs(v - v0) > 0.03 else T, T)
        #tw = min(abs((w - w0) / aw) if abs(w - w0) > 0.03 else T, T)

        # Time spent accelerating (may be partial)
        #print("AV: ", av)
        #print("AW: ", aw)
        #print("TV: ", tv)
        #print("TW: ", tw)
        t1 = min(tv, tw)
        t2 = tv-tw if tv > tw else tw-tv
        t3 = T - (t2 + t1)

        dx1, dy1, dtheta1, v1, w1, theta1 = self.integrate_motion(v0, w0, av, aw, theta0, t1)
        av = self.av * np.sign(v - v1)# if abs(v-v1) > 0.03 else 0.0
        aw = self.aw * np.sign(w - w1)# if abs(w-w1) > 0.03 else 0.0
        dx2, dy2, dtheta2, v2, w2, theta2 = self.integrate_motion(v1, w1, av, aw, theta1, t2)
        dx3, dy3, dtheta3 = self.constant_velocity_motion(v2, w2, theta2, t3)

        # Total change
        dx = dx1 + dx2 + dx3
        dy = dy1 + dy2 + dy3
        dtheta = dtheta1 + dtheta2 + dtheta3

        next_position = np.zeros_like(position)
        next_position[0] = x0 + dx
        next_position[1] = y0 + dy
        next_position[2] = theta0 + dtheta
        next_position[11] = v2 if t3 <= 0.0 else v  # if not enough time, we didn't reach v
        next_position[12] = w2 if t3 <= 0.0 else w

        if self._vamp_env.collision_checker(list(next_position)[:11] + [0., 0.]):
            return position
        return next_position

    def get_next_position(self, position, action):
        return self.move_if_valid_next_position(position, action)
    
    #def move_if_valid_next_position(self, position, action):
        """
        Transition function for the navigation model.
        :param position: agent current position (x,y,z, roll, pitch, yaw)
        :param action: The action to take.
        :return: The next state under environment constraints.
        """
        """action = action._motion
        next_position = np.zeros_like(position)
        x, y, yaw = position[0], position[1], position[2]

        dx_body = action[0] # (v*t + 0.5*self.av*t**2)*np.cos(yaw)
        dy_body = action[1]
        dyaw = action[2]

        dx_world = dx_body * np.cos(yaw) - dy_body * np.sin(yaw)
        dy_world = dx_body * np.sin(yaw) + dy_body * np.cos(yaw)

        next_position[0] = x + dx_world
        next_position[1] = y + dy_world
        next_position[2] = yaw + dyaw
        next_position[3:] = position[3:] + action[3:]
        if self._vamp_env.collision_checker(list(next_position) +[0., 0.]):
            return position
        return next_position"""

    def sample(self, state: State, action: Action):

        if state.terminal:
            return state

        realized_action = action
        #realised_action = random.choices([Action("None"), action], weights=[0.005, 0.995])[0] # TODO: Add randomness (action.sample())
        #realised_action = Action(realised_action._name, v_noise = np.random.normal(0, 0.05), w_noise = np.random.normal(0, 0.03))
        #state_pos = np.array(list(state.get_position) + [0., 0.])
        next_position = self.move_if_valid_next_position(state.get_position, realized_action)

        return State(next_position,
                     self._vamp_env.dz_checker(next_position),
                     self._vamp_env.lm_checker(next_position),
                     self._vamp_env.goal_checker(next_position))

    def get_all_actions(self):
        return self.ACTIONS

    def get_handcraft_macro_actions(self, macro_action_size=1):
        return [tuple([a] * macro_action_size) for a in self.ACTIONS]
