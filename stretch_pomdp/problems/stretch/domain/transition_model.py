import random
from pomdp_py.framework.basics import TransitionModel
from stretch_pomdp.problems.stretch.domain.action import Action
from stretch_pomdp.problems.stretch.domain.state import State
import numpy as np
import math

import rerun as rr

np.set_printoptions(precision=3, suppress=True)  # for neat printing of numpy arrays.
np_generator = np.random.default_rng()


class StretchTransitionModel(TransitionModel):

    def __init__(self, vamp_env):
        self._vamp_env = vamp_env
        self.ACTIONS = [Action(i) for i in Action("F").MOTIONS.keys()]
        self.a = 0.12
        self.max_v = 0.3

        self.vrs = []
        self.vls = []
        self.ts = []
        self.true_t = 0

    def next(self, v0, w0, v_goal, w_goal, theta_start, T, dt=0.1):
        x, y, theta = 0.0, 0.0, theta_start
        L = 0.3196581671875644

        t = 0.0
        while t < T:

            # Get current linear velocity for each wheel
            vr = v0 + (w0*L)/2.0
            vr = np.sign(vr) * min(abs(vr), self.max_v)
            vl = v0 - (w0*L)/2.0
            vl = np.sign(vl) * min(abs(vl), self.max_v)

            # Get goal velocity for each wheel (with velocity limits)
            vrg = v_goal + (w_goal*L)/2.0
            vrg = np.sign(vrg) * min(abs(vrg), self.max_v)
            vlg = v_goal - (w_goal*L)/2.0
            vlg = np.sign(vlg) * min(abs(vlg), self.max_v)

            # Accelerate the right wheel according to trapezoidal motion profile (constant acceleration)
            if vr < vrg:
                vr = min(vr + self.a * dt, vrg)
            elif vr > vrg:
                vr = max(vr - self.a * dt, vrg)

            # Accelerate the left wheel according to trapezoidal motion profile (constant acceleration)
            if vl < vlg:
                vl = min(vl + self.a * dt, vlg)
            elif vl > vlg:
                vl = max(vl - self.a * dt, vlg)

            # Calculate actual linear and angular velocities given vel limits for each wheel
            v0 = (vr + vl) / 2.0
            w0 = (vr - vl) / L

            self.vrs.append(vr)
            self.vls.append(vl)
            self.ts.append(self.true_t)

            # If no acceleration, we have a closed form solution and end integration in one step here
            if vl == vlg and vr == vrg:
                dx, dy, dtheta = self.constant_velocity_motion(v0, w0, theta, T - t)
                theta = (theta + dtheta + math.pi) % (2 * math.pi) - math.pi
                return x+dx, y+dy, theta, v0, w0

            dl = vl * dt
            dr = vr * dt

            delta_travel = (dr + dl) / 2.0
            delta_theta = (dr - dl) / L

            if dl == dr:
                delta_x = delta_travel * math.cos(theta)
                delta_y = delta_travel * math.sin(theta)
            else:
                # calculate the instantaneous center of curvature (ICC)
                icc_radius = delta_travel / delta_theta
                icc_x = x - (icc_radius * math.sin(theta))
                icc_y = y + (icc_radius * math.cos(theta))

                # calculate the change in position based on the ICC
                delta_x = ((math.cos(delta_theta) * (x - icc_x))
                            - (math.sin(delta_theta) * (y - icc_y))
                            + icc_x - x)

                delta_y = ((math.sin(delta_theta) * (x - icc_x))
                            + (math.cos(delta_theta) * (y - icc_y))
                            + icc_y - y)

            # Update position
            x += delta_x#v0 * math.cos(theta) * dt
            y += delta_y#v0 * math.sin(theta) * dt
            theta += delta_theta#w0 * dt

            # Normalize final heading
            theta = (theta + math.pi) % (2 * math.pi) - math.pi

            t += dt
            self.true_t += dt
        return x, y, theta, v0, w0

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
        """
        Transition function for the navigation model.
        :param position: agent current position (x,y,z, roll, pitch, yaw)
        :param action: The action to take.
        :return: The next state under environment constraints.
        """
        x0, y0, theta0, v0, w0 = position[0], position[1], position[2], position[11], position[12]
        v, w = action._motion[0], action._motion[1]
        T = action.T

        dx, dy, theta, v, w = self.next(v0, w0, v, w, theta0, T)

        next_position = np.zeros_like(position)
        next_position[0] = x0 + dx
        next_position[1] = y0 + dy
        next_position[2] = theta
        next_position[3:11] = position[3:11]
        next_position[3] = 0.5
        next_position[11] = v
        next_position[12] = w

        # TODO: Ensure you don't have to throw out too many paths!
        """if self._vamp_env.collision_checker(list(next_position)[:11] + [0., 0.]):
            #print("COLLISION! ", list(next_position)[:11] + [0., 0.])
            #print("COLLISION! ", list(position))#[:11] + [0., 0.])
            #raise ValueError("AHHHHHH")
            #rr.log("Collisions", rr.Points3D([next_position[0], next_position[1], 0.0], radii=[0.05]))
            return position"""
        #rr.log("FREE", rr.Points3D([next_position[0], next_position[1], 0.0], radii=[0.05]))
        return next_position

    def get_next_position(self, position, action):
        return self.move_if_valid_next_position(position, action)

    def sample(self, state: State, action: Action):

        if state.terminal:
            return state

        realized_action = action
        realized_action = Action(realized_action._name, v_noise = np.random.normal(0, 0.03), w_noise = np.random.normal(0, 0.07))
        next_position = self.move_if_valid_next_position(state.get_position, realized_action)

        env = self._vamp_env.state_to_vamp(state)
        if self._vamp_env.collision_checker(list(next_position)[:11] + [0., 0.], env):
            print("collision!")
            next_position = state.get_position
        else:
            print("no collision")

        return State(next_position,
                     state.get_obstacle_loc,
                     self._vamp_env.dz_checker(next_position),
                     self._vamp_env.lm_checker(next_position),
                     self._vamp_env.goal_checker(next_position))

    def get_all_actions(self):
        return self.ACTIONS

    def get_handcraft_macro_actions(self, macro_action_size=1):
        return [tuple([a] * macro_action_size) for a in self.ACTIONS]


if __name__ == '__main__':
    model = StretchTransitionModel(None)

    x, y, theta, v, w = model.next(0, 0, 0.2, 0.2, 0.0, 3.0)
    n = model.next(v, w, 0.2, -0.2, theta, 3.0)

    import matplotlib.pyplot as plt
    plt.scatter(model.ts, model.vrs)
    plt.scatter(model.ts, model.vls)
    plt.show()