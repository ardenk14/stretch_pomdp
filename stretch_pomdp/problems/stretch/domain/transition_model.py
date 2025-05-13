from pomdp_py.framework.basics import TransitionModel
from stretch_pomdp.problems.stretch.domain.action import Action
from stretch_pomdp.problems.stretch.domain.state import State
import numpy as np

np.set_printoptions(precision=3, suppress=True)  # for neat printing of numpy arrays.
np_generator = np.random.default_rng()


class StretchTransitionModel(TransitionModel):

    def __init__(self, vamp_env):
        self._vamp_env = vamp_env
        self.ACTIONS = [Action(i) for i in Action("Forward").MOTIONS.keys()]

    def move_if_valid_next_position(self, position, action):
        """
        Transition function for the navigation model.
        :param position: agent current position (x,y,z, roll, pitch, yaw)
        :param action: The action to take.
        :return: The next state under environment constraints.
        """
        action = action._motion
        next_position = np.zeros(position.shape)
        next_position[2:] = position[2:] + np.array(action)[2:]
        next_position[0] = position[0] + action[0] * np.cos(position[2])
        next_position[1] = position[1] + action[0] * np.sin(position[2])
        if self._vamp_env.collision_checker(list(next_position) +[0., 0.]):
            return position
        return next_position

    def sample(self, state: State, action: Action):

        if state.terminal:
            return state

        realised_action = action # TODO: Add randomness (action.sample())
        #state_pos = np.array(list(state.get_position) + [0., 0.])
        next_position = self.move_if_valid_next_position(state.get_position, realised_action)

        return State(next_position,
                     self._vamp_env.dz_checker(next_position),
                     self._vamp_env.lm_checker(next_position),
                     self._vamp_env.goal_checker(next_position))

    def get_all_actions(self):
        return self.ACTIONS

    def get_handcraft_macro_actions(self, macro_action_size=1):
        return [tuple([a] * macro_action_size) for a in self.ACTIONS]
