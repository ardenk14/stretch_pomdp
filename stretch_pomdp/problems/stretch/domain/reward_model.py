from pomdp_py.framework.basics import RewardModel
from stretch_pomdp.problems.stretch.domain.state import State
from stretch_pomdp.problems.stretch.domain.action import Action
import numpy as np


class StretchRewardModel(RewardModel):

    STEP_REWARD = -3.0 #-0.1 # The step penalty for non-terminal states.
    DZ_REWARD = -800.0#-800 # The collision penalty.
    GOAL_REWARD = 800.0  # The goal reward.

    def __init__(self, vamp_env):
        self._vamp_env = vamp_env

    def probability(self, reward, state, action, next_state):
        raise NotImplementedError

    def sample(self, state: State, action: Action, next_state: State):

        if state.terminal:
            return 0.0

        if next_state._danger_zone:
            return self.DZ_REWARD

        if next_state._goal:
            return self.GOAL_REWARD

        if np.linalg.norm(np.array(next_state._position)[:2] - np.array(self._vamp_env._goal[1])[:2]) < np.linalg.norm(np.array(state._position)[:2] - np.array(self._vamp_env._goal[1])[:2]):
           return self.STEP_REWARD + 2.0 
        return self.STEP_REWARD