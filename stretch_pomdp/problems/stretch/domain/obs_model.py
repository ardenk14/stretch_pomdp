from pomdp_py.framework.basics import ObservationModel
from stretch_pomdp.problems.stretch.domain.observation import Observation
from stretch_pomdp.problems.stretch.domain.state import State
from stretch_pomdp.problems.stretch.domain.action import Action
#from pomdp_py.utils.transformations import normal_log_prob
import numpy as np

def normal_log_prob(x, mean, std):
    """
    Compute the log probability of x under a normal distribution N(mean, std^2).

    :param x: value or array of values
    :param mean: mean of the normal distribution
    :param std: standard deviation (must be > 0)
    :return: log probability of x under N(mean, std^2)
    """
    var = std ** 2
    log_prob = -0.5 * np.log(2 * np.pi * var) - ((x - mean) ** 2) / (2 * var)
    return log_prob

class StretchObservationModel(ObservationModel):
    """
    The navigation observation model.
    """
    def __init__(self, vamp_env):
        self._vamp_env = vamp_env
        self.obs_noise = 0.1#0.2

    def probability(self, observation: Observation, next_state: State, action: Action):
        """
        Given an observation, compute its probability according to the observation model.
        Specifically, we compute the log normal likelihood of observation.
        :param observation: current observation received by the agent.
        :param next_state: The state the agent just moved into that generates the observation.
        :param action: The action the agent just took to move to the state.
        :return: Probability of the observation
        """
        log_prob = 0
        if observation.get_reading is None:
            log_prob += -1000000
        else:
            x = observation.get_reading[0]
            y = observation.get_reading[1]
            yaw = observation.get_reading[2]
            log_prob += normal_log_prob(x, next_state.get_position[0], self.obs_noise)
            log_prob += normal_log_prob(y, next_state.get_position[1], self.obs_noise)
            log_prob += normal_log_prob(yaw, next_state.get_position[2], self.obs_noise)
        return log_prob

    def sample(self, next_state: State, action: Action):

        reading = next_state.get_position[:3] + np.random.normal(0, self.obs_noise, 3)

        return Observation(pos_reading=tuple(reading))
