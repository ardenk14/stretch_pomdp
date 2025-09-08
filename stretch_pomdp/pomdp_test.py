import pomdp_py
import numpy as np
from stretch_pomdp.problems.stretch.problem import init_stretch_pomdp
from stretch_pomdp.problems.stretch.environments.vamp_template import VAMPEnv

class StretchPOMDP:
    def __init__(self):
        self.current_config = np.array([0., 0., 0., 0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.vamp_env = VAMPEnv()
        self.problem = init_stretch_pomdp(self.current_config, self.vamp_env)
        self.planner = pomdp_py.ROPRAS3(
            planning_time=2,
            max_depth=100,
            rollout_depth=450,
            eta=0.2,
            ref_policy_heuristic='uniform',
            use_prm=False
        )
        print("finished problem initialisations.")

    def plan(self):
        action = self.planner.plan(self.problem.agent, no_pomdp=False)
        return action

    def random_action_executions(self, n = 3):
        while n > 0:
            action = self.problem.agent.policy_model.sample(self.problem.env.cur_state)
            ns, r, _ = self.problem.env.state_transition(action, execute=False)
            self.problem.env.apply_transition(ns)
            real_observation = self.problem.agent.observation_model.sample(self.problem.env.state, action)
            print(
                f"---\n"
                f"Action: {action}\n"
                f"Observation: {real_observation}\n"
                f"Reward: {r}\n"
                f"State: {self.problem.env.state}")
            n-=1
        return



def main(args=None):
    stretch_pomdp = StretchPOMDP()
    stretch_pomdp.plan()

if __name__ == '__main__':
    main()
