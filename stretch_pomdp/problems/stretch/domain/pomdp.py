from pomdp_py.framework import basics
from stretch_pomdp.problems.stretch.domain.transition_model import StretchTransitionModel
from stretch_pomdp.problems.stretch.domain.obs_model import StretchObservationModel
from stretch_pomdp.problems.stretch.domain.reference_policy_model import StretchReferencePolicyModel
from stretch_pomdp.problems.stretch.domain.reward_model import StretchRewardModel
import pathlib

class StretchPOMDP(basics.MPPOMDP):
    def __init__(self,
                 init_state,
                 init_belief,
                 vamp_env,
                 planner,
                 visual_shortest_path=False,
                 step=0.1,
                 macro_action_size=1,
                 discount_factor=.99,
                 finite_ref_actions = False):
        self.macro_action_size = macro_action_size
        self._vamp_env = vamp_env
        self.planner = planner
        self.finite_ref_actions = finite_ref_actions
        a, e = self.init_models(self._vamp_env, self.planner, init_belief, init_state)

        super().__init__(a, e, name="Motion Planning POMDP")

    def visualise_traj(self, x, y):
        self._vamp_env.traj_plot(x, y)

    def visualize_world(self):
        self._vamp_env.set_config(self.env._cur_state._position)
        robot_id = self._vamp_env._sim.loadURDF(
            str((pathlib.Path(__file__).parent.parent / "data/helicopter_small.urdf")),
            self.env._cur_state._position[:3],
            self._vamp_env._sim.getQuaternionFromEuler((0, 0, 0)),
            globalScaling=.8
        )
        self._vamp_env._sim.changeVisualShape(robot_id, -1, rgbaColor=[0, 0, 0, 0.3])

    def visualize_belief(self):
        pass
        #for b in self.agent.belief:
        #    self._vamp_env.set_config(b.get_position())

    def init_models(self, vamp_env, planner, init_belief, init_state):
        Tm = StretchTransitionModel(vamp_env)
        Zm = StretchObservationModel(vamp_env=vamp_env)
        if planner in ['ref_solver_fast', 'ref_solver_nips']:
            self.Pm = StretchReferencePolicyModel(vamp_env)
        else:
            raise NotImplementedError(f"{planner} cannot be found for the corresponding policy model.")

        Rm = StretchRewardModel(vamp_env)

        "Agent"
        a = basics.Agent(init_belief=init_belief,
                             policy_model=self.Pm,
                             transition_model=Tm,
                             observation_model=Zm,
                             reward_model=Rm,
                             ref_policy_model=None,
                             blackbox_model=None,
                             macro_action_size=self.macro_action_size)

        "Environment"
        e = basics.Environment(
            init_state=init_state,
            transition_model=Tm,
            reward_model=Rm)

        return a, e

    def reset(self):
        """
        Generate a new maze 3D Hard environment and reset everything to it.
        There is no domain randomisation for the 2d Maze
        """
        raise NotImplementedError
        #print("reset environment...")
        """if self._vamp_env.get_env_name == "maze3d":
            self._vamp_env = vmp_maze.VAMPMaze3dHard(gui=self._gui)

            init_pos = self._vamp_env.get_robot_init_config
            init_state = State(init_pos,
                               danger_zone=self._vamp_env.dz_checker(init_pos),
                               landmark=self._vamp_env.lm_checker(init_pos),
                               goal=self._vamp_env.goal_checker(init_pos))
            possible_states_x = np.random.normal(init_pos[0], self._vamp_env.get_start_std, 1000)
            possible_states_y = np.random.normal(init_pos[1], self._vamp_env.get_start_std, 1000)
            possible_states_z = np.random.normal(init_pos[2], self._vamp_env.get_start_std, 1000)
            possible_states_z = np.clip(possible_states_z, -2, 2)
            possible_states = np.zeros((1000, 6))
            possible_states[:, 0] += possible_states_x
            possible_states[:, 1] += possible_states_y
            possible_states[:, 2] += possible_states_z
            init_belief = pomdp_py.Particles([State(tuple(pos),
                                                    danger_zone=self._vamp_env.dz_checker(pos),
                                                    landmark=self._vamp_env.lm_checker(pos),
                                                    goal=self._vamp_env.goal_checker(pos)) for pos in possible_states])
            a, e = self.init_models(self._vamp_env, self.planner, init_belief, init_state)
            self.agent = a
            self.env = e
        else:
            self.agent.reset()
            self.env.reset()"""


    def __str__(self):
        return "Navigation POMDP Problem"