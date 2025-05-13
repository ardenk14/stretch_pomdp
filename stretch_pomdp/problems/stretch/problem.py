from stretch_pomdp.problems.stretch.domain.pomdp import StretchPOMDP
from stretch_pomdp.problems.stretch.domain.state import State
import pomdp_py

def init_stretch_pomdp(init_pos, vamp_env, particle_count=1000, visualize=False):
    # Configuration
    # Initial state, initial belief (particles)
    init_state = State(init_pos,
                        danger_zone=vamp_env.dz_checker(init_pos),
                        landmark=vamp_env.lm_checker(init_pos),
                        goal=vamp_env.goal_checker(init_pos))

    init_belief = pomdp_py.Particles([State(init_pos,
                                            danger_zone=vamp_env.dz_checker(init_pos),
                                            landmark=vamp_env.lm_checker(init_pos),
                                            goal=vamp_env.goal_checker(init_pos))] * particle_count)

    # Problem setup
    # init state, init belief, vamp_env, planner
    problem = StretchPOMDP(init_state, init_belief, vamp_env, 'ref_solver_fast')
    return problem

