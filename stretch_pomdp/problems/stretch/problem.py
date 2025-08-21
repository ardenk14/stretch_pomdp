import numpy as np

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
    
    particles = [State(init_pos,
                danger_zone=vamp_env.dz_checker(init_pos),
                landmark=vamp_env.lm_checker(init_pos),
                goal=vamp_env.goal_checker(init_pos))]
    for i in range(particle_count):
        pos = np.random.normal(init_pos, [0.01, 0.01, 0.017, 0.0, 0., 0, 0, 0, 0, 0, 0, 0, 0])
        particles.append(State(pos,
                        danger_zone=vamp_env.dz_checker(pos),
                        landmark=vamp_env.lm_checker(pos),
                        goal=vamp_env.goal_checker(pos)))

    init_belief = pomdp_py.Particles(particles) #TODO: Add randomness

    # Problem setup
    # init state, init belief, vamp_env, planner
    problem = StretchPOMDP(init_state, init_belief, vamp_env, 'ref_solver_fast')
    return problem

