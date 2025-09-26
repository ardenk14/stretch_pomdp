import numpy as np

from stretch_pomdp.problems.stretch.domain.pomdp import StretchPOMDP
from stretch_pomdp.problems.stretch.domain.state import State
import pomdp_py
import rerun as rr

def init_stretch_pomdp(init_pos,
                       obstacle_loc,
                       vamp_env,
                       particle_count=200,
                       visualize=False):
    
    print(f"Initialising POMDPs, init_pos: {init_pos}, obstacle_loc: {obstacle_loc}")
    # Configuration
    # Initial state, initial belief (particles)
    init_state = State(init_pos,
                       obstacle_loc=obstacle_loc,
                       danger_zone=vamp_env.dz_checker(init_pos),
                       goal=vamp_env.goal_checker(init_pos))
    
    particles = []
    
    for i in range(particle_count):
        pos = np.random.normal(init_pos, [0.01, 0.01, 0.017, 0.0, 0., 0, 0, 0, 0, 0, 0, 0, 0])
        noise_obstacle_pos = np.random.normal(obstacle_loc, [0.1, 0.1])
        particles.append(State(pos,
                        obstacle_loc=noise_obstacle_pos,
                        danger_zone=vamp_env.dz_checker(pos),
                        goal=vamp_env.goal_checker(pos)))
    
    init_belief = pomdp_py.Particles(particles)

    # visualise initial belief
    for i, (k, v) in enumerate(init_belief.get_histogram().items()):
        robot_pos = list(k.get_position[:2]) + [0.]
        rr.log(["stretch_pos_belief", str(i)], rr.Points3D(robot_pos, radii=0.05, colors=[255, 247, 28]))
        obstacle_pos = list(k.get_obstacle_loc) + [0.]
        rr.log(["obstacle_pos_belief", str(i)], rr.Points3D(obstacle_pos, radii=vamp_env.sphere_approx_radius, colors=[28, 111, 255]))

    # Problem setup
    # init state, init belief, vamp_env, planner
    problem = StretchPOMDP(init_state, init_belief, vamp_env, 'ref_solver_fast')
    return problem

