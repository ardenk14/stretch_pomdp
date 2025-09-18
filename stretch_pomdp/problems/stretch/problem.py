import numpy as np

from stretch_pomdp.problems.stretch.domain.pomdp import StretchPOMDP
from stretch_pomdp.problems.stretch.domain.state import State
import pomdp_py
import rerun as rr

def init_stretch_pomdp(init_pos,
                       obstacle_loc,
                       vamp_env,
                       particle_count=1000,
                       visualize=False):
    # Configuration
    # Initial state, initial belief (particles)
    init_state = State(init_pos,
                       obstacle_loc=obstacle_loc,
                       danger_zone=vamp_env.dz_checker(init_pos),
                       landmark=vamp_env.lm_checker(init_pos),
                       goal=vamp_env.goal_checker(init_pos))
    
    particles = []
    
    for i in range(particle_count):
        pos = np.random.normal(init_pos, [0.01, 0.01, 0.017, 0.0, 0., 0, 0, 0, 0, 0, 0, 0, 0])
        particles.append(State(pos,
                        obstacle_loc=obstacle_loc,
                        danger_zone=vamp_env.dz_checker(pos),
                        landmark=vamp_env.lm_checker(pos),
                        goal=vamp_env.goal_checker(pos)))
    

    init_belief = pomdp_py.Particles(particles)

    # visualise belief
    # print(f"visualising initial beliefs")
    positions = []
    obstacle_positions = []
    for k, v in init_belief.get_histogram().items():
        positions.append(list(k.get_position[:2]) + [0.0])
        obstacle_positions.append(list(k.get_obstacle_loc))
    rr.log("current_pos_belief", rr.Points3D(positions))
    sphere_radius = [vamp_env.sphere_approx_radius] * len(obstacle_positions)
    # cylinder_length = [vamp_env.cylinder_height] * len(obstacle_positions)
    # rot = rr.RotationAxisAngle(angle=vamp_env.cylinder_euler)
    # cylinder_euler = [rot] * len(obstacle_positions)
    rr.log("current_obs_belief", rr.Points3D(positions=obstacle_positions, radii=sphere_radius))
    

    # Problem setup
    # init state, init belief, vamp_env, planner
    problem = StretchPOMDP(init_state, init_belief, vamp_env, 'ref_solver_fast')
    return problem

