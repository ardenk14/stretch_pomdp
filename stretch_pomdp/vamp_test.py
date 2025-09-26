import vamp
import rerun as rr
import numpy as np
from copy import deepcopy
import math


class TestVAMPEnv():
    def __init__(self):
        self.initial_robot_config = np.array([0., 0., 0., 0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.goal = (1.0, [2.5, 0.0, 0.0, 0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.env = vamp.Environment()

        # test get_look_ahead_points used in pathToMacroActions
        self.test_vamp_path = [[-0.8950363993644714, -1.413517951965332, 1.5770461559295654, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [-0.6030513048171997, -0.6213897466659546, 1.3676095008850098, 0.47584548592567444, 0.006380943115800619, 0.005623701959848404, 0.007062612567096949, 0.007265507243573666, 0.08625435829162598, 0.07867830991744995, -0.02798696979880333, -0.08034594357013702, -0.11594691127538681],
                               [-0.311066210269928, 0.17073842883110046, 1.158172845840454, 0.4516909718513489, 0.012761886231601238, 0.011247403919696808, 0.014125225134193897, 0.014531014487147331, 0.17250871658325195, 0.1573566198348999, -0.05597393959760666, -0.16069188714027405, -0.23189382255077362],
                               [-0.4094286561012268, 0.4720119535923004, 1.2266830205917358, 0.4597424864768982, 0.010634904727339745, 0.009372835978865623, 0.011771021410822868, 0.012109179049730301, 0.14375725388526917, 0.13113051652908325, -0.04664494842290878, -0.1339099109172821, -0.1932448446750641],
                               [-0.5077911019325256, 0.773285448551178, 1.2951931953430176, 0.4677940011024475, 0.008507924154400826, 0.007498268969357014, 0.009416816756129265, 0.009687342680990696, 0.11500580608844757, 0.1049044132232666, -0.037315960973501205, -0.10712791979312897, -0.15459588170051575]]
        

    def validate_test(self, n = 30):
        count = 0
        i = 0
        while i < 30:
            x, y = np.random.uniform(low=-1, high = 1.0, size=2)
            sampled_config = np.array(self.initial_robot_config)
            sampled_config[0] = x
            sampled_config[1] = y
            if not vamp.stretch.validate(list(sampled_config), self.env):
                count += 1
            i += 1

        return count / n
    
    def validity_test(self, n = 30):
        count = 0
        
        i = 0
        while i < 30:
            x, y = np.random.uniform(low=-1, high = 1.0, size=2)
            sampled_config = np.array(self.initial_robot_config)
            sampled_config[0] = x
            sampled_config[1] = y
            colliding_jnts = vamp.stretch.sphere_validity(list(sampled_config), self.env)
            if len(colliding_jnts[0]) > 0:
                count += 1
            i += 1

        return count / n
    
    def actual_obstacle_validity(self):
        self.env.add_sphere(vamp.Sphere((1, 0, 0), 0.2))
        collision_config = np.array(self.initial_robot_config)
        collision_config[0] = 1.
        colliding_jnts = vamp.stretch.sphere_validity(list(collision_config), self.env)
        if len(colliding_jnts):
            return True 
        return False
    
    def get_lookahead_point(self, path, current_pos, lookahead_dist):
        """
        Finds a point `lookahead_dist` ahead of the closest point on the path to current_pos.
        Guarantees forward progress along the path.
        """
        pos = np.array(current_pos[:2])
        closest_dist = float('inf')
        closest_proj = None
        closest_idx = None
        closest_t = 0.0  # relative position along the segment (0=start, 1=end)

        # Step 1: find closest point on path
        for i in range(len(path) - 1):
            a = np.array(path[i][:2])
            b = np.array(path[i + 1][:2])
            ab = b - a
            ab_len = np.linalg.norm(ab)
            if ab_len < 1e-6:
                continue

            ap = pos - a
            t = np.clip(np.dot(ap, ab) / (ab_len ** 2), 0.0, 1.0)
            proj = a + t * ab
            dist = np.linalg.norm(proj - pos)

            if dist < closest_dist:
                closest_dist = dist
                closest_proj = proj
                closest_idx = i
                closest_t = t

        if closest_proj is None:
            # fallback
            end = path[-1]
            return (end[0], end[1], end[2])

        # Step 2: walk forward along the path from closest point
        remaining = lookahead_dist
        i = closest_idx
        t = closest_t
        curr_pt = closest_proj

        while i < len(path) - 1:
            a = np.array(path[i][:2])
            b = np.array(path[i + 1][:2])
            ab = b - a
            ab_len = np.linalg.norm(ab)

            if i == closest_idx:
                seg_remain = (1.0 - t) * ab_len
                walk_dir = (b - a) / ab_len
                offset = walk_dir * ((1.0 - t) * ab_len)
                base = a + t * ab
            else:
                seg_remain = ab_len
                walk_dir = ab / ab_len
                base = a

            if remaining <= seg_remain:
                lookahead = base + walk_dir * remaining
                yaw = math.atan2(walk_dir[1], walk_dir[0])
                return (lookahead[0], lookahead[1], yaw)

            remaining -= seg_remain
            i += 1

        # If we run out of path, just return the final point
        end = path[-1]
        return (end[0], end[1], end[2])

    
### Test Modules Below ###
    
def simple_collision_test():
    # general collision test
    print("Tests begin...")
    e = TestVAMPEnv()
    validate_stat = e.validate_test()
    print(f"collision percentage in EMPTY vamp scene (stretch validate), expect 0 but got: {validate_stat}")
    validity_stat = e.validity_test()
    print(f"collision percentage in EMPTY vamp scene (sphere_validity), expect 0 but got {validity_stat}")
    should_collid = e.actual_obstacle_validity()
    print(f"using sphere validity, this one should collide: {should_collid}")

def simple_rrtc_test():
    from stretch_pomdp.problems.stretch.domain.path_planner import PathPlanner
    from stretch_pomdp.problems.stretch.environments.vamp_template import VAMPEnv
    from pathlib import Path
    rr.init("vamp test")
    server_uri = rr.serve_grpc()
    rr.serve_web_viewer(connect_to=server_uri)
    # test vamp template
    #lab_map_dir = Path(__file__).parent / "lab.png"
    env = VAMPEnv(lab_map_dir=None)
    env.visualize_key_features()

    pp = PathPlanner(env)
    source = [-0.84, -1.26, 0.0, 0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    target = [1.5, -0., 0.0, 0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    s_collision = env.collision_checker(source)
    t_collision = env.collision_checker(target)
    if s_collision or t_collision:
        print(f"collision of source {s_collision} or target {t_collision}")
    else:
        print("no collision")

    path = pp.shortest_path(source, target, vamp_env=env.env)
    print(path)
    # # env.pb_visualiser()
    while True:
        ...

def lookup_point_test():
    rr.init("interpolation test")
    server_uri = rr.serve_grpc()
    rr.serve_web_viewer(connect_to=server_uri)
    e = TestVAMPEnv()

    # visualise the path to be interpolated
    for i in range(len(e.test_vamp_path) - 1):
        origin = list(e.test_vamp_path[i])[:2] + [0.]
        next = list(e.test_vamp_path[i+1])[:2] + [0.]
        vector = np.array(next) - np.array(origin)
        rr.log(["path", str(i)], rr.Arrows3D(origins=origin, vectors=vector))

    # assume the current position is at the start of the path with some noises
    current_pos = e.test_vamp_path[0]
    current_pos = np.random.normal(current_pos, [0.1, 0.1, 0., 0., 0., 0, 0, 0, 0, 0, 0, 0, 0])
    init_pos = list(current_pos)[:2] + [0.]
    rr.log("init_pos", rr.Points3D(init_pos, radii=0.05))

    # get the next point ahead
    for i in range(5):
        next_p = e.get_lookahead_point(e.test_vamp_path, current_pos, lookahead_dist=0.5)
        next_pos = list(next_p[:2]) + [0.]
        rr.log(["next_p", str(i)], rr.Points3D(next_pos, radii=0.05))
        current_pos = np.random.normal(next_p, [0.1, 0.1, 0.])
        rr.log(["current_p", str(i)], rr.Points3D(list(current_pos)[:2] + [0.], radii=0.05))


    while True:
        ...

    
def main():
    lookup_point_test()

    

if __name__ == "__main__":
    main()

