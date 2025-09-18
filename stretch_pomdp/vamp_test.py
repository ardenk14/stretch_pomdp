import vamp
# import rerun as rr
import numpy as np
from copy import deepcopy

class TestVAMPEnv():
    def __init__(self):
        self.initial_robot_config = np.array([0., 0., 0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.goal = (1.0, [2.5, 0.0, 0.0, 0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.env = vamp.Environment()

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

    
def main():
    e = TestVAMPEnv()
    validate_stat = e.validate_test()
    print(f"collision percentage in EMPTY vamp scene (stretch validate), expect 0 but got: {validate_stat}")
    validity_stat = e.validity_test()
    print(f"collision percentage in EMPTY vamp scene (sphere_validity), expect 0 but got {validity_stat}")
    should_collid = e.actual_obstacle_validity()
    print(f"using sphere validity, this one should collide: {should_collid}")
    return 

if __name__ == "__main__":
    main()

