import vamp
import rerun as rr
import time

class PathPlanner:
    def __init__(self, vamp_env):
        self.vmp = vamp_env
        self._vamp_env = vamp_env._env
        (self.vamp_module, self.planner_func, self.plan_settings,
         self.simp_settings) = vamp.configure_robot_and_planner_with_kwargs("stretch", "rrtc")

        self.plan_settings.max_iterations = 1000
        self.sampler = getattr(self.vamp_module, "halton")()
        self.sampler.skip(0)

    def shortest_path(self, source, target, vamp_env = None, max_iterations=100000, restarts=0):
        #print("PLANNING")
        t1 = time.time()
        vamp_env = self._vamp_env if vamp_env is None else vamp_env 
        result = self.planner_func(source, target, vamp_env, self.plan_settings, self.sampler)
        #print("DONE: ", time.time() - t1)

        if time.time() - t1 > 1.0:
            print("PLANNING TOOK TOO LONG: ", time.time() - t1)
            print("SOURCE: ", source)
            print("TARGET: ", target)
        
        for i in range(restarts):
            if len(result.path) > 0:
                break

            print(f"Trying restart attempt #{i}...")
            result = self.planner_func(source, target, self._vamp_env, self.plan_settings, self.sampler)

        if result is None or result.size == 0:
            print(f"Warning: RRT-C failed to find a path despite {restarts} attempts!")
            return []

        simple = self.vamp_module.simplify(result.path, self._vamp_env, self.simp_settings, self.sampler)

        # path = [s.to_list()[:2]+[0.0] for s in simple.path]
        # src = source[:2] + [0.0]
        # trgt = target[:2] + [0.0]
        #print("PATH: ", path)
        #print("SRC: ", src)
        #print("TRGT: ", trgt)
        # rr.log("VAMP", rr.LineStrips3D(path))#, rr.Points3D([src, trgt]))

        #simple.path.interpolate(vamp.stretch.resolution())

        return [s.to_list() for s in simple.path]

def main():
    # TODO: Test the path planner for stretch here
    pass

if __name__ == '__main__':
    main()
