import vamp

class PathPlanner:
    def __init__(self, vamp_env):
        self._vamp_env = vamp_env._env # TODO: Problem with passing the right vamp environment vamp.Environment()

    def shortest_path(self, source, target, max_iterations=100000, restarts=0):
        # print(f"finding shortest path between {source} and {target}")
        #settings = vamp.RRTCSettings()
        #settings.range = 1.
        #settings.max_iterations = max_iterations

        (vamp_module, planner_func, plan_settings,
         simp_settings) = vamp.configure_robot_and_planner_with_kwargs("stretch", "rrtc")

        sampler = getattr(vamp_module, "halton")()
        sampler.skip(0)
        #source = [0., 0., 0., 0.4, 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        #target = [1., 0., 0.0, 0.95, 0.1, 0.1, 0.1, 0.1, 0., 0., 0., 0., 0.]
        #print("SOURCE: ", len(source))
        #print("TARGET: ", len(target))
        result = planner_func(source, target, self._vamp_env, plan_settings, sampler)
        
        for i in range(restarts):
            if len(result.path) > 0:
                break

            print(f"Trying restart attempt #{i}...")
            result = planner_func(source, target, self._vamp_env, plan_settings, sampler)

        if result is None or result.size == 0:
            print(f"Warning: RRT-C failed to find a path despite {restarts} attempts!")
            return []

        simple = vamp_module.simplify(result.path, self._vamp_env, simp_settings, sampler)
        simple.path.interpolate(vamp.stretch.resolution())

        return [s.to_list() for s in simple.path]

def main():
    # TODO: Test the path planner for stretch here
    pass

if __name__ == '__main__':
    main()
