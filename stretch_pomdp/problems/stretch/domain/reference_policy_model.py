import random

import pomdp_py
from stretch_pomdp.problems.stretch.domain.transition_model import StretchTransitionModel
from stretch_pomdp.problems.stretch.domain.obs_model import StretchObservationModel
from stretch_pomdp.problems.stretch.domain.action import Action, MacroAction
from stretch_pomdp.problems.stretch.domain.state import State
import stretch_pomdp.problems.stretch.domain.path_planner as pp
import numpy as np
import math
import time

import heapq
import itertools


import rerun as rr


class StretchReferencePolicyModel(pomdp_py.RolloutPolicy):

    def __init__(self,
                 vamp_env,
                 visual_shortest_path: bool = False,
                 finite_ref_actions: bool = True):
        #print(f"restrict ref actions: {finite_ref_actions}")
        self.ACTIONS = [Action(i) for i in Action("F").MOTIONS.keys()]
        self._vamp_env = vamp_env
        self._Tm = StretchTransitionModel(vamp_env)
        self._Om = StretchObservationModel(vamp_env)
        self._path_planner = pp.PathPlanner(vamp_env)
        self._visual_shortest_path = visual_shortest_path
        self.max_nodes = 8
        self.finite_ref_actions = finite_ref_actions
        self.all_ref_actions = []
        # pre-fill the reference actions with primitive ones
        for a in self.ACTIONS:
            self.all_ref_actions.append(MacroAction([a]))
        self._ref_actions_num = 15
        self.total_time = 0.
        self.rrtc_time = 0.

    def random_sample(self):
        return MacroAction([self.ACTIONS[np.random.choice(np.arange(len(self.ACTIONS)))]])

    def heuristic_sample(self, state, heuristics, epsilon=0.0, h=0):
        """
        General sampling queries, with prob(epsilon) sample the state space uniformly
        Otherwise, sample landmarks according to the given heuristics.
        :return: macro action
        """
        if np.random.uniform(0, 1) < epsilon:
            # uniformly sample from the state
            while True:
                sampled_lm = list(np.random.uniform(state.LOWER_BOUNDS, state.UPPER_BOUNDS)) # TODO: Fix this
                #print("SAMPLED LM: ", len(sampled_lm))
                if not self._vamp_env.collision_checker(sampled_lm):
                    state_pos = list(state.get_position)[:11] + [0., 0.]
                    path = self._path_planner.shortest_path(state_pos, list(np.array(sampled_lm)))[:self.max_nodes]
                    return self.path_to_macro_action(state, path)
        else:
            if heuristics == "entropy":
                return self.dynamic_entropy_sample(state, h)
            elif heuristics == "distance":
                return self.weighted_distance_sample(state)
            elif heuristics == "uniform":
                start = time.time()
                action = self.sample(state)
                self.total_time += time.time() - start
                return action
            else:
                raise Exception(f"Invalid heuristics {heuristics}")

    def sample(self, state: State, goal_only=False):
        """
        Sample a landmark (milestone) at random and build a macro action that leads to this landmark
        :param state: a state sampled from a belief
        :param goal_only: If true, goal state is always sampled as milestone
        :return: a macro action (list of actions)
        """
        # sample milestones (landmarks) TODO: implement different milestone sampling heuristic
        p = np.random.uniform(0, 1)
        lm_pos = self._vamp_env.get_landmarks_pos()
        if goal_only:
            sampled_lm = lm_pos[-1]
        else:
            # heuristic path diversity sampling
            if p < 0.5:
                sampled_lm = lm_pos[-1]
            else:
                sampled_lm = lm_pos[np.random.choice(len(lm_pos)-1)]

        state_pos = list(state.get_position)[:11] + [0., 0.]

        # Generate the shortest path from the sampled state to the landmark
        start = time.time()
        vamp_env = self._vamp_env.state_to_vamp(state)
        path = self._path_planner.shortest_path(state_pos, np.array(sampled_lm), vamp_env=vamp_env)[:self.max_nodes]
        self.rrtc_time += time.time() - start
        # find macro actions that resemble the shortest path
        # TODO: refine the approximation using continuous actions representation instead of discrete ones
        if len(path) < 2:
            return MacroAction([])
        return self.path_to_macro_action(state, path)

    def diversity_sample(self, state: State):
        """
        Sample positions inside the landmark zones instead of just the center point of the landmark to promote
        diversities in paths generated from path planner.
        :param state: Sampled state from the agent's belief
        :return: macro actions leads to the sampled position.
        """
        p = np.random.uniform(0, 1)
        if p < 0.5:
            sampled_lm = self._vamp_env.get_goal_pos
        else:
            sampled_lm = self._vamp_env.sample_landmark_zones(np.random.choice(self._vamp_env.get_num_lms))
        # Generate the shortest path from the sampled state to the landmark
        path = self._path_planner.shortest_path(state.get_position, np.array(sampled_lm))[:self.max_nodes]
        return self.path_to_macro_action(state, path)

    def weighted_distance_sample(self, state: State):
        """
        Similar to the sample function, but use distance as weights to sample milestones
        :param state: State sampled from a belief
        :return: Action(s) that leads to the milestone
        """
        # sample milestones (landmarks) TODO: implement different milestone sampling heuristic
        p = np.random.uniform(0, 1)
        lm_pos = self._vamp_env.get_landmarks_pos()

        # calculate distances and convert them into weights
        distances = [np.linalg.norm(np.array(state.get_position[:3]) - np.array(p)[:3]) for p in lm_pos[:-1]]
        distances -= np.max(distances)
        distances /= sum(distances)

        # heuristic path diversity sampling
        if p < 0.5:
            sampled_lm = lm_pos[-1]
        else:
            sampled_lm = lm_pos[np.random.choice(len(lm_pos)-1, p=distances)]

        # Generate the shortest path from the sampled state to the landmark
        path = self._path_planner.shortest_path(state.get_position, np.array(sampled_lm))[:self.max_nodes]
        if len(path) < 2:
            return MacroAction([])
        return self.path_to_macro_action(state, path)

    def weighted_distance_diversity_sample(self, state: State):
        # sample milestones (landmarks) TODO: implement different milestone sampling heuristic
        p = np.random.uniform(0, 1)
        lm_pos = self._vamp_env.get_landmarks_pos()

        # calculate distances and convert them into weights
        distances = [np.linalg.norm(np.array(state.get_position[:3]) - np.array(p)) for p in lm_pos[:-1]]
        distances -= np.max(distances)
        distances /= sum(distances)

        # heuristic path diversity sampling
        if p < 0.5:
            sampled_lm = lm_pos[-1]
        else:
            sampled_idx = np.random.choice(len(lm_pos) - 1, p=distances)
            sampled_lm = self._vamp_env.sample_landmark_zones(sampled_idx)

        # Generate the shortest path from the sampled state to the landmark
        path = self._path_planner.shortest_path(state.get_position, np.array(sampled_lm))[:self.max_nodes]
        if len(path) < 2:
            return MacroAction([])
        return self.path_to_macro_action(state, path)

    def prm_path_sampling(self, state: State, interp_gap=0.5):
        """
        Use pre-computed prm to generate the shortest path from the current state to a selected landmark.
        :param state: sampled state of the agent
        :param interp_gap: the gap used to interpolate paths
        :return: macro actions that leads to the sampled landmark
        """
        raise NotImplementedError

    def dynamic_entropy_sample(self, state: State, h, use_prm=False):
        """
        Dynamic adjust milestone weights depends on the entropy of the belief. When the entropy is high,
        we sample more towards nearby landmarks to localise, otherwise we sample towards the goal.
        :param state: state sampled from a belief
        :param h: entropy of the belief
        :param use_prm: use pre-computed prm if true, use rrtc otherwise
        :return: macro actions
        """
        h = np.clip(h, 0, 1)

        # compute all landmark positions
        lm_pos = self._vamp_env.get_landmarks_pos()
        distances = [np.linalg.norm(np.array(state.get_position[:2]) - np.array(p[:2])) for p in lm_pos]
        distances -= np.max(distances)
        distances /= sum(distances)

        # compute mixture of probabilities weighted by entropy
        g = np.zeros(len(distances))
        g[-1] = 1
        w = h * np.array(distances) + (1 - h) * g

        # sample a milestone according to above probability
        sampled_idx = np.random.choice(len(lm_pos), p=w)
        sampled_lm = lm_pos[sampled_idx]
        state_pos = list(state.get_position)[:11] + [0., 0.]
        #print("SAMPLED LM: ", sampled_lm)
        #print("STATE: ", state_pos)
        vamp_env = self._vamp_env.state_to_vamp(state)
        path = self._path_planner.shortest_path(state_pos, list(np.array(sampled_lm)), vamp_env=vamp_env)[:self.max_nodes]

        if len(path) < 2:
            return MacroAction([])

        return self.path_to_macro_action(state, path)


    def path_to_macro_action(self, state, path):
        

        """def get_lookahead_point(path, current_pos, lookahead_dist, last_lookahead_idx):
            pos = np.array(current_pos[:2])
    
            for i in range(last_lookahead_idx, len(path) - 1):
                a = np.array(path[i][:2])
                b = np.array(path[i+1][:2])
                ab = b - a
                ab_len = np.linalg.norm(ab)
                if ab_len < 1e-6:
                    continue

                ap = pos - a
                proj_len = np.clip(np.dot(ap, ab) / ab_len, 0.0, ab_len)
                closest = a + (proj_len / ab_len) * ab
                dist_from_robot = np.linalg.norm(closest - pos)
                remaining = lookahead_dist - dist_from_robot

                if remaining <= ab_len - proj_len:
                    lookahead = closest + ab * (remaining / ab_len)
                    yaw = math.atan2(ab[1], ab[0])
                    return (lookahead[0], lookahead[1], yaw), i

            # Fallback to final point
            end = path[-1]
            return (end[0], end[1], end[2]), len(path) - 1"""

        def get_lookahead_point(path, current_pos, lookahead_dist):
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


        def angle_diff(a, b):
            a = (a + np.pi) % (2 * np.pi) - np.pi
            b = (b + np.pi) % (2 * np.pi) - np.pi

            diff1 = abs((a - b + np.pi) % (2 * np.pi) - np.pi)
            diff2 = abs((a - (b + np.pi) + np.pi) % (2 * np.pi) - np.pi)
            return min(diff1, diff2)

        def get_cost(pose, lookahead):
            dx = lookahead[0] - pose[0]
            dy = lookahead[1] - pose[1]
            dist_error = math.hypot(dx, dy)

            heading_to_target = math.atan2(dy, dx)
            heading_error = angle_diff(pose[2], heading_to_target)

            # Weighted cost function
            cost = 1.0 * dist_error + 0.5 * heading_error# - 0.3 * v
            return cost

        if len(path) < 2:
            print("PATH TOO SHORT")
            return MacroAction([])

        current_pose = state._position  # (x, y, theta)
        actions = []

        #last_lookahead_idx = 0
        for _ in range(self.max_nodes):
            lookahead= get_lookahead_point(path, current_pose, 0.5)#, last_lookahead_idx)
            #lookahead2 = get_lookahead_point(path, current_pose, 0.25)
            #rr.log("LOOKAHEAD", rr.Points3D([lookahead[0], lookahead[1], 0.0], radii=[0.05]))

            best_action = None
            best_cost = float('inf')
            best_new_pose = None

            for action in self.ACTIONS:
                if action._name == "None":
                    continue

                v = action._motion[0]
                w = action._motion[1]

                predicted_pose = self._Tm.get_next_position(current_pose, action)

                if np.array_equal(predicted_pose, current_pose):
                    continue

                cost = get_cost(predicted_pose, lookahead) #+ get_cost(predicted_pose, lookahead2)) / 2.0

                if cost < best_cost:
                    best_cost = cost
                    best_action = action
                    best_new_pose = predicted_pose#self._Tm.get_next_position(current_pose, action)

            #if best_action is None:
            #    print("No suitable macro action found. Stopping.")
            #    break

            if best_action is not None:
                actions.append(best_action)
                current_pose = best_new_pose

        if len(actions) == 0:
            print("NO ACTIONS FOUND FROM POSE: ", state)

        # Visualization
        nodes = [state._position]
        for action in actions:
            nodes.append(self._Tm.get_next_position(nodes[-1], action))

        rr.log("Macro_Actions", rr.LineStrips3D([list(n[:2]) + [0.0] for n in nodes]))

        actions.extend([Action("None"), Action("None")])
        return MacroAction(actions)

    def get_all_actions(self, state: State = None, history=None):
        """
        Returns a list of all possible actions, if feasible.
        """
        if self.finite_ref_actions:
            return self.expand_ref_actions(state)
        return self.ACTIONS

    def expand_ref_actions(self, state: State = None):
        expanded = self.all_ref_actions
        while len(expanded) < self._ref_actions_num:
            action = self.sample(state)
            if len(action.action_sequence) == 0:
                break
            expanded.append(action)
        return expanded

    def rollout(self, state: State, goal_only = False, history=None, sample_heuristic="base"):
        """
        Return an action given current state and history for primitive rollout, history is a list of tuples (a, o).
        Warning: this is a primitive rollout used for non vamp based planners.
        Vamp based planner should have rollout policy implemented within itself.
        """
        # if self.finite_ref_actions:
        #    return random.sample(self.all_ref_actions, 1)[0]
        if self.finite_ref_actions:
            sampled_action = self.sample(state)
            if len(sampled_action.action_sequence) == 0:
                return random.sample(self.ACTIONS, 1)[0]
        return random.sample(self.get_all_actions(state, history), 1)[0]
