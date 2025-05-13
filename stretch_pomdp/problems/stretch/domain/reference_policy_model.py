import random

import pomdp_py
from stretch_pomdp.problems.stretch.domain.transition_model import StretchTransitionModel
from stretch_pomdp.problems.stretch.domain.obs_model import StretchObservationModel
from stretch_pomdp.problems.stretch.domain.action import Action, MacroAction
from stretch_pomdp.problems.stretch.domain.state import State
import stretch_pomdp.problems.stretch.domain.path_planner as pp
import numpy as np


class StretchReferencePolicyModel(pomdp_py.RolloutPolicy):

    def __init__(self,
                 vamp_env,
                 visual_shortest_path: bool = False,
                 finite_ref_actions: bool = True):
        #print(f"restrict ref actions: {finite_ref_actions}")
        self.ACTIONS = [Action(i) for i in Action("Forward").MOTIONS.keys()]
        self._vamp_env = vamp_env
        self._Tm = StretchTransitionModel(vamp_env)
        self._Om = StretchObservationModel(vamp_env)
        self._path_planner = pp.PathPlanner(vamp_env)
        self._visual_shortest_path = visual_shortest_path
        self.max_nodes = 20
        self.finite_ref_actions = finite_ref_actions
        self.all_ref_actions = []
        # pre-fill the reference actions with primitive ones
        for a in self.ACTIONS:
            self.all_ref_actions.append(MacroAction([a]))
        self._ref_actions_num = 15

    def random_sample(self):
        return MacroAction([self.ACTIONS[np.random.choice(np.arange(len(self.ACTIONS)))]])

    def heuristic_sample(self, state, heuristics, epsilon=.5, h=0):
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
                    state_pos = list(state.get_position) + [0., 0.]
                    path = self._path_planner.shortest_path(state_pos, list(np.array(sampled_lm)))[:self.max_nodes]
                    return self.path_to_macro_action(path)
        else:
            if heuristics == "entropy":
                return self.dynamic_entropy_sample(state, h)
            elif heuristics == "distance":
                return self.weighted_distance_sample(state)
            elif heuristics == "uniform":
                return self.sample(state)
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

        # Generate the shortest path from the sampled state to the landmark
        path = self._path_planner.shortest_path(state.get_position, np.array(sampled_lm))[:self.max_nodes]
        # find macro actions that resemble the shortest path
        # TODO: refine the approximation using continuous actions representation instead of discrete ones
        if len(path) < 2:
            return MacroAction([])
        return self.path_to_macro_action(path)

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
        return self.path_to_macro_action(path)

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
        distances = [np.linalg.norm(np.array(state.get_position[:3]) - np.array(p)) for p in lm_pos[:-1]]
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
        return self.path_to_macro_action(path)

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
        return self.path_to_macro_action(path)

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

    # print("sampled_lm = {}".format(sampled_lm))

        state_pos = list(state.get_position) + [0., 0.]
        #print("SAMPLED LM: ", sampled_lm)
        #print("STATE: ", state_pos)
        path = self._path_planner.shortest_path(state_pos, list(np.array(sampled_lm)))[:self.max_nodes]

        if len(path) < 2:
            return MacroAction([])

        return self.path_to_macro_action(path)

    def path_to_macro_action(self, path):
        """
        :param path: A list of position nodes lead from start path[0] to goal path[-1]
        :return: A list of macro actions that best approximate the path
        """
        def get_next_position(position, action):
            """
            Transition function for the navigation model.
            :param position: agent current position (x,y,z, roll, pitch, yaw)
            :param action: The action to take.
            :return: The next state under environment constraints.
            """
            #print("ACTION: ", action.shape)
            next_position = np.zeros(position.shape)
            next_position[2:] = position[2:] + action[2:]
            next_position[0] = position[0] + action[0] * np.cos(position[2])
            next_position[1] = position[1] + action[0] * np.sin(position[2])
            return next_position
        
        actions = []

        for i in range(len(path) - 1):
            node = np.array(path[i])
            next_node = np.array(path[i + 1])

            #("NODE: ", node.shape)
            #print("NEXT: ", next_node.shape)

            action = min(self.ACTIONS,
                     key=lambda a: np.linalg.norm(next_node - get_next_position(node, np.array(list(a._motion)+[0., 0.]))))
            actions.append(action)

        # This code is for rendering the shortest path to the GUI. It substantially slows down the simulations.
        """if self._visual_shortest_path:
            for i in range(len(path) - 1):
                self._vamp_env._sim.addUserDebugLine(lineFromXYZ=path[i][0:3], lineToXYZ=path[i + 1][0:3],
                                                     lineColorRGB=[0.1, 0.1, 0.1],
                                                     lineWidth=5., lifeTime=0)"""
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
