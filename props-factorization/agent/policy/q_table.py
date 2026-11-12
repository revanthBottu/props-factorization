from agent.policy.base_policy import Policy
import itertools
import random
import numpy as np


# states = [
#   [0, 1, 2], # dim 1
#   [0, 1, 2, 3], # dim 2
#   ...
#   [0, 1, 2]  # dim n
# ]


# actions = [
#   [0, 1, 2], # dim 1
#   [0, 1, 2, 3], # dim 2
#   ...
#   [0, 1, 2]  # dim m
# ]
class QTable(Policy):
    def __init__(self, states, actions):
        super().__init__(states, actions)


        self.q_table_length = self._calculate_q_table_length(states)
        print(f"Q Table length: {self.q_table_length}")
        if self.q_table_length > 120:
            raise Exception("Q Table is too large to handle")

        self.actions = actions
        self.states = states
        self.states = list(itertools.product(*self.states))
        if len(self.states[0]) == 1:
            self.states = [state[0] for state in self.states]
        self.initialize_policy()

    def _calculate_q_table_length(self, states):
        length = 1
        for state in states:
            length *= len(state)
        return length

    def initialize_policy(self):
        """
        Initializes the policy mapping for the agent.
        This method creates a nested dictionary structure where each state-action pair
        is assigned a random value. The states and actions are generated using the
        Cartesian product of the provided states and actions lists.
        Attributes:
            self.mapping (dict): A dictionary where keys are states (tuples) and values
                                 are dictionaries. The inner dictionaries have actions
                                 (tuples) as keys and random float values as values.
        """

        self.mapping = dict()
        if len(self.actions) == 1:
            actions = self.actions[0]
        else:
            actions = list(itertools.product(*self.actions))
        
        for state in self.states:

            self.mapping[state] = random.choice(actions)

    def get_action(self, state):
        """
        Returns the action with the highest Q-value for the given state.
        Args:
            state (tuple): The state for which to select an action.
        Returns:
            tuple: The action with the highest Q-value.
        """
        best_action = self.mapping[state]
        return best_action

    def __str__(self):
        """
        Returns a string representation of the Q-table.
        Returns:
            str: A string representation of the Q-table.
        """
        # TODO: Change the title of the table to the name of each dim, e.g., "cos(theta) | sin(theta) | velocity | action | q_value"
        table = ["State | Action"]
        for state, action in self.mapping.items():
            table.append(f"{state} | {action}")
        return "\n".join(table)

    def update_q_value(self, state, action):
        if state in self.mapping:
            self.mapping[state] = action

    def update_policy(self, new_q_table):
        """
        Updates the policy with a new Q-table.
        This method iterates over the provided new Q-table and updates the Q-values
        for each state-action pair using the `update_q_value` method.
        Args:
            new_q_table (list of tuples): A list where each element is a tuple containing
                                          (state, action, q_value). `state` is the current state,
                                          `action` is the action taken, and `q_value` is the
                                          corresponding Q-value.
        Example:
            new_q_table = [
                action1,
                action2,
                ...
            ]
            policy.update_policy(new_q_table)
        """

        for idx, action in enumerate(new_q_table):
            self.update_q_value(self.states[idx], action)
