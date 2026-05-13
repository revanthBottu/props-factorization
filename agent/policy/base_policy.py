from typing import List

class Policy:
    def __init__(self, states: List[tuple[float]], actions: List[str]):
        self.states = states
        self.actions = actions
        self.mapping = dict() # datastructure for tracking policy


    def get_action(self, state):
        """
        choose action based on some condition
        """
        # this is just an example
        return self.mapping[state]


    def __str__(self):
        """
        print the action, directly calling this from LLM should print it
        """
        _table = list(f"States\t\tAction")
        for key, val in self.mapping.items():
            _table.append(f"{key}\t\t{val}")

        return "\n".join(_table)


    def initlize_policy(self):
        """
        ideally should setup the self.mapping state and action pairs
        """
        pass


    def update_policy(self):
        """
        algorithm for updating
        """
        pass
