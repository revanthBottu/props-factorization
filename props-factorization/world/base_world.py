class BaseWorld:
    def __init__(self, name):
        self.name = name
    
    def run(self):
        raise NotImplementedError("run method not implemented")
    
    def interpolate_state(self, state):
        raise NotImplementedError("interpolate_state method not implemented")

    def discretize_state(self, state):
        raise NotImplementedError("discretize_state method not implemented")

    def interpolate_action(self, action):
        raise NotImplementedError("interpolate_action method not implemented")

    def discretize_action(self, action):
        raise NotImplementedError("discretize_action method not implemented")    

