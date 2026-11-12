from collections import deque
import numpy as np
import os
import copy
np.set_printoptions(precision=2)


class TrajectoryBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def add(self, state, action, reward):
        if self.count < self.buffer_size:
            self.buffer.append((state, action, reward))
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append((state, action, reward))

    def size(self):
        return self.count

    def clear(self):
        self.buffer.clear()
        self.count = 0

    def get_trajectory(self):
        return self.buffer

    def __str__(self):
        buffer_table = "State | Action | Reward\n"
        idx = 0
        for state, action, reward in self.buffer:
            if idx % 1 == 0:
                buffer_table += f"{state} | {action} | {reward}\n"
            idx += 1
        return buffer_table


class ReplayBuffer:
    def __init__(self, max_traj_count, max_traj_length):
        self.buffer_size = max_traj_count
        self.max_traj_length = max_traj_length
        self.count = 0
        self.buffer = deque()
    
    def start_new_trajectory(self):
        if self.count >= self.buffer_size:
            self.buffer.popleft()
            self.count -= 1
        self.buffer.append(TrajectoryBuffer(self.max_traj_length))
        self.count += 1

    def add_step(self, state, action, reward):
        if len(self.buffer) == 0:
            raise Exception("Replay buffer is empty. Start a new trajectory first.")
        self.buffer[-1].add(state, action, reward)

    def size(self):
        return self.count

    def clear(self):
        self.buffer.clear()
        self.count = 0
    
    def __str__(self):
        buffer_table = ""
        for i, trajectory in enumerate(self.buffer):
            buffer_table += f"Trajectory {i}:\n"
            buffer_table += str(trajectory)
            buffer_table += "\n"
        return buffer_table


class EpisodeRewardBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, weights: np.ndarray, bias, reward):
        self.buffer.append((weights, bias, reward))
    
    def __str__(self):
        buffer_table = "Parameters | Reward\n"
        for weights, bias, reward in self.buffer:
            parameters = np.concatenate((weights, bias))
            buffer_table += f"{parameters.reshape(1, -1)} | {reward}\n"
        return buffer_table

    def load(self, folder):
        # Find all episode files
        all_files = [os.path.join(folder, x) for x in os.listdir(folder) if x.startswith('warmup_rollout')]
        all_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

        # Load parameters from all episodes
        for filename in all_files:
            with open(filename, 'r') as f:
                lines = f.readlines()
                parameters = []
                for line in lines:
                    if "parameter ends" in line:
                        break
                    try:
                        parameters.append([float(x) for x in line.split(',')])
                    except:
                        continue
                parameters = np.array(parameters)

                rewards = []
                for line in lines:
                    if "Total reward" in line:
                        try:
                            rewards.append(float(line.split()[-1]))
                        except:
                            continue
                rewards_mean = np.mean(rewards)
                self.add(parameters[:-1], parameters[-1:], rewards_mean)
                f.close()
        print(self)



class EpisodeRewardBufferNoBias:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, weights: np.ndarray, reward):
        self.buffer.append((weights, reward))
    
    def sort(self):
        self.buffer = deque(sorted(self.buffer, key=lambda x: x[1], reverse=False), maxlen=self.buffer.maxlen)
    
    def __str__(self):
        buffer_table = "Parameters | Reward\n"
        for weights, reward in self.buffer:
            buffer_table += f"{weights.reshape(1, -1)} | {reward}\n"
        return buffer_table

    def load(self, folder):
        # Find all episode files
        all_files = [os.path.join(folder, x) for x in os.listdir(folder) if x.startswith('warmup_rollout')]
        all_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

        # Load parameters from all episodes
        for filename in all_files:
            with open(filename, 'r') as f:
                lines = f.readlines()
                parameters = []
                for line in lines:
                    if "parameter ends" in line:
                        break
                    try:
                        parameters.append([float(x) for x in line.split(',')])
                    except:
                        continue
                parameters = np.array(parameters)

                rewards = []
                for line in lines:
                    if "Total reward" in line:
                        try:
                            rewards.append(float(line.split()[-1]))
                        except:
                            continue
                rewards_mean = np.mean(rewards)
                self.add(parameters, rewards_mean)
                f.close()
        print(self)


class EpisodeRewardMeanStdBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, weights: np.ndarray, bias, reward_mean, reward_std):
        self.buffer.append((weights, bias, reward_mean, reward_std))
    
    def __str__(self):
        buffer_table = "Parameters | Reward_Mean | Reward_Std \n"
        for weights, bias, reward_mean, reward_std in self.buffer:
            parameters = np.concatenate((weights, bias))
            parameters = parameters.reshape(1, -1)[0]
            parameters = ', '.join([f"{x:.3f}" for x in parameters])
            buffer_table += f"{parameters} | {reward_mean} | {reward_std}\n"
        return buffer_table
    
    def load(self, folder):
        # Find all episode files
        all_files = [os.path.join(folder, x) for x in os.listdir(folder) if x.startswith('warmup_rollout')]
        all_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

        # Load parameters from all episodes
        for filename in all_files:
            with open(filename, 'r') as f:
                lines = f.readlines()
                parameters = []
                for line in lines:
                    if "parameter ends" in line:
                        break
                    try:
                        parameters.append([float(x) for x in line.split(',')])
                    except:
                        continue
                parameters = np.array(parameters)

                rewards = []
                for line in lines:
                    if "Total reward" in line:
                        try:
                            rewards.append(float(line.split()[-1]))
                        except:
                            continue
                rewards_mean = np.mean(rewards)
                rewards_std = np.std(rewards)
                self.add(parameters[:-1], parameters[-1:], rewards_mean, rewards_std)
                f.close()
        print(self)




class QTableBuffer:
    def __init__(self):
        self.tables = []
        self.rewards = []
    
    def add(self, table, reward):
        self.tables.append(copy.deepcopy(table))
        self.rewards.append(reward)
    
    def to_string(self):
        table = ''
        for idx, t in enumerate(self.tables):
            table += f"Policy Table {idx + 1}:\n"
            table += t.to_string()
            table += f"Total reward: {self.rewards[idx]}\n\n"
        return table
    

class MemoryTable:
    def __init__(self):
        self.initialize()
    
    def to_string(self):
        table_template = f"""
state | action | reward
------------------------
        """
        table = table_template
        for state, action, value in self.table:
            table += f"{state} | {action} | {value}\n"
        return table

    def get_dict(self):
        result = {}
        for state, action, value in self.table:
            if state not in result:
                result[state] = {}
            result[state][action] = value
        return result
    
    def initialize(self):
        self.table = []


class QTableRewardTrajBufferRow:
    def __init__(self):
        self.q_table_reward = QTableBuffer()
        self.traj_buffer = MemoryTable()


class QTableRewardTrajBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def start_new_row(self):
        self.buffer.append(QTableRewardTrajBufferRow())
    
    def add_q_table(self, table, reward):
        self.buffer[-1].q_table_reward.add(table, reward)
    
    def add_traj(self, table):
        self.buffer[-1].traj_buffer.table = table
    
    def add_traj_row(self, state, action, reward):
        self.buffer[-1].traj_buffer.table.append((state, action, reward))

    def to_string(self):
        table = ''
        for idx, row in enumerate(self.buffer):
            table += f"Policy {idx + 1}:\n"
            table += row.q_table_reward.to_string()
            table += f"Trajectory by executing Policy {idx + 1}:\n"
            table += row.traj_buffer.to_string()
            table += "\n"
        return table



class EpisodeRewardTrajBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def start_new_row(self):
        self.buffer.append(QTableRewardTrajBufferRow())
    
    def add_q_table(self, table, reward):
        self.buffer[-1].q_table_reward.add(table, reward)
    
    def add_traj(self, table):
        self.buffer[-1].traj_buffer.table = table
    
    def add_traj_row(self, state, action, reward):
        self.buffer[-1].traj_buffer.table.append((state, action, reward))

    def to_string(self):
        table = ''
        for idx, row in enumerate(self.buffer):
            table += f"Policy {idx + 1}:\n"
            table += row.q_table_reward.to_string()
            table += f"Trajectory by executing Policy {idx + 1}:\n"
            table += row.traj_buffer.to_string()
            table += "\n"
        return table