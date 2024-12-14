import numpy as np

class Buffer:
    #initialize the buffer class with size, batch size, seperate initilization for each actor's memory
    def __init__(self, max_size, critic_dims, actor_dims, 
            n_actions, n_agents, batch_size):
        self.mem_size = max_size  # 设置经验缓冲区的最大容量，即可以存储多少个经验样本。
        self.mem_cntr = 0  # 初始化计数器为 0，该计数器用于追踪当前缓冲区中存储的经验数量。它会在存储新的经验时递增
        self.n_agents = n_agents
        self.actor_dims = actor_dims
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.state_memory = np.zeros((self.mem_size, critic_dims))  # 初始化一个数组，用于存储所有智能体的当前状态，每个经验的状态信息会被存储在这个数组中
        self.new_state_memory = np.zeros((self.mem_size, critic_dims))  # 初始化一个数组，用于存储所有智能体的下一个状态，在采取动作后的状态变化。
        self.reward_memory = np.zeros((self.mem_size, n_agents))  #初始化一个数组，用于存储每个智能体在某个时刻收到的奖励。每个智能体的奖励信息将被存储在该数组的对应列中。

        self.init_actor_memory()

    def init_actor_memory(self):
        self.actor_state_memory = []  # 初始化为空列表，用于存储每个智能体的当前状态。
        self.actor_new_state_memory = []  # 初始化为空列表，用于存储每个智能体的下一状态。
        self.actor_action_memory = []  # 初始化为空列表，用于存储每个智能体的动作。

        for i in range(self.n_agents):
            self.actor_state_memory.append(
                            np.zeros((self.mem_size, self.actor_dims[i])))  # 创建一个数组，用来存储该智能体的状态信息。
            self.actor_new_state_memory.append(
                            np.zeros((self.mem_size, self.actor_dims[i])))  # 创建一个数组，用来存储该智能体的下一状态。
            self.actor_action_memory.append(
                            np.zeros((self.mem_size, self.n_actions)))  # 创建一个数组，用来存储该智能体的动作。

    #store the records in buffer, state is the concatenated state for the critic and raw_obs is the array of obs of each actor
    def store_transition(self, raw_obs, state, action, reward, 
                               raw_obs_, state_, done):
        
        
        index = self.mem_cntr % self.mem_size

        for agent_idx in range(self.n_agents):
            self.actor_state_memory[agent_idx][index] = raw_obs[agent_idx]
            self.actor_new_state_memory[agent_idx][index] = raw_obs_[agent_idx]
            self.actor_action_memory[agent_idx][index] = action[agent_idx]

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.mem_cntr += 1

    def sample_buffer(self):
        if(self.ready()!=True):
          return

        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        states = self.state_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]

        actor_states = []
        actor_new_states = []
        actions = []
        for agent_idx in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agent_idx][batch])
            actor_new_states.append(self.actor_new_state_memory[agent_idx][batch])
            actions.append(self.actor_action_memory[agent_idx][batch])

        return actor_states, states, actions, rewards, \
               actor_new_states, states_

    def ready(self):
        if self.mem_cntr >= self.batch_size:
            return True