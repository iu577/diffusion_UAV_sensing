from uav_env import UAVTASKENV
import random
import numpy as np
from buffer import Buffer
from helperclass import MADDPG
import matplotlib.pyplot as plt


#training of the agents in the environment using MADDPG

#function to concatenate states of each agent for the critic
def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

# 定义了一个函数，用于创建UE_cluster
def create_UE_cluster(x1, y1, x2, y2):
  # 初始化三个空列表，用于保存，x,y,z坐标
  X = []
  Y = []
  Z = []
  # 生成10个在x1，x2之间的不同的x坐标
  while(len(X)<10):
    cord_x = round(random.uniform(x1,x2),2)
    if(cord_x not in X):
      X.append(cord_x)
  # 生成10个在y1，y2之间的不同的y坐标
  while(len(Y)<10):
    cord_y = round(random.uniform(y1,y2),2)
    if(cord_y not in Y):
      Y.append(cord_y)
  # 生成10个为0的z坐标
  while(len(Z)<10):
      Z.append(0)

  k = []  # 用k来保存10个UE的坐标
  i = 0
  while(i<10):
      k.append([X[i],Y[i],Z[i]])
      i += 1
        
  return k

# 输出的是10个设备的位置，只是分别所在的位置不同
ue_cluster_1 = create_UE_cluster(400, 450, 470, 520)
# 输出的是10个设备的位置，只是分别所在的位置不同
ue_cluster_2 = create_UE_cluster(30,30,100,100)


#main loop
if __name__ == '__main__':
    
    
    env = UAVTASKENV(ue_cluster_1, ue_cluster_2)  # 输入两个ue_cluster将环境初始化
    n_agents = 2  # 包含两个智能体，后面可能会为每个智能体配置动作空间和评价网络
    actor_dims = []  # 初始化一个空列表，存储每个智能体的动作维度
    for i in range(n_agents):
        actor_dims.append(3)  # 将3赋值到actor_dims中，actor_dims = [3，3]
    critic_dims = sum(actor_dims)  # critic_dims 对actor_dims求和得6
    PRINT_INTERVAL = 1  # 通常用于控制打印信息

    
    n_actions = 22 # n_actions设置为 22，表示每个智能体的动作空间大小。假设这是一个离散的动作空间，每个智能体有 22 个可能的动作选择。
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions,
                           # 表示每个智能体的 Actor 和 Critic 网络都有两个隐藏层，第一层有 300 个神经元，第二层有 400 个神经元。
                           fc1=300, fc2=400,    # fc1：第一层的神经元数量 fc2：第二层的神经元数量
                           # 学习率-分别用于 Actor 和 Critic 网络的优化
                           alpha=0.001, beta=0.001,  # alpha：学习率
                           chkpt_dir='./')  # 文件报错路径

    # 1000000：经验回放缓冲区的最大容量，表示可以存储最多 1,000,000 个经验（状态、动作、奖励、下一状态等）。当缓冲区已满时，新的经验将覆盖最旧的经验。
    # batch_size=100 设置批处理大小，即每次从经验回放缓冲区中随机抽取 100 个经验进行训练。
    memory = Buffer(1000000, critic_dims, actor_dims, 
                        n_actions, n_agents, batch_size=100)

    # 用于跟踪智能体在所有训练过程中执行的总步数。在每个回合（episode）结束时，可以根据 total_steps 来分析智能体的学习进度和训练时间。
    total_steps = 0
    score_history = []  # 初始化为空列表，用于存储每个训练回合（episode）结束时智能体的得分（或奖励）
    n_episodes = 1000
    timestamp = 200  # 设定为 200，这个变量通常用于控制训练中某些操作的执行频率或间隔。在一些实现中，它可能表示每隔多少步或回合输出一次日志，或者每隔多少步进行一次模型的保存。
    avg = []  # 初始化为空列表，可能用于存储智能体在每隔一定步数或回合时的平均得分。通过计算平均得分，可以更清楚地看到智能体学习的长期进展，而不是关注单个回合的波动
    

    #standard implemntaion of MADDPG algorithim
    for i in range(n_episodes):
        obs = env.reset()  # 输出无人机位置信息等
        score = 0
        episode_step = 0

        # timestamp：代表每个训练回合中的时间步数，与环境交互的最大步数，在每个时间步，智能体将与环境进行一次交互，选择动作并接收奖励。
        for j in range(timestamp):
            
            actions = maddpg_agents.choose_action(obs)  # 智能体根据当前的状态（obs）选择动作
            obs_, reward, done, info = env.step(actions)  # obs_：下一个状态，reward：执行该动作后的奖励

            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)

            # 将智能体与环境交互的信息存储到经验回放缓冲区
            memory.store_transition(obs, state, actions, reward, obs_, state_, done)

            # 每10步学习一次
            if total_steps % 10 == 0:
                maddpg_agents.learn(memory)

            obs = obs_

            score += reward
            total_steps += 1
            episode_step += 1

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if i % PRINT_INTERVAL == 0 and i > 0:
            print('episode', i, 'average score {:.1f}'.format(avg_score))
            avg.append(avg_score)

    #plot the final results
    maddpg_agents.save_checkpoint()
    plt.plot(avg)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")
    plt.show()