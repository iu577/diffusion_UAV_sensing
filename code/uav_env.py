
import numpy as np
import random
import matplotlib.pyplot as plt
import math 

class UAVTASKENV():

  # 环境初始化
  def __init__(self, c1, c2):

    #initializing UAV fog server cordinates and data for each UAV task and other variables
    self.k = plt.subplots()
    self.fig = self.k[0]
    self.ax = self.k[1]
    self.cluster1 = c1  # 我猜这两个cluster，一个是MU,一个是EC
    self.cluster2 = c2
    self.uecord = self.cluster1+self.cluster2
    self.fog = [[450,450,100],[50,50,100]]
    self.ue_data = self.UE_data()  # 输出20个任务
    self.connected_array = np.full(20,-1)  # 长度为20的向量，赋值为-1

    #transmit power of each UAV is decided to Be 5 watts
    self.P = 5

  # 传入两个需要计算的物体的位置，计算其距离
  def dist(self, pos1, pos2):
    distance = ((pos1[0]-pos2[0])**2+(pos1[1]-pos2[1])**2+(pos1[2]-pos2[2])**2)**0.5
    return distance

  # 判断UAV_i是否与UE_j相连接，如果两者之间距离小于300，则判断其是连接的
  def connected(self, pos_ue, pos_uav, i, j):
    #if UAV and UE are in a range of distance then connect them
    if(self.dist(pos_ue, pos_uav)<=300):
      if(self.connected_array[j]==-1):
        self.connected_array[j] = i  # 作用：判断每个任务可以到哪一个无人机进行处理

  # 信道增益
  def channelGain(self, d):
    #calculation of channel gain when power gain per meter is -50dB
    return -50 - 20*(math.log10(d))

  # 计算信噪比
  def SNR(self, gain, Transmitpower):
    #snr = signal power (gain*transmit power)/noise power(-100dBm)
    P = Transmitpower*10**(gain/10)
    N = 10**(-13)
    return P/N

  # 计算数据传输速率
  def datarate(self, SNR):
    #bandiwdth of 1Mhz is assigned to each UE
    return math.log2(1+SNR)


  def step(self, actions):
    # 遍历当前智能体的状态
    for i, state in enumerate(self.state):
      # 模拟无人机轨迹移动
      action = actions[i]  # 为每个智能体分配一个动作
      d = (action[0])*270
      theta = (action[1])*180
      dummy = []
      dummy.append(state[0])
      dummy.append(state[1])
      dummy = np.array(dummy, dtype = np.float32)
      state[0] += d*math.cos(math.radians(theta))
      state[1] += d*math.sin(math.radians(theta))
      if((state[0]>600 or state[0]<0 )or (state[1]>600 or state[1]<0)):
        state[0] = dummy[0]
        state[1] = dummy[1]

      # 对所有用户设备（UE）进行循环，由于无人机位置更新，需要重新定义连接性
      for j in range(0,len(self.uecord)):
        self.connected(self.uecord[j], state, i, j)

    #move the UAVs and connect them, if the UAV is out of area then don't move it

    U = 0
    for i, state in enumerate(self.state):
      action = actions[i]
      X = []
      T = 0
      E = 0
      th = 100000000
      for j in range(2,22):
        X.append(action[j]/2 + 0.5)

      for j in range(0,20):
        uav = state  # 取一个UAV
        ue = self.uecord[j]  # 取一个UE
        dis = self.dist(ue, uav)  # 计算两者之间的距离
        C = self.ue_data[j][0]  # 所需周期数
        D = self.ue_data[j][1]  # 任务大小
        # transmit power of each UE(iot device) is assigned to be 0.1 watts
        # further calculation of total time and energey and thorughput
        r1 = self.datarate(self.SNR(self.channelGain(dis), 0.1))  # 计算传输速率
        t1 = D/r1 # 传输延迟
        e1 = 0.1*t1  # 传输能耗=传输功率*传输延迟
        x = X[j]
        t2 = (0.001*(x*C*D))/3  # 计算延迟
        e2 = 0.3*t2  # 计算能耗
        if(x==1):
          th = min(th, r1)
        
        dis2 = self.dist(uav, self.fog[i])
        r2 = self.datarate(self.SNR(self.channelGain(dis2), 5))
        t3 = (1-x)*D/r2  # 传输延迟
        e3 = 5*t3  # 传输能耗
        t4 = 0.0001*((1-x)*C*D)  # 计算延迟

        if(x!=1):
          th = min(th, min(r1,r2))
        
        
        factor = 10
        
        if(self.connected_array[j]==i):
          factor = 1

        t_X = t1+t2+t3+t4  # 总延迟
        e_X = e1+e2+e3  # 总能耗
        
        T += factor*t_X  # 因子
        E += factor*e_X  # 因子

      #minimise the system cost so U = sum of energy and time and reciprocal of throughput as we have to maximise thorughput
      U += ((E+T)+(1/th))
    ct = 0
    for x in self.connected_array:
      if(x==-1):
        ct += 1

    #U = U*ct
    #2 ways to calculate coverage 
    #either by multiplying the count of disconnected UAVs in reward itsef, or by giving penalty 
    #like
    if(ct>6):
      U += 10*ct
    return self.state, -1*U, False, {}
    

  
  def UE_data(self):

    C = []  # 每比特处理任务所需的 CPU 周期数。
    D = []  # 任务的数据大小（以 MB 为单位）。
    L = []  # L is task genereted in a UAV in each timestamp for now considered to be 1
    while(len(C)<20):  # 随机生成20个任务，所需周期数在100-200之间
      C.append(random.randint(100,200))
    while(len(D)<20):  # 随机生成20个任务，数据大小在1-5之间
      D.append(random.randint(1,5))
    while(len(L)<20):  # 任务到达率，设置为1
      L.append(1)
    k = []  # 用K表示任务
    i = 0
    while(i<20):
      k.append([C[i],D[i],L[i]])
      i += 1
    self.statex = np.array(k, dtype = np.float32)
    return self.statex

  def reset(self):
    #reset the connections and UAV cordinates
    self.connected_array = np.full(20,-1)  # 初始化一个长度为30的数组，赋值为-1
    self.ue_data = self.UE_data()  # 调用UE_data

    # 生成两个UAV的位置
    X = round(random.uniform(270,300),2)
    Y = round(random.uniform(295,300),2)
    X1 = round(random.uniform(270,300),2)
    Y1 = round(random.uniform(295,300),2)
    if(X1==X):
      while(X1==X and X1-X<=20):
        X1 = round(random.uniform(300,400),2)
    
    if(Y1==Y):
      while(Y1==Y):
        Y1 = round(random.uniform(300,400),2)
    
    u1 = np.array([X,Y,60], dtype = np.float32)  # 将UAV坐标转化为np数组，60为高度
    u2 = np.array([X1,Y1,60], dtype = np.float32)  # 将UAV坐标转化为np数组，60为高度
    self.state = [u1,u2]  # 将两个 UAV 的状态 [u1, u2] 存储在 self.state 中。这里 self.state 保存了两个 UAV 的位置和高度信息。
    return self.state

  def render(self):
    #plot the UAV's, UE's and the Fog servers
    self.ax.cla()
    X = []
    Y = []
    for i in range(0,10):
        X.append(self.cluster1[i][0])
        Y.append(self.cluster1[i][1])

    for i in range(0,10):
        X.append(self.cluster2[i][0])
        Y.append(self.cluster2[i][1])
    self.ax.set_xlim([0,600])
    self.ax.set_ylim([0,600])
    self.ax.set_title('UE-Clusters')
    self.ax.plot(np.array(X),np.array(Y),'ro')
    uavcord1 = (self.state[0][0],self.state[0][1])
    uavcord2 = (self.state[1][0],self.state[1][1])
    self.ax.scatter(uavcord1[0], uavcord1[1], marker = "X", s= 100)
    self.ax.scatter(uavcord2[0], uavcord2[1], marker = "X", s= 100)
    self.ax.scatter(self.fog[0][0], self.fog[0][1], marker = "^", s= 100)
    self.ax.scatter(self.fog[1][0], self.fog[1][1], marker = "^", s= 100)
    self.ax.grid(True)
    plt.show()
    plt.pause(0.8)
