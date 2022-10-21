import random
from collections import deque

import cv2  # 需要安装OpenCV的包
import numpy as np
import torch
import torch.nn as nn

import cnn_Model as cn
import game_Model as gm

global sss


def train_model(GAME, ACTIONS, GAMMA, OBSERVE, EXPLORE, FINAL_EPSILON, INITIAL_EPSILON, REPLAY_MEMORY, BATCH,
                FRAME_PER_ACTION):
    # 开始在内存／GPU上定义一个网络
    use_cuda = torch.cuda.is_available()  # 检测本台机器中是否有GPU
    # 创建一个神经网络
    net = cn.Net()
    # 初始化网络权重。之所以自定义初始化过程是为了增加神经网络权重的多样性
    net.init()
    # 如果有GPU，就把神经网络全部搬到GPU内存中做运算
    net = net.cuda() if use_cuda else net

    # 定义损失函数为MSE
    criterion = nn.MSELoss().cuda() if use_cuda else nn.MSELoss()
    # 定义优化器，并设置初始学习率维10^-6
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-6)

    # 开启一个游戏进程，开始与游戏引擎通话
    game_state = gm.GameState()
    # 学习样本的存储区域deque是一个类似于list的存储容器
    D = deque()
    # 状态打印log记录位置
    # a_file = open("logs_" + GAME + "/readout.txt", 'w')
    # h_file = open("logs_" + GAME + "/hidden.txt", 'w')

    # 将游戏设置为初始状态，并获得一个80*80的游戏湖面
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)

    # 将初始的游戏画面叠加成4张作为神经网络的初始输入状态s_t
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=0)
    # 设置初始的epsilon（采取随机行动的概率），并准备训练
    epsilon = INITIAL_EPSILON
    t = 0

    #########################################################################
    #                              开始训练                                 #
    #########################################################################
    # 记录每轮平均得分的容器
    scores = []
    all_turn_scores = []
    while "flappy bird" != "angry bird":
        # 开始游戏循环
        ######################################################
        ##########首先，按照贪婪策略选择一个行动 ##################
        s = torch.from_numpy(s_t).type(torch.FloatTensor).requires_grad_(True)
        s = s.cuda() if use_cuda else s
        s = s.view(-1, s.size()[0], s.size()[1], s.size()[2])
        # 获取当前时刻的游戏画面，输入到神经网络中
        readout, h_fc1 = net(s)
        # 神经网络产生的输出为readout：选择每一个行动的预期Q值
        readout = readout.cpu() if use_cuda else readout
        # readout为一个二维向量，分别对应每一个动作的预期Q值
        readout_t = readout.data.numpy()[0]

        # 按照epsilon贪婪策略产生小鸟的行动，即以epsilon的概率随机输出行动或者以
        # 1-epsilon的概率按照预期输出最大的Q值给出行动
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if t % FRAME_PER_ACTION == 0:
            # 如果当前帧可以行动，则
            if random.random() <= epsilon:
                # 产生随机行动
                # print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
            else:
                # 选择神经网络判断的预期Q最大的行动
                action_index = np.argmax(readout_t)
            a_t[action_index] = 1
        else:
            a_t[0] = 1  # do nothing

        # 模拟退火：让epsilon开始降低
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        #########################################################################
        ##########其次，将选择好的行动输入给游戏引擎，并得到下一帧的状态 ###################
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        # 返回的x_t1_colored为游戏画面，r_t为本轮的得分，terminal为游戏在本轮是否已经结束

        # 记录一下每一步的成绩
        scores.append(r_t)
        if terminal:
            # 当游戏结束的时候，计算一下本轮的总成绩，并将总成绩存储到all_turn_scores中
            all_turn_scores.append(sum(scores))
            scores = []

        # 对游戏的原始画面做相应的处理，从而变成一张80*80的，朴素的（无背景画面）的图
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (1, 80, 80))
        # 将当前帧的画面和前三帧的画面合并起来作为Agent获得的环境反馈结果
        s_t1 = np.append(x_t1, s_t[:3, :, :], axis=0)
        # 生成一个训练数据，分别将本帧的输入画面s_t,本帧的行动a_t，得到的环境回报r_t以及环境被转换的新状态s_t1存到D中
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            # 如果D中的元素已满，则扔掉最老的一条训练数据
            D.popleft()

        #########################################################################
        ##########最后，当运行周期超过一定次数后开始训练神经网络 ###################
        if t > OBSERVE:
            # 从D中随机采样出一个batch的训练数据
            minibatch = random.sample(D, BATCH)
            optimizer.zero_grad()

            # 将这个batch中的s变量都分别存放到列表中
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            # 接下来，要根据s_j1_batch，神经网络给出预估的未来Q值

            s = torch.tensor(np.array(s_j1_batch, dtype=float), dtype=torch.float, requires_grad=True)
            s = s.cuda() if use_cuda else s
            readout, h_fc1 = net(s)
            readout = readout.cpu() if use_cuda else readout
            readout_j1_batch = readout.data.numpy()
            # readout_j1_batch存储了一个minibatch中的所有未来一步的Q预估值
            # 根据Q的预估值，当前的反馈r，以及游戏是否结束，更新待训练的目标函数值
            y_batch = []
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # 当游戏结束的时候，则用环境的反馈作为目标，否则用下一状态的Q值＋本期的环境反馈
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            # 开始梯度更新
            y = torch.tensor(y_batch, dtype=torch.float, requires_grad=True)
            a = torch.tensor(a_batch, dtype=torch.float, requires_grad=True)
            s = torch.tensor(np.array(s_j_batch, dtype=float), dtype=torch.float, requires_grad=True)
            if use_cuda:
                y = y.cuda()
                a = a.cuda()
                s = s.cuda()
            # 计算s_j_batch的Q值
            readout, h_fc1 = net(s)
            readout_action = readout.mul(a).sum(1)
            # 根据s_j_batch下所选择的预估Q和目标y的Q值的差来作为损失函数训练网络
            loss = criterion(readout_action, y)
            loss.backward()
            optimizer.step()
            if t % 1000 == 0:
                print('损失函数：', loss)

        # 将状态更新一次，时间步＋1
        s_t = s_t1
        t += 1

        # 每隔 10000 次循环，存储一下网络
        if t % 10000 == 0:
            torch.save(net, 'saving_nets/' + GAME + '-dqn' + str(t) + '.txt')

        # 状态信息的转化，基本分为Observe，explore和train三个阶段
        # Observe没有训练，explore开始训练，并且开始模拟退火，train模拟退火结束
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        # 打印当前运行的一些基本数据，分别输出到屏幕以及log文件中
        if t % 1000 == 0:
            sss = "时间步 {}/ 状态 {}/ Epsilon {:.2f}/ 行动 {}/ 奖励 {}/ Q_MAX {:e}/ 轮得分 {:.2f}".format(
                t, state, epsilon, action_index, r_t, np.max(readout_t), np.mean(all_turn_scores[-1000:]))
            print(sss)
            f = open('log_file.txt', 'a')
            f.write(sss + '\n')
            f.close()
    # write info to files
# if __name__=='__main__':
#     GAME = 'bird' # 游戏名称
#     ACTIONS = 2 # 有效输出动作的个数
#     GAMMA = 0.99 # 强化学习中未来的衰减率
#     OBSERVE = 10000. # 训练之前的时间步，需要先观察10000帧
#     EXPLORE = 3000000. # 退火所需的时间步，所谓的退火就是指随机选择率epsilon逐渐变小
#     FINAL_EPSILON = 0.0001 # epsilon的最终值
#     INITIAL_EPSILON = 0.1 # epsilon的初始值
#     REPLAY_MEMORY = 50000 # 最多记忆多少帧训练数据
#     BATCH = 32 # 每一个批次的数据记录条数
#     FRAME_PER_ACTION = 1 # 每间隔多少时间完成一次有效动作的输出
#     train_model(GAME,ACTIONS,GAMMA,OBSERVE,EXPLORE,FINAL_EPSILON,INITIAL_EPSILON,REPLAY_MEMORY,BATCH,FRAME_PER_ACTION)
