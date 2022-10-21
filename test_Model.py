# coding=utf-8
import random

import numpy as np
from IPython.display import clear_output

import game_Model as gm
from cnn_Model import *


def test_model(file_path):
    use_cuda = torch.cuda.is_available()  # 检测本台机器中是否有GPU
    net = torch.load(file_path)
    FINAL_EPSILON = 0.0001  # epsilon的最终值
    FRAME_PER_ACTION = 1  # 每间隔多少时间完成一次有效动作的输出

    net = net.cuda() if use_cuda else net

    # 开启一个游戏进程，开始与游戏引擎通话
    game_state = gm.GameState()
    ACTIONS = 2  # 有效输出动作的个数

    # 将游戏设置为初始状态，并获得一个80*80的游戏湖面
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)

    # 将初始的游戏画面叠加成4张作为神经网络的初始输入状态s_t
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=0)

    # 设置初始的epsilon（采取随机行动的概率），并准备训练
    epsilon = FINAL_EPSILON
    t = 0  # 记录每轮平均得分的容器
    scores = []
    all_turn_scores = []

    while "flappy bird" != "angry bird":
        # 开始游戏循环
        ######################################################
        ##########首先，按照贪婪策略选择一个行动 ##################
        s = torch.from_numpy(s_t).type(torch.FloatTensor).requires_grad_(False)
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
        s_t = s_t1
        t += 1
        clear_output(wait=True)

# if __name__=='__main__':
#     file_path = 'final_model.mdl'
#     test_model(file_path)
