import random
from itertools import cycle

import pygame

import game_Resources as gs


class GameState:
    FPS = 30  # 帧率
    SCREENWIDTH = 288  # 屏幕的宽度
    SCREENHEIGHT = 512  # 屏幕的高度

    pygame.init()  # 游戏初始化
    FPSCLOCK = pygame.time.Clock()  # 定义程序时钟
    SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))  # 定义屏幕对象
    pygame.display.set_caption('Flappy Bird')  # 设定窗口名称

    IMAGES, SOUNDS, HITMASKS = gs.load()  # 加载游戏资源
    PIPEGAPSIZE = 100  # 定义两个水管之间的宽度
    BASEY = SCREENHEIGHT * 0.79  # 设定基地的高度

    # 设定小鸟属性：宽度、高度等
    PLAYER_WIDTH = IMAGES['player'][0].get_width()
    PLAYER_HEIGHT = IMAGES['player'][0].get_height()

    # 设定水管属性：高度、宽度
    PIPE_WIDTH = IMAGES['pipe'][0].get_width()
    PIPE_HEIGHT = IMAGES['pipe'][0].get_height()

    # 背景宽度
    BACKGROUND_WIDTH = IMAGES['background'].get_width()
    # cycle只接受可以迭代的参数，如列表，元组，字符串,该函数会对可迭代的所有元素进行循环：
    PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])

    def getRandomPipe(self):
        # 随机生成管道的函数
        """returns a randomly generated pipe"""
        # 两个管道之间的竖直间隔从下列数中直接取
        gapYs = [20, 30, 40, 50, 60, 70, 80, 90]
        index = random.randint(0, len(gapYs) - 1)
        gapY = gapYs[index]

        # 设定新生成管道的位置
        gapY += int(GameState.BASEY * 0.2)
        pipeX = GameState.SCREENWIDTH + 10

        # 返回管道的坐标
        return [
            {'x': pipeX, 'y': gapY - GameState.PIPE_HEIGHT},  # upper pipe
            {'x': pipeX, 'y': gapY + GameState.PIPEGAPSIZE},  # lower pipe
        ]

    def checkCrash(player, upperPipes, lowerPipes):
        # 检测碰撞的函数，基本思路为：将每一个物体都看作是一个矩形区域，然后检查两个矩形区域是否有碰撞
        # 检查碰撞是细到每个对象的图像蒙板级别，而不单纯是看矩形之间的碰撞
        """returns True if player collders with base or pipes."""
        pi = player['index']
        player['w'] = GameState.IMAGES['player'][0].get_width()
        player['h'] = GameState.IMAGES['player'][0].get_height()

        # 检查小鸟是否碰撞到了地面
        if player['y'] + player['h'] >= GameState.BASEY - 1:
            return True
        else:
            # 检查小鸟是否与管道碰撞
            playerRect = pygame.Rect(player['x'], player['y'],
                                     player['w'], player['h'])

            for uPipe, lPipe in zip(upperPipes, lowerPipes):
                # 上下管道矩形
                uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], GameState.PIPE_WIDTH, GameState.PIPE_HEIGHT)
                lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], GameState.PIPE_WIDTH, GameState.PIPE_HEIGHT)

                # 获得每个元素的蒙板
                pHitMask = GameState.HITMASKS['player'][pi]
                uHitmask = GameState.HITMASKS['pipe'][0]
                lHitmask = GameState.HITMASKS['pipe'][1]

                # 检查是否与上下管道相撞
                uCollide = GameState.pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
                lCollide = GameState.pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

                if uCollide or lCollide:
                    return True

        return False

    def pixelCollision(rect1, rect2, hitmask1, hitmask2):
        """在像素级别检查两个物体是否发生碰撞"""
        rect = rect1.clip(rect2)

        if rect.width == 0 or rect.height == 0:
            return False

        # 确定矩形框，并针对矩形框中的每个像素进行循环，查看两个对象是否碰撞
        x1, y1 = rect.x - rect1.x, rect.y - rect1.y
        x2, y2 = rect.x - rect2.x, rect.y - rect2.y

        for x in range(rect.width):
            for y in range(rect.height):
                if hitmask1[x1 + x][y1 + y] and hitmask2[x2 + x][y2 + y]:
                    return True
        return False

    def __init__(self):
        # 初始化
        # 初始成绩、玩家索引、循环迭代都为0
        self.score = self.playerIndex = self.loopIter = 0

        # 设定玩家的初始位置
        self.playerx = int(GameState.SCREENWIDTH * 0.2)
        self.playery = int((GameState.SCREENHEIGHT - GameState.PLAYER_HEIGHT) / 2)
        self.basex = 0
        # 地面的初始移位
        self.baseShift = GameState.IMAGES['base'].get_width() - GameState.BACKGROUND_WIDTH

        # 生成两个随机的水管
        newPipe1 = GameState.getRandomPipe(self)
        newPipe2 = GameState.getRandomPipe(self)

        # 设定初始水管的位置x，y坐标
        self.upperPipes = [
            {'x': GameState.SCREENWIDTH, 'y': newPipe1[0]['y']},
            {'x': GameState.SCREENWIDTH + (GameState.SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
        ]
        self.lowerPipes = [
            {'x': GameState.SCREENWIDTH, 'y': newPipe1[1]['y']},
            {'x': GameState.SCREENWIDTH + (GameState.SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
        ]

        # 定义玩家的属性
        self.pipeVelX = -4
        self.playerVelY = 0  # 小鸟在y轴上的速度，初始设置维playerFlapped
        self.playerMaxVelY = 10  # Y轴上的最大速度, 也就是最大的下降速度
        self.playerMinVelY = -8  # Y轴向上的最大速度
        self.playerAccY = 1  # 小鸟往下落的加速度
        self.playerFlapAcc = -9  # 扇动翅膀的加速度
        self.playerFlapped = False  # 玩家是否煽动了翅膀

    def frame_step(self, input_actions):
        # input_actions是一个行动数组，分别存储了0或者1两个动作的激活情况
        # 游戏每一帧的循环
        pygame.event.pump()

        # 每一步的默认回报
        reward = 0.1
        terminal = False

        # 限定每一帧只能做一个动作
        if sum(input_actions) != 1:
            raise ValueError('Multiple input actions!')

        # input_actions[0] == 1: 对应什么都不做
        # input_actions[1] == 1: 对应小鸟煽动了翅膀
        if input_actions[1] == 1:
            # 小鸟煽动翅膀向上
            if self.playery > -2 * GameState.PLAYER_HEIGHT:
                self.playerVelY = self.playerFlapAcc
                self.playerFlapped = True
                # SOUNDS['wing'].play()

        # 检查是否通过了管道，如果通过，则增加成绩
        playerMidPos = self.playerx + GameState.PLAYER_WIDTH / 2
        for pipe in self.upperPipes:
            pipeMidPos = pipe['x'] + GameState.PIPE_WIDTH / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                self.score += 1
                # SOUNDS['point'].play()
                reward = 1

        # playerIndex轮换
        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = next(GameState.PLAYER_INDEX_GEN)
        self.loopIter = (self.loopIter + 1) % 30
        self.basex = -((-self.basex + 100) % self.baseShift)

        # 小鸟运动
        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False
        self.playery += min(self.playerVelY, GameState.BASEY - self.playery - GameState.PLAYER_HEIGHT)
        if self.playery < 0:
            self.playery = 0

        # 管道的移动
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

        # 当管道快到左侧边缘的时候，产生新的管道
        if 0 < self.upperPipes[0]['x'] < 5:
            newPipe = GameState.getRandomPipe(self)
            self.upperPipes.append(newPipe[0])
            self.lowerPipes.append(newPipe[1])

        # 当第一个管道移出屏幕的时候，就把它删除
        if self.upperPipes[0]['x'] < -GameState.PIPE_WIDTH:
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)

        # 检查碰撞
        isCrash = GameState.checkCrash({'x': self.playerx, 'y': self.playery,
                                        'index': self.playerIndex},
                                       self.upperPipes, self.lowerPipes)
        # 如果有碰撞发生，则游戏结束，terminal＝True
        if isCrash:
            # SOUNDS['hit'].play()
            # SOUNDS['die'].play()
            terminal = True
            self.__init__()
            reward = -1

        # 将所有角色都根据每个角色的坐标画到屏幕上
        GameState.SCREEN.blit(GameState.IMAGES['background'], (0, 0))

        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            GameState.SCREEN.blit(GameState.IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            GameState.SCREEN.blit(GameState.IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        GameState.SCREEN.blit(GameState.IMAGES['base'], (self.basex, GameState.BASEY))

        # print score so player overlaps the score
        # showScore(self.score)
        GameState.SCREEN.blit(GameState.IMAGES['player'][self.playerIndex],
                              (self.playerx, self.playery))

        # 将当前的游戏屏幕生成一个二维画面返回
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        GameState.FPSCLOCK.tick(GameState.FPS)
        # print self.upperPipes[0]['y'] + PIPE_HEIGHT - int(BASEY * 0.2)
        # 该函数的输出有三个变量：游戏当前帧的游戏画面，当前获得的游戏得分，游戏是否已经结束
        return image_data, reward, terminal
