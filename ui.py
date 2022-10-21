import os
import sys

import PySide6
from PySide6 import QtCore, QtWidgets

import playGame
import train_Model

dirname = os.path.dirname(PySide6.__file__)
plugin_path = os.path.join(dirname, 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path


class Window(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.top = QtWidgets.QHBoxLayout()

        self.paddingRight = QtWidgets.QVBoxLayout()
        self.paddingRight.addWidget(QtWidgets.QTextBrowser(enabled=False))
        self.parameter = QtWidgets.QVBoxLayout()

        self.para_GAMMA = QtWidgets.QHBoxLayout()
        self.gamma_text = QtWidgets.QLabel("gamma:", alignment=QtCore.Qt.AlignLeft)
        self.gamma_in = QtWidgets.QLineEdit()
        self.para_GAMMA.addWidget(self.gamma_text)
        self.para_GAMMA.addWidget(self.gamma_in)

        self.para_OBSERVE = QtWidgets.QHBoxLayout()
        self.ob_text = QtWidgets.QLabel("observe:", alignment=QtCore.Qt.AlignLeft)
        self.ob_in = QtWidgets.QLineEdit()
        self.para_OBSERVE.addWidget(self.ob_text)
        self.para_OBSERVE.addWidget(self.ob_in)

        self.para_EXPLORE = QtWidgets.QHBoxLayout()
        self.ex_text = QtWidgets.QLabel("explore:", alignment=QtCore.Qt.AlignLeft)
        self.ex_in = QtWidgets.QLineEdit()
        self.para_OBSERVE.addWidget(self.ex_text)
        self.para_OBSERVE.addWidget(self.ex_in)

        self.para_Final_Epsilon = QtWidgets.QHBoxLayout()
        self.fe_text = QtWidgets.QLabel("final Epsilon:", alignment=QtCore.Qt.AlignLeft)
        self.fe_in = QtWidgets.QLineEdit()
        self.para_Final_Epsilon.addWidget(self.fe_text)
        self.para_Final_Epsilon.addWidget(self.fe_in)

        self.para_Initial_Epsilon = QtWidgets.QHBoxLayout()
        self.ie_text = QtWidgets.QLabel("initial Epsilon:", alignment=QtCore.Qt.AlignLeft)
        self.ie_in = QtWidgets.QLineEdit()
        self.para_Initial_Epsilon.addWidget(self.ie_text)
        self.para_Initial_Epsilon.addWidget(self.ie_in)

        self.parameter.addLayout(self.para_GAMMA)
        self.parameter.addLayout(self.para_OBSERVE)
        self.parameter.addLayout(self.para_EXPLORE)
        self.parameter.addLayout(self.para_Initial_Epsilon)
        self.parameter.addLayout(self.para_Final_Epsilon)

        self.top.addLayout(self.parameter)
        self.top.addLayout(self.paddingRight)

        self.startBtn = QtWidgets.QPushButton("开始")
        self.exitBtn = QtWidgets.QPushButton("结束")
        self.playBtn = QtWidgets.QPushButton("手打游戏")
        self.loadBtn = QtWidgets.QPushButton("加载模型")

        self.text = QtWidgets.QLabel("AI打游戏 Flappy Bird", alignment=QtCore.Qt.AlignCenter)

        self.output = QtWidgets.QTextBrowser()
        self.btnBox = QtWidgets.QHBoxLayout()
        self.btnBox.addWidget(self.startBtn)
        self.btnBox.addWidget(self.loadBtn)
        self.btnBox.addWidget(self.playBtn)
        self.btnBox.addWidget(self.exitBtn)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.text)
        self.layout.addLayout(self.top)
        self.layout.addWidget(self.output)
        self.layout.addLayout(self.btnBox)
        self.startBtn.clicked.connect(self.start)
        self.exitBtn.clicked.connect(self.exit)
        self.loadBtn.clicked.connect(self.loadMdl)

    @QtCore.Slot()
    def exit(self):
        sys.exit(0)

    @QtCore.Slot()
    def start(self):
        train_Model.train_model(GAME='bird', ACTIONS=2, GAMMA=float(self.gamma_in.text()),
                                OBSERVE=float(self.ob_in.text()),
                                EXPLORE=float(self.ex_in.text()), FINAL_EPSILON=float(self.fe_in.text()),
                                INITIAL_EPSILON=float(self.fe_in.text()),
                                REPLAY_MEMORY=50000, BATCH=32, FRAME_PER_ACTION=1)

    @QtCore.Slot()
    def loadMdl(self):
        message = QtWidgets.QInputDialog("请输入模型路径:", parent=self)

        pass

    @QtCore.Slot()
    def play(self):
        playGame.PlayGame()
