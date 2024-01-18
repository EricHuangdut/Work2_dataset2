import queue
import random

import requests
import openpyxl
from openpyxl import Workbook
import numpy as np
from tqdm import tqdm  # 进度条设置
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib
import math
import copy
import operator
import csv
import pandas as pd

matplotlib.use('TkAgg')
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


#
class GaMultiobjective(object):
    def __init__(self):
        # =========定义群=========
        self.NP = 500  # 种群个数 尝试更大的种群
        self.max_gen = 900  # 最大迭代次数
        self.max_mission = 14  # 无人机最多执行的任务个数
        # self.min_x = [-10] 算法测试用例中的范围下限
        # self.max_x = [10]  算法测试用例中的范围上限
        self.Pc = 0.7  # 交叉率
        self.min_Pc = 0.5
        self.Pm = 0.3  # 变异率
        self.max_Pm = 0.5
        self.maxsolution = 5000
        self.N = 1  # 变量个数
        self.num_target = 40  # 区域内的目标数，可调整
        self.num_uav = 15  # 无人机集群的数量，可调整
        self.len_target = 6  # 编码时，每len_target位存储1个目标优先级信息
        self.len_uav = 4  # 编码时，每len_uav位存储1个无人机优先级信息
        #    self.L = self.num_uav * self.num_target * 5  # 二进制数字串长度，这个数字串存储了无人机的各种行为
        self.Len = 2 * self.num_target * 6 + 2 * self.num_target * 4  # *2是因为，2*num_target代表任务总数，前60%排任务顺序
        # 后40%为这个任务顺序每个分配1个uav
        self.xmin = 0  # 区域坐标范围，可调整
        self.ymin = 0
        self.xmax = 700
        self.ymax = 700
        self.singletasktarget = 31
        #    self.singletasktarget = 10 #测试用数据
        #  self.ptmin = 10 #评分上下限，目前是对每1个目标来生成1个目标价值评分。
        #  self.ptmax = 100
        self.maxpath = 5000
        self.Mcost = 0.05
        self.Dcost = 0.2
        self.Ccost = 0.5
        self.resourcesA = 1000  # 无人机总资源
        self.resourcesB = 2000  # 无人机总资源
        self.resourcesC = 2000  # 无人机总资源
        self.velocityA = 150  # 侦察无人机速度
        self.velocityB = 120  # 打击无人机速度
        self.velocityC = 180  # 一体无人机速度
        self.maxtime = 40  # 无人机最大巡航时间
        self.priceA = 80  # 侦察无人机价值
        self.priceB = 100  # 打击无人机价值
        self.priceC = 150  # 一体无人机价值
        self.map = [[]]  # 存放x个无人机初始位置和y个目标这x+y个点之间两两距离的2维向量
        self.pos = [[]]  # 存放目标的初始位置
        self.gain = [[]]  # 存放目标的价值评分
        self.threat = []  # 存放目标的威胁程度
        self.ex = [2, 4, 7, 12, 16, 18, 26, 31, 34]
        # self.parent = np.random.randint(0, 2, (self.NP, self.N, self.L))  # 随机获得二进制 初始种群f.shape (50,1, 20) .1表示有1个变量

    # 计算两点间距离
    def dis(self, x1, x2, y1, y2):
        dist = np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))
        return dist

    # 对初始数据集的计算
    def cal_distance(self, pos1):
        map = [[0 for i in range(0, self.num_target)] for i in range(0, self.num_uav + self.num_target)]
        map = np.array(map)
        for i in range(0, self.num_uav + self.num_target):
            for j in range(0, self.num_target):
                map[i, j] = self.dis(pos1[i, 0], pos1[j + self.num_uav, 0], pos1[i, 1], pos1[j + self.num_uav, 1])
        return map

    # 对任意数据集的计算
    def CAL_distance(self, pos1):
        map = [[0 for i in range(0, len(pos1))] for i in range(0, len(pos1))]
        map = np.array(map)
        for i in range(0, len(pos1)):
            for j in range(0, len(pos1)):
                map[i, j] = self.dis(pos1[i, 0], pos1[j, 0], pos1[i, 1], pos1[j, 1])
        return map

    # 生成初始位置
    def getpos(self):
        # 生成目标点的随机位置
        pos = [[0, 0] for i in range(0, self.num_target + self.num_uav)]
        # 这里的目的是去掉一些目标点距离出发位置过近的极端情况。
        pos = np.array(pos)
        # dist = [0 for i in range(0,self.num_uav)]

        # 生成3种无人机的初始位置
        num = int(self.num_uav / 3)  # 3种无人机
        for i in range(0, num):
            pos[i, 0] = 0
            pos[i, 1] = self.ymax  # 侦察无人机在（0，ymax）
        for i in range(num, 2 * num):
            pos[i, 0] = 0
            pos[i, 1] = 0  # 打击无人机在（0，0）
        for i in range(2 * num, 3 * num):
            pos[i, 0] = self.xmax
            pos[i, 1] = 0
        # for i in range(0,self.num_uav):
        #    dist[i] = self.dis(pos[i,0],pos[tp+self.num_uav,0],pos[i,1],pos[tp+self.num_uav,1])
        #    while(dist[i]<100):
        #        pos[i,0] = self.xmin + (self.xmax - self.xmin) * np.random.random()
        #        pos[i,1] = self.ymin + (self.ymax - self.ymin) * np.random.random()
        #        dist[i] = self.dis(pos[i, 0], pos[tp + self.num_uav, 0], pos[i, 1], pos[tp + self.num_uav, 1])
        init = True
        if (init):
            pos[15, 0] = 600.83
            pos[15, 1] = 321.33
            pos[16, 0] = 315.93
            pos[16, 1] = 35.61
            pos[17, 0] = 499.51
            pos[17, 1] = 582.19
            pos[18, 0] = 113.10
            pos[18, 1] = 473.75
            pos[19, 0] = 212.67
            pos[19, 1] = 305.43
            pos[20, 0] = 420.99
            pos[20, 1] = 345.68
            pos[21, 0] = 160.65
            pos[21, 1] = 115.69
            pos[22, 0] = 611.97
            pos[22, 1] = 167.39
            pos[23, 0] = 369.55
            pos[23, 1] = 651.00
            pos[24, 0] = 36.05
            pos[24, 1] = 665.32
            pos[25, 0] = 137.08
            pos[25, 1] = 252.02
            pos[26, 0] = 258.73
            pos[26, 1] = 608.61
            pos[27, 0] = 194.74
            pos[27, 1] = 168.65
            pos[28, 0] = 341.20
            pos[28, 1] = 428.85
            pos[29, 0] = 285.90
            pos[29, 1] = 541.63
            pos[30, 0] = 466.01
            pos[30, 1] = 288.82
            pos[31, 0] = 456.66
            pos[31, 1] = 443.61
            pos[32, 0] = 630.20
            pos[32, 1] = 587.80
            pos[33, 0] = 414.07
            pos[33, 1] = 488.37
            pos[34, 0] = 63.68
            pos[34, 1] = 25.71
            pos[35, 0] = 505.83
            pos[35, 1] = 151.01
            pos[36, 0] = 283.13
            pos[36, 1] = 289.23
            pos[37, 0] = 55.68
            pos[37, 1] = 183.93
            pos[38, 0] = 574.15
            pos[38, 1] = 582.28
            pos[39, 0] = 563.05
            pos[39, 1] = 408.13
            pos[40, 0] = 188.44
            pos[40, 1] = 490.16
            pos[41, 0] = 375.14
            pos[41, 1] = 585.38
            pos[42, 0] = 583.60
            pos[42, 1] = 491.11
            pos[43, 0] = 240.63
            pos[43, 1] = 412.94
            pos[44, 0] = 497.27
            pos[44, 1] = 507.26
            pos[45, 0] = 533.83
            pos[45, 1] = 67.17
            pos[46, 0] = 592.88
            pos[46, 1] = 516.02
            pos[47, 0] = 323.61
            pos[47, 1] = 274.89
            pos[48, 0] = 181.82
            pos[48, 1] = 329.21
            pos[49, 0] = 126.30
            pos[49, 1] = 666.20
            pos[50, 0] = 244.22
            pos[50, 1] = 76.28
            pos[51, 0] = 70.47
            pos[51, 1] = 303.19
            pos[52, 0] = 536.34
            pos[52, 1] = 665.68
            pos[53, 0] = 393.43
            pos[53, 1] = 215.46
            pos[54, 0] = 371.57
            pos[54, 1] = 148.23

        return pos

    def getmap(self, pos):
        self.map = self.cal_distance(pos)
        return self.map

    def GETmap(self, pos):
        self.map = self.CAL_distance(pos)
        return self.map

        # 初始化威胁度和威胁半径 同样根据之前的代码做对比，目前定为固定值，基础函数5

    def threat_initialize(self, pos):
        # 初始威胁度 #
        #    threat_radius = [10 + np.random.random() * 40 for i in range(0, self.num_target)]
        threat_radius = [0.00 for i in range(0, self.num_target)]
        initT = True
        if (initT):
            threat_radius[0] = 24.29
            threat_radius[1] = 17.71
            threat_radius[2] = 21.59
            threat_radius[3] = 11.39
            threat_radius[4] = 29.94
            threat_radius[5] = 22.34
            threat_radius[6] = 21.64
            threat_radius[7] = 28.80
            threat_radius[8] = 23.84
            threat_radius[9] = 10.54
            threat_radius[10] = 19.29
            threat_radius[11] = 10.31
            threat_radius[12] = 20.48
            threat_radius[13] = 19.08
            threat_radius[14] = 16.10
            threat_radius[15] = 21.36
            threat_radius[16] = 22.41
            threat_radius[17] = 26.68
            threat_radius[18] = 18.56
            threat_radius[19] = 16.61
            threat_radius[20] = 12.16
            threat_radius[21] = 10.80
            threat_radius[22] = 18.60
            threat_radius[23] = 20.20
            threat_radius[24] = 11.96
            threat_radius[25] = 29.48
            threat_radius[26] = 17.70
            threat_radius[27] = 14.63
            threat_radius[28] = 18.40
            threat_radius[29] = 15.12
            threat_radius[30] = 29.73
            threat_radius[31] = 22.01
            threat_radius[32] = 23.00
            threat_radius[33] = 10.67
            threat_radius[34] = 26.18
            threat_radius[35] = 11.18
            threat_radius[36] = 26.24
            threat_radius[37] = 26.64
            threat_radius[38] = 25.88
            threat_radius[39] = 22.44

        return threat_radius

    def settimewindow(self, timewindow):
        initT = True

        if (initT):

            for i in range(self.num_target):
                timewindow[i, 0] = 5
                timewindow[i, 1] = 9999
                if i not in self.ex:
                    timewindow[i, 1] = 10

        return timewindow

    # 已完成验证，可以实现任务序列的排序
    # 有向图排序，
    def reorder(self, task):
        #    print(task)
        #    temp = [[]] #2维，存放排序后的task
        ord = [0 for i in range(2 * self.num_target)]  # 1维 把排序后的task只提取优先级返回
        #    ord = np.array(ord)
        temp = sorted(task, key=operator.itemgetter(1))  # 按照key=2也就是优先级来排序
        temp = np.array(temp)
        #    print(temp)

        for i in range(0, (2 * self.num_target)):
            ord[i] = temp[i, 0]  # 根据key=1，也就是任务序号，排出ord

        #    print("ordinary ord=",ord)

        # 排序后，对具体的任务进行重排，确保对同一个任务，其侦察任务在干扰任务之前完成
        index = 0  # 标记当前任务
        id = 0  # 侦察任务下标
        ic = 0  # 干扰任务下标
        for i in range(0, self.num_target):
            for j in range(0, 2 * self.num_target):
                if (ord[j] == index):
                    id = j
                if (ord[j] == index + 1):
                    ic = j
            if (id > ic):
                t = ord[id]
                ord[id] = ord[ic]
                ord[ic] = t
            index = index + 2

        #    print("ord= ", ord)

        return ord

    # 每个无人机可以做有限个任务

    def reallo(self, allo, len):
        #    print("allo = ",allo)
        temp = [[0 for i in range(0, 2)] for i in range(0, 2 * self.num_target - len)]  # 2维，存放排序后的allo方案
        cnt = [0 for i in range(0, self.num_uav)]  # 1维，存放每个无人机已经被分配的任务数量
        cnt = np.array(cnt)
        num_u = 0
        temp = sorted(allo, key=operator.itemgetter(0), reverse=True)  # 这里按无人机编号排
        temp = np.array(temp)

        m = [9999 for i in range(0, 2 * self.num_target - len)]
        m = np.array(m)
        val = [9999 for i in range(0, 2 * self.num_target - len)]
        val = np.array(val)
        pos = 0
        for i in range(0, 2 * self.num_target - len):
            u = temp[i, 0]
            cnt[u] = cnt[u] + 1
            if (cnt[u] > self.max_mission):  # 存储分配不下的任务序列号
                temp[i, 0] = 9999  # 设定为无主的任务
                cnt[u] = cnt[u] - 1
            #    val[pos] = temp[i,1]
            #    m[pos] = i
            #    pos = pos + 1
        #    print(cnt)
        #    print(temp)
        # 再遍历一次
        for j in range(0, 2 * self.num_target - len):

            if (temp[j, 0] == 9999):

                for k in range(0, self.num_uav):  # 这里的问题是，如果分配了一个任务，下一个任务还得分配给同一个无人机的时候会跳过
                    if (temp[j, 1] % 2 == 0):
                        if ((cnt[k] < self.max_mission) and (
                                (k < (self.num_uav / 3)) or (k >= (self.num_uav / 3 * 2)))):
                            temp[j, 0] = k
                            cnt[k] = cnt[k] + 1
                            k = k - 1
                    if (temp[j, 1] % 2 == 1):
                        if ((cnt[k] < self.max_mission) and (k >= (self.num_uav / 3))):
                            temp[j, 0] = k
                            cnt[k] = cnt[k] + 1
                            k = k - 1

        #    print("cnt=",cnt)
        #    print(temp)
        """
        print(cnt)
        print(val)
        print(m)
        for j in range(0,2*self.num_target): #j是m的位置索引
            if(val[j]<999):
                for k in range(0,self.num_uav):  #这里k是无人机编号索引
                    if(val[j]%2 == 0): #是个侦察类任务
                        if(((k<3) or (k>=6)) and (cnt[k]<self.max_mission)): #可以被分配任务
                            temp[m[j],0] = k
                            cnt[k] = cnt[k] + 1
                            print("cnt=",cnt)
                    if (val[j] % 2 == 1):  # 是个干扰类任务
                        if ((k >= 3) and (cnt[k] < self.max_mission)):  # 可以被分配任务
                            temp[m[j], 0] = k
                            cnt[k] = cnt[k] + 1
                            print("cnt=",cnt)
        print(temp)
        """
        """
           # 需要把任务分配给其他无人机
            if(cnt[u]>4): #当前无人机任务超标
                if(temp[i,1]%2 == 0): #是一个侦察任务,分配给012678,从8开始搜索
                    #这里第一次进来的时候uav = 8
                    if((uav>=6) or (uav<=2)): #无人机编号合适,不合适直接跳过,跳过则-1
                        if(cnt[uav]<4): #且能够继续执行任务，不能执行任务直接跳过
                            temp[i,0] = uav
                            print("temp uav=",temp[i,0],uav)
                            cnt[uav] = cnt[uav] + 1
                            cnt[u] = cnt[u] - 1 #任务数减回上限
                            uav = 8
                        if(cnt[uav]==4):
                            uav = uav - 1


                if (temp[i, 1] % 2 == 1):  # 是一个干扰任务,分配给345678
                    if (uav >= 3):  # 无人机编号合适,不合适直接跳过
                        if (cnt[uav] < 4):  # 且能够继续执行任务，不能执行任务直接跳过
                            temp[i, 0] = uav
                            print("temp uav=", temp[i,0], uav)
                            cnt[uav] = cnt[uav] + 1
                            cnt[u] = cnt[u] - 1  # 任务数减回上限
                            uav = 8
                        if (cnt[uav] == 4):
                            uav = uav - 1

        """
        temp1 = [[0 for i in range(0, 2)] for i in range(0, 2 * self.num_target - len)]  # 2维，存放排序后的allo方案
        temp1 = sorted(temp, key=operator.itemgetter(0))  # 检索结束后再重新排一次
        temp1 = np.array(temp1)
        #    print("sorted =",temp1)
        if (temp1[23, 0] == 9999):
            for k in range(0, self.num_uav):  # 这里的问题是，如果分配了一个任务，下一个任务还得分配给同一个无人机的时候会跳过
                if (temp1[j, 1] % 2 == 0):
                    if ((cnt[k] < 4) and ((k < 3) or (k >= 6))):
                        temp1[j, 0] = k

                if (temp1[j, 1] % 2 == 1):
                    if ((cnt[k] < 4) and (k >= 3)):
                        temp1[j, 0] = k

        temp2 = [[0 for i in range(0, 2)] for i in range(0, 2 * self.num_target - len)]  # 2维，存放排序后的allo方案
        temp2 = sorted(temp1, key=operator.itemgetter(0))  # 检索结束后再重新排一次
        temp2 = np.array(temp2)
        #    print("sorted =", temp2)

        return temp2

    # 输入：2进制序列,这里输入的时候是输入1个个体，所以需要对每个个体分别进行一次解码
    # 种群形式是一个2维数组，第2维是一个线性串，所以只需要2层循环
    # 输出：一个1维数组，index为任务编号，值为无人机编号
    # index：奇数为侦察任务，偶数为干扰任务
    def decode(self, f):
        #    print("f = ",f)
        L1 = self.num_target * 6 * 2  # 前半部分长度
        L2 = self.num_target * 4 * 2  # 后半部分长度
        mission = [[0 for i in range(0, 2)] for i in range(0, 2 * self.num_target)]  # 2维，形式是[任务index，优先级]
        order = []
        allo = [[0 for i in range(0, 2)] for i in range(0, 2 * self.num_target)]  # 2维，形式是[任务index，无人机编号]
        allo = np.array(allo)
        mission = np.array(mission)
        order = np.array(order)
        pos = 0  # 指示此时mission的位置
        # 对任务进行排序得到一个序列
        count = 0
        m = 0

        for j in range(0, L1):

            if (count < 6):
                m = f[j] * np.power(2, count) + m  # 转化成10进制,这里m可以正常计算了
                count = count + 1
                if (count == 6):  # 此时需要计算下个任务
                    count = 0
                    mission[pos, 0] = pos  # 任务的实际编号
                    mission[pos, 1] = m  # 第pos+1个任务的优先级是m
                    #    print(mission[pos])
                    m = 0
                    pos = pos + 1
        #   print(mission)
        taskord = self.reorder(mission)  # 需要写一个reorder函数对mission排序
        # task的输出结果为一个1维的量，每个位置上的值代表任务序列

        # 排除掉不需要干扰的目标
        # 这里这个数据集要改的
        # 属于目标2，4，7，12，16，18 // 1,3,6,11,15,17
        #    self.ex = [3, 7, 13, 21, 27]  # 属于目标2，4，7，11，14 // 1,3,6,10,13
        task = [0 for i in range(0, 2 * self.num_target - len(self.ex))]
        task = np.array(task)
        ind = 0
        conf = 0
        for i in range(0, len(taskord)):
            for j in range(0, len(self.ex)):
                if (taskord[i] != self.ex[j]):  # 全检索
                    conf = conf
                if (taskord[i] == self.ex[j]):
                    conf = 1
            if (conf == 0):
                task[ind] = taskord[i]
                ind = ind + 1
            conf = 0

        #    print("task =",task)
        pos = 0
        count = 0
        n = 0
        cnt = 0
        # 为无人机分配任务,这里先得到一个初始仅约束无人机任务种类的序列，这里逻辑和已经排好序的task相关
        for i in range(L1, L1 + L2 - len(self.ex) * 4):
            if (cnt < 4):
                n = f[i] * np.power(2, cnt) + n  # 转化成10进制
                cnt = cnt + 1
                if (cnt == 4):
                    cnt = 0
                    # %2=0意味着这是个侦察任务
                    # 对9个无人机，选能执行的6个，即编号1-3和编号7-9，数组里存储的数值是-1的
                    if ((task[count] % 2) == 0):
                        num_u = n % (self.num_uav / 3 * 2)
                        if (num_u >= self.num_uav / 3):  # 选中了一体型无人机
                            num_u = num_u + self.num_uav / 3  # 对应一体型无人机的编号，+3的3 = self.num_uav/3
                        allo[pos, 0] = num_u
                        allo[pos, 1] = task[count]
                        #    print("allo =",allo)
                        pos = pos + 1
                        n = 0

                    # %2=1代表干扰任务
                    # 编号4-9
                    # num_u在这里=3的时候会跳过一组i值导致out of index
                    if ((task[count] % 2) == 1):
                        num_u = n % (self.num_uav / 3 * 2) + self.num_uav / 3  # 取值范围是3-8
                        allo[pos, 0] = num_u
                        allo[pos, 1] = task[count]
                        #    print("allo =", allo)

                        pos = pos + 1

                        n = 0
                    count = count + 1
        Allo = [[0 for i in range(0, 2)] for i in range(0, 2 * self.num_target - len(self.ex))]
        Allo = np.array(Allo)
        for i in range(0, len(Allo)):
            Allo[i, 0] = allo[i, 0]
            Allo[i, 1] = allo[i, 1]

        # 感觉这步可能有点多余，因为要去除不执行的任务，保留一部分
        allo = self.reallo(Allo, len(self.ex))  # 需要写一个reallo函数对allo调整顺序，这里涉及到无人机执行任务数量约束
        # allo的输出结果为一个2维变量，格式为[任务编号，执行的无人机编号]

        Finalorder = [[0 for i in range(0, 2)] for i in range(0, 2 * self.num_target - len(self.ex))]
        Finalorder = np.array(Finalorder)
        for i in range(0, len(task)):
            for j in range(0, len(Finalorder)):
                if (allo[j, 1] == task[i]):
                    Finalorder[i, 0] = allo[j, 0]
                    Finalorder[i, 1] = allo[j, 1]

        #    print("Order = ",Finalorder)
        return Finalorder

    # 目前的工作序列中，无人机的路径顺序是随机的
    # 需要编辑一个函数，根据指定的目标编号决定其路径，以排除比较差的序列情况
    # 在静态的问题中，只需要对该问题进行排序即可
    # 但是问题扩展到动态的时候，需要考虑任务的执行顺序
    # 在这个问题中，根据decode函数进行排序的序列为已经考虑了任务执行1-2次序的序列
    # 对于探测和干扰型无人机，序列的改变不会影响具体的执行顺序，最差情况是在执行任务的时候，同一个任务点的任务1执行很晚，导致任务2起始晚
    # 从而耽误整个个体的总时间，会影响函数3中的时间窗口计算，出很差的结果
    # 或者，在任务3中改变任务先后顺序的判定思路，在解码时采用另一种分配策略

    # 任务序列-目标序列转化
    def targetseq(self, order):
        for i in range(len(order)):
            order[i, 1] = order[i, 1] // 2  # 向下取整即可得到目标点的定位，下标从0开始，比如任务13对应目标6，任务1对应目标0

        return order

    # 提取子map
    # 这里需要读一下pos
    # 输入：无人机-目标点序列（注意不是无人机-任务序号），需要生成子图的uav编号
    # 输出：子图（可以实现）
    # 输出内容下午回来检验一下，就算完成0315的基本工作。
    def submap(self, seq, uavnum, n):
        sub = [0 for i in range(0, n + 1)]  # 存储无人机所要完成的目标序号，这里是map上的序号，转化为任务序号需要-self.num_uav
        sub = np.array(sub)
        p = self.getpos()  # 获取初始数据集
        pointer = 1
        sub[0] = uavnum  # 从0开始的无人机序号下标
        # 提取1个无人机的工作序列并加入pos
        for i in range(len(seq)):
            if (seq[i, 0] == uavnum):  # 序号就是要排的这个无人机的序号
                sub[pointer] = seq[i, 1] + self.num_uav  # 这里加num_uav是为了检索任务点对应位置的下标
                pointer = pointer + 1

        pos = [[0 for i in range(0, 2)] for i in range(len(sub))]
        pos = np.array(pos)

        # 对应数据集中的位置点，输入sub数组中
        for i in range(len(sub)):
            pos[i, 0] = p[sub[i], 0]
            pos[i, 1] = p[sub[i], 1]

        m = self.GETmap(pos)  # GETmap实现正确，已经验证
        #    print("m=",m)
        ans = self.subseq(m, sub, n)
        ans = np.array(ans)
        #    print("ans =",ans)

        index = 0
        for i in range(len(seq)):
            if (seq[i, 0] == uavnum):
                seq[i, 1] = ans[index]
                index = index + 1

        return seq

    # 输入：1个方案中，1个无人机包含所有任务的子图
    # 输出：1维数组，代表无人机的任务执行顺序，以任务目标编号形式返回
    def subseq(self, map, s, n):
        map = np.array(map)
        for i in range(len(map)):
            map[i, i] = 99999  # 防止自己这个点干扰检索
            map[i, 0] = 99999  # 确保不会搜回起点
        #    map[0,i] = 99999
        #    print("original map =",map)
        s = np.array(s)
        seq = [0 for i in range(0, n)]  # 存储结果
        seq = np.array(seq)
        ifallo = [0 for i in range(len(map))]  # 该点是否被搜索过
        ifallo[0] = 1  # 起点默认是已经搜过的，这个map第1行就是无人机初始位置
        ifallo = np.array(ifallo)
        positionindex = 0  # 当前搜索的行数
        # 内层循环找到距离最近的目标，j是同一行有几个可比对的目标
        # 外层循环在找最近的目标，
        for i in range(len(map) - 1):  # 只需要搜索n-1个点，因为第1个是原点
            m = min(map[positionindex, :])  # 找到这行的最小值
            #        print("minimum =",m)
            for j in range(len(map)):
                if ((map[positionindex, j] == m)):  # 找到最小值了，获取其下标也就是第几个目标
                    map[j, positionindex] = 9999  # 排除干扰
                    map[:, j] = 9999
                    positionindex = j
                    seq[i] = positionindex
        #                print("map=",map)

        for i in range(len(seq)):
            seq[i] = s[seq[i]] - self.num_uav

        return seq

    def timewindow_ord(self, order):
        # seq = 最终输出序列
        seq = [[0 for i in range(2)] for i in range(self.num_target + self.singletasktarget)]
        seq = np.array(seq)
        detect_amt = 0
        dis_amt = 0
        # dis_det_match = 干扰无人机任务序列中，可以与侦察型匹配的部分。中间变量
        dis_det_match = [[0 for i in range(2)] for i in range(self.num_target + self.singletasktarget)]
        dis_det_match = np.array(dis_det_match)
        dis_det_array = 0

        # comp_det_match = 一体无人机任务序列中，可以与侦察型匹配的部分，中间变量
        comp_det_match = [[0 for i in range(2)] for i in range(self.num_target + self.singletasktarget)]
        comp_det_match = np.array(comp_det_match)
        comp_det_array = 0

        # comp_dis_match = 干扰无人机任务序列中，可以与一体型匹配的部分。中间变量
        comp_dis_match = [[0 for i in range(2)] for i in range(self.num_target + self.singletasktarget)]
        comp_dis_match = np.array(comp_dis_match)
        comp_dis_array = 0

        # comp_dis_match_2 = 一体无人机任务序列中，可以与干扰匹配的部分。中间变量
        comp_dis_match_2 = [[0 for i in range(2)] for i in range(self.num_target + self.singletasktarget)]
        comp_dis_match_2 = np.array(comp_dis_match_2)
        comp_dis_array_2 = 0

        for i in range(len(order)):
            # 如果是侦察型无人机，序列不做处理
            if (order[i, 0] < (self.num_uav / 3)):
                detect_amt += 1
            # 如果是干扰无人机,分成2类：侦察任务被侦察型完成的，侦察任务被一体型完成的
            if (order[i, 0] >= (self.num_uav / 3) and order[i, 0] < (self.num_uav / 3 * 2)):
                dis_det_match_flag = 0
                for j in range(detect_amt):
                    # 被分配给侦察无人机的目标，在干扰无人机这里可以找到对应干扰任务
                    if (order[i, 1] == order[j, 1]):
                        dis_det_match[dis_det_array] = order[i]
                        dis_det_array += 1
                        dis_det_match_flag = 1
                # 这里走完一遍就可以确认哪些干扰类无人机分配到的任务已经被一体型执行
                if (dis_det_match_flag == 0):
                    comp_dis_match[comp_dis_array] = order[i]
                    comp_dis_array += 1
                dis_amt += 1
            # 如果是一体无人机，区分：
            # 1，需要先执行任务给干扰型无人机做的
            # 2. 需要执行侦察无人机侦察完的目标的
            if (order[i, 0] >= (self.num_uav / 3 * 2)):
                for j in range(detect_amt):
                    # 一体-侦察匹配
                    if (order[i, 1] == order[j, 1]):
                        comp_det_match[comp_det_array] = order[i]
                        comp_det_array += 1

                for k in range(detect_amt, detect_amt + dis_amt):
                    if (order[i, 1] == order[k, 1]):
                        comp_dis_match_2[comp_dis_array_2] = order[i]
                        comp_dis_array_2 += 1
        """
        print("detect amt =",detect_amt)
        print("dis amt =", dis_amt)
        print("disdet amt =", dis_det_array)
        print("compdis amt =", comp_dis_array)

        print("disdet =",dis_det_match)
        print("compdet =", comp_det_match)
        print("compdis =", comp_dis_match)
        """
        pos = 0
        orderpos = 0
        pointer_disdet = 0
        pointer_compdis = 0
        # 截至到这里全部匹配完，开始输入排序结果
        while (orderpos < len(order)):


            # 侦察型，不用判断直接输入
            if (order[orderpos, 0] < (self.num_uav / 3)):
                seq[pos] = order[orderpos]
                pos += 1
                orderpos += 1
            # 干扰型，顺序为侦察-干扰匹配，干扰-一体匹配
            if ((order[pos, 0] >= (self.num_uav / 3)) and (order[pos, 0] < (self.num_uav / 3 * 2))):
                if ((dis_det_match[pointer_disdet, 0] != 0)):
                    seq[pos] = dis_det_match[pointer_disdet]
                    pos += 1
                    pointer_disdet += 1
                    orderpos += 1

                if (orderpos >= (self.num_target + self.singletasktarget)):
                    return seq

                if ((dis_det_match[pointer_disdet, 0] == 0) and (comp_dis_match[pointer_compdis, 0] != 0)):
                    seq[pos] = comp_dis_match[pointer_compdis]
                    pos += 1
                    pointer_compdis += 1
                    orderpos += 1


            if (orderpos >= (self.num_target + self.singletasktarget)):
                return seq

            # 一体型匹配，顺序为一体-干扰，一体-侦察，一体×2
            if (order[orderpos, 0] >= (self.num_uav / 3 * 2)):
                seq[orderpos] = order[orderpos]
                orderpos += 1

                """
                print("seq =",seq)
                pos = dis_amt+detect_amt

                posdict = pos + comp_dis_array_2
                end = len(order)-1
                if(order[orderpos] in comp_dis_match_2):
                    seq[pos] = order[orderpos]
                    orderpos += 1
                    pos += 1
                if(order[orderpos] in comp_det_match):
                    seq[posdict] = order[orderpos]
                    orderpos += 1
                    posdict += 1

                if((order[orderpos] not in comp_dis_match_2) and (order[orderpos] not in comp_det_match)):
                    seq[end] = order[orderpos]
                    orderpos += 1
                    end -= 1
                """



        return seq

    # f1:时间成本-能耗综合函数
    # 输入：无人机工作序列，目标威胁半径，目标坐标
    # 过程：
    # 1，计算各个目标之间的坐标差值。
    # 2. 计算各个无人机的总路程，并除以无人机速度
    # 3. 分母为无人机最大工作时间。
    # 4. 由于侦察无人机速度快，所以侦察无人机被分配的任务越多，总时间消耗就越短
    def f1(self, order, radius, map,flag):
        #    print("map =",map)
        #    print("radius =",radius)
        # 先按照order排序，以便后续对单独一个无人机数值的静态计算，涉及时间窗口的f3不需要这么做。
        ord = sorted(order, key=operator.itemgetter(0))
        ord = np.array(ord)
        #    print(ord)

        seq = ord
        seq = np.array(seq)  # 临时数组，保证对seq进行操作不会影响ord内的值，万一有逻辑错误可以随时修改的版本

        # 区分每个无人机的目标号
        # for i in range(len(seq)):
        #    for j in range(self.num_uav):
        #        if(ord[i,0] == j):
        #            seq[i,0] = ord[i,0]
        #            seq[i,1] = ord[i,1]//2
        for i in range(len(seq)):
            seq[i, 1] = seq[i, 1] // 2
        # 这里调用提取子map的函数

        n = 0
        for i in range(self.num_uav):
            for j in range(len(seq)):
                if (seq[j, 0] == i):
                    n = n + 1
            seq = self.submap(seq, i, n)
            n = 0

        #    ord = seq
        seq = self.timewindow_ord(seq)
        ord = seq
        #    print("ord =",ord)
        if (flag == 0):
            return ord


        route = [0 for i in range(self.num_uav)]
        route = np.array(route)
        #    print(len(ord))
        for i in range(0, len(ord)):
            #    ind0 = int(ord[i-1,1]/2) #上一个任务的位置索引
            #    ind1 = int(ord[i,1]/2) #当前任务的位置索引
            ind0 = int(ord[i - 1, 1])  # 上一个任务的位置索引
            ind1 = int(ord[i, 1])  # 当前任务的位置索引

            if (i == 0):  # 0
                route[0] = route[0] + map[0, ind1] + 2 * math.pi * radius[ind1]
            if (((i > 0) and (i < len(ord) - 1)) and (ord[i, 0] == ord[i + 1, 0])):  # 不是0，但不换无人机
                route[ord[i, 0]] = route[ord[i, 0]] + \
                                   map[self.num_uav + ind0, ind1] + 2 * math.pi * radius[ind1]
            if (((i > 0) and (i < len(ord) - 1)) and (ord[i, 0] != ord[i + 1, 0])):  # 不是0，换无人机
                #    ind2 = int(ord[i + 1, 1] / 2)  # 下一个任务的位置索引
                ind2 = int(ord[i + 1, 1])
                route[ord[i, 0]] = route[ord[i, 0]] + map[self.num_uav + ind0, ind1] + \
                                   2 * math.pi * radius[ind1] + map[0, ind1]
                route[ord[i + 1, 0]] = route[ord[i + 1, 0]] + map[0, ind2]
            if (i == len(ord) - 1):  # 最后一个
                route[ord[i, 0]] = route[ord[i, 0]] + map[self.num_uav + ind0, ind1] + \
                                   2 * math.pi * radius[ind1] + map[0, ind1]
        #    print(route)
        # 到这里，已经计算出了每个无人机的总路程
        time = [0.00 for i in range(self.num_uav)]
        time = np.array(time)
        timecost = 0.00
        for i in range(0, self.num_uav):
            if (i < self.num_uav / 3):
                time[i] = route[i] / self.velocityA
                #    timecost = time[i]  / self.maxtime + timecost
                timecost = time[i] + timecost
            if ((i >= self.num_uav / 3) and (i < self.num_uav / 3 * 2)):
                time[i] = route[i] / self.velocityB
                #    timecost = time[i] / self.maxtime + timecost
                timecost = time[i] + timecost
            if (i >= self.num_uav / 3 * 2):
                time[i] = route[i] / self.velocityC
                #    timecost = time[i] / self.maxtime + timecost
                timecost = time[i] + timecost
        #    timecost = timecost / (self.num_target+self.singletasktarget)
        #    print(time)
        #    print("f1 =",timecost)
        #    print("time =",time)

        Ecost = self.f2(ord, radius, map)
        Ecost = np.array(Ecost)
        cost = [0.00 for i in range(self.num_uav)]
        cost = np.array(cost)
        c = 0
        #    print(cost)

        for i in range(len(cost)):
            cost[i] = 0.02 * Ecost[i] + time[i]
            c = c + cost[i]

        c = c / self.num_uav

        return c

    # f2:能耗函数
    # 输入：无人机工作序列，目标威胁半径
    # 过程：
    # 1. 计算每个无人机的总路程
    # 2. 根据分配到的任务类型计算每个无人机的能耗
    # 3. 分母为理论能耗上限
    def f2(self, order, radius, map):
        ord = sorted(order, key=operator.itemgetter(0))
        ord = np.array(ord)

        route = [0 for i in range(self.num_uav)]
        route = np.array(route)
        #    print(len(ord))
        for i in range(0, len(ord)):
            #    ind0 = int(ord[i - 1, 1] / 2)  # 上一个任务的位置索引
            #    ind1 = int(ord[i, 1] / 2)  # 当前任务的位置索引
            ind0 = int(ord[i - 1, 1])  # 上一个任务的位置索引
            ind1 = int(ord[i, 1])  # 当前任务的位置索引

            if (i == 0):  # 0
                route[0] = route[0] + map[0, ind1] + 2 * math.pi * radius[ind1]
            if (((i > 0) and (i < len(ord) - 1)) and (ord[i, 0] == ord[i + 1, 0])):  # 不是0，但不换无人机
                route[ord[i, 0]] = route[ord[i, 0]] + \
                                   map[self.num_uav + ind0, ind1] + 2 * math.pi * radius[ind1]
            if (((i > 0) and (i < len(ord) - 1)) and (ord[i, 0] != ord[i + 1, 0])):  # 不是0，换无人机
                #    ind2 = int(ord[i + 1, 1] / 2)  # 下一个任务的位置索引
                ind2 = int(ord[i + 1, 1])
                route[ord[i, 0]] = route[ord[i, 0]] + map[self.num_uav + ind0, ind1] + \
                                   2 * math.pi * radius[ind1] + map[0, ind1]
                route[ord[i + 1, 0]] = route[ord[i + 1, 0]] + map[0, ind2]
            if (i == len(ord) - 1):  # 最后一个
                route[ord[i, 0]] = route[ord[i, 0]] + map[self.num_uav + ind0, ind1] + \
                                   2 * math.pi * radius[ind1] + map[0, ind1]

        Ecost = [0.00 for i in range(self.num_uav)]
        Ecost = np.array(Ecost)
        cost = 0.00
        for i in range(0, len(ord)):
            ind1 = int(ord[i, 1] / 2)  # 当前任务的位置索引

            if (ord[i, 1] % 2 == 0):  # 侦察任务
                Ecost[ord[i, 0]] = 2 * math.pi * radius[ind1] * self.Dcost + Ecost[ord[i, 0]]
            if (ord[i, 1] % 2 == 1):  # 干扰任务
                Ecost[ord[i, 0]] = 2 * math.pi * radius[ind1] * self.Ccost + Ecost[ord[i, 0]]

        for i in range(0, self.num_uav):
            if (i < self.num_uav / 3):
                Ecost[i] = Ecost[i] + route[i] * self.Mcost
                #    Ecost[i] = Ecost[i] / self.resourcesA
                cost = cost + Ecost[i]
            if ((i >= self.num_uav / 3) and (i < self.num_uav / 3 * 2)):
                Ecost[i] = Ecost[i] + route[i] * self.Mcost
                #    Ecost[i] = Ecost[i] / self.resourcesB
                cost = cost + Ecost[i]
            if (i >= self.num_uav / 3 * 2):
                Ecost[i] = Ecost[i] + route[i] * self.Mcost
                #    Ecost[i] = Ecost[i] / self.resourcesC
                cost = cost + Ecost[i]

        #    cost = cost / self.num_uav

        return Ecost

    # 表示每个一体型无人机的平均任务数量
    def f3(self, order):
        num = 0
        for i in range(0, len(order)):
            if (order[i, 0] >= self.num_uav / 3 * 2):
                num = num + 1

        avg_num = num / (self.num_uav / 3)

        return avg_num

    # f4 根据无人机按照时间窗口处理任务的情况评价任务
    # 输入一个经过解码处理的顺序序列，威胁半径和任务区域信息，返回值为方案可靠性评价
    # 步骤：
    # 1. 创建数组，记录以下内容：
    # *每个任务的开始时间 *每个任务的结束时间 *uav重复性检测
    # 2. 任务按照队列来执行
    def f4(self, order, radius, map, timewindow):

        uavtime = [0.00 for i in range(self.num_uav)]  # 每个无人机执行当前工作的时间
        uavtime = np.array(uavtime)
        starttime = [0.00 for i in range(2 * self.num_target)]  # 任务开始时间集合
        starttime = np.array(starttime)
        endtime = [0.00 for i in range(2 * self.num_target)]  # 任务结束时间集合
        endtime = np.array(endtime)

        uavinuse = [0 for i in range(self.num_uav)]  # 记录当前无人机是否在队列中
        uavinuse = np.array(uavinuse)

        q = queue.Queue()

        # 计算无人机完成每个工作所需要的时间
        tasktime = [0.00 for i in range(2 * self.num_target)]  # 任务开始时间集合
        tasktime = np.array(tasktime)
        ord = sorted(order, key=operator.itemgetter(0))
        ord = np.array(ord)
        ord = self.timewindow_ord(ord)
        #    print("ord =",ord)
        # route = [0 for i in range(self.num_uav)]
        # route = np.array(route)
        for i in range(0, len(ord)):
            ind0 = int(ord[i - 1, 1] / 2)  # 上一个任务的位置索引
            ind1 = int(ord[i, 1] / 2)  # 当前任务的位置索引

            if (i == 0):  # 0
                route = map[0, ind1] + 2 * math.pi * radius[ind1]

                if (ord[i, 0] < self.num_uav / 3):
                    tasktime[ord[i, 1]] = route / self.velocityA
                if ((ord[i, 0] >= self.num_uav / 3) and (ord[i, 0] < self.num_uav / 3 * 2)):
                    tasktime[ord[i, 1]] = route / self.velocityB
                if (ord[i, 0] >= self.num_uav / 3 * 2):
                    tasktime[ord[i, 1]] = route / self.velocityC

            if (((i > 0) and (i < len(ord) - 1)) and (ord[i, 0] == ord[i + 1, 0])):  # 不是0，但不换无人机
                route = map[self.num_uav + ind0, ind1] + 2 * math.pi * radius[ind1]
                if (ord[i, 0] < self.num_uav / 3):
                    tasktime[ord[i, 1]] = route / self.velocityA
                if ((ord[i, 0] >= self.num_uav / 3) and (ord[i, 0] < self.num_uav / 3 * 2)):
                    tasktime[ord[i, 1]] = route / self.velocityB
                if (ord[i, 0] >= self.num_uav / 3 * 2):
                    tasktime[ord[i, 1]] = route / self.velocityC
            if (((i > 0) and (i < len(ord) - 1)) and (ord[i, 0] != ord[i + 1, 0])):  # 不是0，换无人机
                ind2 = int(ord[i + 1, 1] / 2)  # 下一个任务的位置索引
                route = map[self.num_uav + ind0, ind1] + 2 * math.pi * radius[ind1] + map[0, ind1]
                routenext = map[0, ind2]
                if (ord[i, 0] < self.num_uav / 3):
                    tasktime[ord[i, 1]] = route / self.velocityA
                if ((ord[i, 0] >= self.num_uav / 3) and (ord[i, 0] < self.num_uav / 3 * 2)):
                    tasktime[ord[i, 1]] = route / self.velocityB
                if (ord[i, 0] >= self.num_uav / 3 * 2):
                    tasktime[ord[i, 1]] = route / self.velocityC
                if (ord[i + 1, 0] < self.num_uav / 3):
                    tasktime[ord[i + 1, 1]] = routenext / self.velocityA
                if ((ord[i + 1, 0] >= self.num_uav / 3) and (ord[i, 0] < self.num_uav / 3 * 2)):
                    tasktime[ord[i + 1, 1]] = routenext / self.velocityB
                if (ord[i + 1, 0] >= self.num_uav / 3 * 2):
                    tasktime[ord[+1, 1]] = routenext / self.velocityC
            if (i == len(ord) - 1):  # 最后一个
                rotue = map[self.num_uav + ind0, ind1] + 2 * math.pi * radius[ind1] + map[0, ind1]
                if (ord[i, 0] < self.num_uav / 3):
                    tasktime[ord[i, 1]] = route / self.velocityA
                if ((ord[i, 0] >= self.num_uav / 3) and (ord[i, 0] < self.num_uav / 3 * 2)):
                    tasktime[ord[i, 1]] = route / self.velocityB
                if (ord[i, 0] >= self.num_uav / 3 * 2):
                    tasktime[ord[i, 1]] = route / self.velocityC

        Que = np.zeros((self.num_target * 2, 2))
        Que = np.array(Que)
        pointerstart = 0
        pointer = 0

        # 主循环，依次插入任务，目的是计算每个任务的开始和结束时间
        for i in range(len(order)):
            if (uavinuse[order[i, 0]] == 0):  # 当前无人机可以被插入队列中,先记录信息再入队
                uavinuse[order[i, 0]] = 1  # 当前无人机已有工作分配
                starttime[order[i, 1]] = uavtime[order[i, 0]]  # 记录被压入队列的任务的起始时间
                Que[pointer] = order[i]
                #    print(Que)
                pointer = pointer + 1
                #    pointer = pointer + 1
                #    print(order[i])
                #    q.put(order[i])
                #    print("q=",q)
                continue  # 避免和第3项引起冲突
            if (i == len(order) - 1):  # 任务结束时弹出所有无人机
                while (pointerstart < pointer):
                    #    task = q.get()
                    task = Que[pointerstart]
                    pointerstart = pointerstart + 1
                    task = np.asarray(task, dtype=int)
                    endtime[task[1]] = starttime[task[1]] + tasktime[task[1]]  # 计算该任务的结束时间
                    uavtime[task[0]] = uavtime[task[0]] + tasktime[task[1]]  # 计算无人机的时间
                    uavinuse[task[0]] = 0  # 把无人机变为已分配状态
            if ((i < len(order) - 1) and (uavinuse[order[i, 0]] == 1)):  # 当前无人机已被占用，需要弹出队列中所有元素
                while (pointerstart < pointer):
                    task = Que[pointerstart]
                    pointerstart = pointerstart + 1
                    #    print(task)
                    task = np.asarray(task, dtype=int)
                    endtime[task[1]] = starttime[task[1]] + tasktime[task[1]]  # 计算该任务的结束时间
                    uavtime[task[0]] = uavtime[task[0]] + tasktime[task[1]]  # 计算无人机的时间
                    uavinuse[task[0]] = 0  # 把无人机变为已分配状态
                    #    uavinuse[order[i, 0]] = 1  # 当前无人机已有工作分配
                    #    starttime[order[i, 1]] = uavtime[order[i, 0]]  # 记录被压入队列的任务的起始时间
                    #    Que[pointer] = order[i]
                    #    pointer = pointer + 1

                    continue

            # 是否符合时间窗口
        ex = self.ex
        tw = [0.0 for i in range(len(order))]
        success = 0
        for i in range(len(order)):
            j = int(i / 2)
            if (i % 2 == 0):
                if (endtime[i] <= timewindow[j, 0]):
                    tw[i] = 1
                    if j not in ex:
                        timewindow[j, 1] = endtime[i] + 5
                    success += 1
                else:
                    tw[i] = 0.1
            if (i % 2 == 1):
                if ((starttime[i] >= timewindow[j, 0]) and (endtime[i] <= timewindow[j, 1])):
                    tw[i] = 1
                    success += 1
                else:
                    tw[i] = 0.1

            # 评价任务：
        taskevaluate = [0.00 for i in range(len(order))]
        taskevaluate = np.array(taskevaluate)
        ev = 0
        for i in range(len(order)):
            if (endtime[i] - starttime[i] < 0):
                taskevaluate[i] = - (endtime[i] - starttime[i]) / tw[i]
            if (endtime[i] - starttime[i] > 0):
                taskevaluate[i] = (endtime[i] - starttime[i]) / tw[i]
            ev = ev + taskevaluate[i]

        ev = ev / len(order)
        max_time = np.max(endtime)
        success = success / (self.num_target * 2 - len(ex))
        #    print("ev = ", ev)
        return ev, max_time, success

    def setdata(self):
        datapoint = [0 for i in range(0, 10)]
        datapoint[0] = [0.33, 0.18]
        datapoint[9] = [0.30, 0.20]
        datapoint[1] = [0.325, 0.182]
        datapoint[2] = [0.32, 0.184]
        datapoint[3] = [0.317, 0.186]
        datapoint[4] = [0.313, 0.189]
        datapoint[5] = [0.31, 0.192]
        datapoint[6] = [0.307, 0.194]
        datapoint[7] = [0.304, 0.196]
        datapoint[8] = [0.302, 0.198]
        return datapoint

    """
    # 交叉
    # 输入：2进制群体与交叉率
    # 输出：交叉后的2进制种群
    def crossover(self, f, c):
    #    print("f0 =",f[0])
        for i in range(0, self.NP, 2):  # 遍历群体个体
            p = np.random.random()  # 生成一个0-1之间的随机数
            if p < c:
                q = np.random.randint(0, 2, (1, self.Len))  # 生成一个长度为Len的01数组
                for j in range(2*self.num_target*6):  #在前target*12位上交叉一次，也就是交换两个个体的任务优先级信息
                    if( q[:, j] == 1):
                        temp = np.int(f[i + 1, j])  # 下一个个体(i+1) 的第j个元素
                        f[i + 1, j] = f[i, j]
                        f[i, j] = temp
                for j in range(2*self.num_target*6,self.Len):  #在后target*8位上交叉一次，交换两个个体的无人机信息
                    if( q[:, j] == 1):
                        temp = np.int(f[i + 1, j])  # 下一个个体(i+1) 的第j个元素
                        f[i + 1, j] = f[i, j]
                        f[i, j] = temp
    #    print("f0 after =", f[0])
        return f

    """

    # 重写后的交叉函数
    # 输入：2进制群体与交叉率
    # 输出：交叉后的2进制种群
    # 这里是交叉2次，分别在前len_target * 2 * num_target范围内交叉1次，在后半再进行一次交叉
    def Crossover(self, f, c):
        # 生成2个随机数,分别为两段函数需要执行交叉操作的位数
        ran1 = random.randrange(0, self.num_target * 2 - 1) * self.len_target
        ran2 = random.randrange(0, self.num_target * 2 - 1) * self.len_uav
        half = self.len_target * self.num_target * 2
        for i in range(0, self.NP, 2):
            p = np.random.random()  # 生成一个0-1之间的随机数
            if (p < c):
                temp11 = f[i][:ran1]
                temp12 = f[i][ran1:half]
                temp13 = f[i][half:half + ran2]
                temp14 = f[i][half + ran2:]
                temp21 = f[i + 1][:ran1]
                temp22 = f[i + 1][ran1:half]
                temp23 = f[i + 1][half:half + ran2]
                temp24 = f[i + 1][half + ran2:]
                f[i] = np.concatenate([temp11, temp22, temp13, temp24], axis=0)
                f[i + 1] = np.concatenate([temp21, temp12, temp23, temp14], axis=0)
        return f

    # 重写后的交叉函数
    # 输入：2个2进制个体
    # 输出：交叉后的2进制个体（1个）
    # 这里是交叉2次，分别在前len_target * 2 * num_target范围内交叉1次，在后半再进行一次交叉
    def crossover(self, f1, f2):

        # 生成2个随机数,分别为两段函数的断点
        num = int(self.num_target / 5)  # 选择20%的长度进行交叉，可以调整
        ran1 = random.randrange(0, 2 * (self.num_target - num))
        ran2 = random.randrange(0, 2 * (self.num_target - num))
        half = self.len_target * self.num_target * 2  # 前后半分界线
        # 根据断点，找对应的位置,把染色体切成4-5段
        p1 = ran1 * self.len_target
        p11 = p1 + num * 2 * self.len_target
        p2 = ran2 * self.len_uav
        p22 = p2 + num * 2 * self.len_uav
        temp11 = f1[:p1]
        temp12 = f1[p1:p11]
        temp13 = f1[p11:half + p2]
        temp14 = f1[half + p2:half + p22]
        temp15 = f1[half + p22:]
        temp21 = f2[:p1]
        temp22 = f2[p1:p11]
        temp23 = f2[p11:half + p2]
        temp24 = f2[half + p2:half + p22]
        temp25 = f2[half + p22:]
        f1 = np.concatenate([temp11, temp22, temp13, temp24, temp15], axis=0)
        f2 = np.concatenate([temp21, temp12, temp23, temp14, temp25], axis=0)
        return f1, f2

    # 变异操作
    # 输入：2进制群体与变异率
    # 输出：交叉后的2进制种群
    def mutation(self, f, m):
        for i in range(np.int(np.round(self.NP * m))):  # 指定变异个数
            h = np.random.randint(0, self.NP, 1)[0]  # 随机选择一个（0-NP）之间的整数
            for j in range(int(np.round(self.Len * m))):  # 指定变异元素个数
                g = np.random.randint(0, self.Len, 1)[0]  # 随机选择一个(0-L）之间的整数
                #    for k in range(self.N):  # 遍历每一个变量
                f[h, g] = np.abs(1 - f[h, g])  # 将该元素取反
        #    print("f0 after =", f[0])
        return f

    def Mutation(self, f1, m):
        for i in range(int(self.Len)):  # 遍历整条染色体
            p = np.random.random()  # 生成一个0-1之间的随机数
            if (p < m):
                f1[i] = np.abs(1 - f1[i])  # 将该元素取反
        #    print("f0 after =", f[0])
        return f1

        # 快速非支配排序

    def fast_non_dominated_sort(self, values):
        values11 = values[0]  # 函数1解集
        S = [[] for i in range(0, len(values11))]  # 存放 每个个体支配解的集合。
        front = [[]]  # 存放群体的级别集合，一个级别对应一个[]
        n = [0 for i in range(0, len(values11))]  # 每个个体被支配解的个数 。即针对每个解，存放有多少好于这个解的个数
        rank = [np.inf for i in range(0, len(values11))]  # 存放每个个体的级别

        for p in range(0, len(values11)):  # 遍历每一个个体
            # ====得到各个个体 的被支配解个数 和支配解集合====
            S[p] = []  # 该个体支配解的集合 。即存放差于该解的解
            n[p] = 0  # 该个体被支配的解的个数初始化为0  即找到有多少好于该解的 解的个数
            for q in range(0, len(values11)):  # 遍历每一个个体
                less = 0  # 的目标函数值小于p个体的目标函数值数目
                equal = 0  # 的目标函数值等于p个体的目标函数值数目
                greater = 0  # 的目标函数值大于p个体的目标函数值数目
                for k in range(len(values)):  # 遍历每一个目标函数
                    if values[k][p] > values[k][q]:  # 目标函数k时，q个体值 小于p个体
                        less = less + 1  # q比p 好
                    if values[k][p] == values[k][q]:  # 目标函数k时，p个体值 等于于q个体
                        equal = equal + 1
                    if values[k][p] < values[k][q]:  # 目标函数k时，q个体值 大于p个体
                        greater = greater + 1  # q比p 差

                if (less + equal == len(values)) and (equal != len(values)):
                    n[p] = n[p] + 1  # q比p,  比p好的个体个数加1

                elif (greater + equal == len(values)) and (equal != len(values)):
                    S[p].append(q)  # q比p差，存放比p差的个体解序号

            # =====找出Pareto 最优解，即n[p]===0 的 个体p序号。=====
            if n[p] == 0:
                rank[p] = 0  # 序号为p的个体，等级为0即最优
                if p not in front[0]:
                    # 如果p不在第0层中
                    # 将其追加到第0层中
                    front[0].append(p)  # 存放Pareto 最优解序号

        # =======划分各层解========

        i = 0
        while (front[i] != []):  # 如果分层集合为不为空
            Q = []
            for p in front[i]:  # 遍历当前分层集合的各个个体p
                for q in S[p]:  # 遍历p 个体 的每个支配解q
                    n[q] = n[q] - 1  # 则将fk中所有给对应的个体np-1
                    if (n[q] == 0):
                        # 如果nq==0
                        rank[q] = i + 1
                        if q not in Q:
                            Q.append(q)  # 存放front=i+1 的个体序号

            i = i + 1  # front 等级+1
            front.append(Q)

        del front[len(front) - 1]  # 删除循环退出 时 i+1产生的[]

        return front

        # =============多目标优化：拥挤距离================

    def crowding_distance(self, values, front, popsize):
        #   print("front = ",front)
        distance = np.zeros(shape=(popsize,))  # 拥挤距离初始化为0
        for rank in front:  # 遍历每一层Pareto 解 rank为当前等级

            # for i in range(len(values)):  # 遍历每一层函数值（先遍历群体函数值1，再遍历群体函数值2...）,
            for i in range(len(rank)):  # 对该front里每个个体
                min_distance = 99999.0
                for j in range(len(rank)):
                    if (j == i):
                        continue
                    dis = self.dis_3d(values[0, rank[i]], values[0, rank[j]], values[1, rank[i]], values[1, rank[j]],
                                      values[2, rank[i]], values[2, rank[j]])
                    if ((dis < min_distance) and (dis > 0)):
                        max_distance = dis
                distance[rank[i]] = min_distance

        # 按照格式存放distances
        distanceA = [[] for i in range(len(front))]  #
        for j in range(len(front)):  # 遍历每一层Pareto 解 rank为当前等级
            for i in range(len(front[j])):  # 遍历给rank 等级中每个解的序号
                distanceA[j].append(distance[front[j][i]])
        #  print(distanceA)
        return distanceA
        # =============多目标优化：精英选择================

    def elitism(self, front, distance, solution):
        #   输入格式和内容没有问题
        #   print("front =",front)
        #   print("distance =",distance)
        #   print("solution =",solution)

        # 思路是直接用编号来筛选解

        X1index = []  # 存储群体编号
        pop_size = len(solution) / 2  # 保留的群体个数 即（父辈+子辈)//2
        # pop_size = self.NP

        for i in range(len(front)):  # 遍历各层,len(front)的值是pareto层数
            rank_distancei = zip(front[i], distance[i])  # 当前等级 与当前拥挤距离的集合
            sort_rank_distancei = sorted(rank_distancei, key=lambda x: (x[1], x[0]),
                                         reverse=True)  # 先按拥挤距离大小x1排序，再按序号大小x0排序,逆序
            sort_ranki = [j[0] for j in sort_rank_distancei]  # 排序后当前等级rank

            sort_distancei = [j[1] for j in sort_rank_distancei]  # 排序后当前等级对应的 拥挤距离i

            if (pop_size - len(X1index)) >= len(sort_ranki):  # 如果X1index还有空间可以存放当前等级i 全部解
                X1index.extend([A for A in sort_ranki])
            #        print("x1 index =",X1index)

            # print('已存放len(X1index)', len(X1index))
            # print('当前等级长度', len(sort_ranki))
            # print('需要存放的总长度,popsize)
            # num = pop_size-len(X1index)# X1index 还能存放的个数
            elif len(sort_ranki) > (pop_size - len(X1index)):  # 如果X1空间不可以存放当前等级i 全部解
                num = pop_size - len(X1index)
                num = int(num)

                X1index.extend([A for A in sort_ranki[0:num]])
                # 这里由于按拥挤距离排序，所以会优先保留同一rank里拥挤距离大，即拥挤度小的个体

            #    print("X1 index =",X1index)
            X1 = [solution[i] for i in X1index]
        # print(solution[0])
        return X1

    def HV(self, front, num):
        # 这里需要排序

        front = sorted(front, key=lambda x: x[0])
        front = np.array(front)

        hv = 0.0
        volume = 0.0
        for i in range(0, num):

            volume = (30 - front[i, 0]) * (18 - front[i, 1]) * (18 - front[i,2])
         #   if (i > 0):
         #       volume = (30 - front[i, 0]) * (front[i - 1, 1] - front[i, 1]) * (front[i-1,2] - front[i,2])

            hv = hv + volume
        hv = hv/num
        print("hv =",hv)

        return hv

    # 根据父代排名选择概率
    # 返回一个取值范围为0到1之间的列表
    def parent_choice(self, alpha, beta, NP):

        pi = np.zeros(NP)
        pi = np.array(pi)
        sum = 0.0

        for i in range(NP):
            pow = np.power((beta - alpha), ((i - 1) / (NP - 1)))
            pi[NP - i - 1] = 1 / NP * (alpha + pow)
            sum = sum + pi[NP - i - 1]

        for i in range(NP):
            pi[i] = pi[i] / sum

        return pi

    # 根据迭代代数变化Pm
    def Pm_by_gen(self, gen, max_gen, p):
        x = 3
        Pm = 2 * p * np.power((1 - gen / max_gen), x)
        return Pm

    # 种群预处理
    # 1. 一个一个生成种群中的个体，计算其适应度值，如果和已有的差距太小则重新生成一个
    # 2. 返回值：初始种群parenttwo
    def pre(self, radius, map):
        parenttwo = np.zeros((self.NP, self.Len))
        parenttwo = np.array(parenttwo)
        p10 = np.zeros(self.num_target + self.singletasktarget)
        p10 = np.array(p10)
        res = np.zeros((2, self.NP), dtype=float)
        for i in range(self.NP):
            p = np.random.randint(0, 2, self.Len)

            p10 = self.decode(p)

            # 计算新生成个体的适应度值
            x = self.f1(p10, radius, map,1)
            y = self.f3(p10)
            if (i == 0):
                res[0, 0] = x
                res[1, 0] = y
                parenttwo[i] = p
            if (i >= 1):
                for j in range(i):
                    # 个体过于接近了 排除掉
                    if (self.dis(x, res[0, j], y, res[1, j]) <= 0):
                        i = i - 1
                        continue
                res[0, i] = x
                res[1, i] = y
                parenttwo[i] = p

        return parenttwo

    def main(self):
        pos = self.getpos()  # 生成目标点初始位置和无人机初始位置
        map = self.getmap(pos)  # 计算距离
        #    print("pos=",pos)
        #    print("map=",map)
        #    testt = min(map[1,:])
        # testl = list.index(min(map[1,:]))
        #    print(testt)
        # print(testl)
        threat = self.threat_initialize(pos)
        parent = np.zeros((self.NP, self.num_target + self.singletasktarget, 2))
        parent = np.array(parent)
        parentchild10 = np.zeros((2 * self.NP, self.num_target + self.singletasktarget, 2))
        parentchild10 = np.array(parentchild10)
        #    parenttwo = np.random.randint(0, 2, (self.NP, self.Len))  # 随机获得二进制 初始种群f.shape (50,1, 20) .1表示有1个变量
        parenttwo = self.pre(threat, map)
        timewindow = np.zeros((self.num_target, 2))
        timewindow = np.array(timewindow)
        timewindow = self.settimewindow(timewindow)
        min_f1 = 100
        min_f2 = 100
        #    print(parenttwo)
        #    print(parenttwo[1])
        #    print(parenttwo)
        #    parentten = self.decode(parenttwo[0])
        # 这里parent就是解码后的结果
        #    for i in range(self.NP):
        #        parent[i] = self.decode(parenttwo[i])
        #    parent = np.asarray(parent,dtype=int)
        paretovalues1 = []
        paretovalues2 = []

        f1_values = [0.0 for i in range(self.max_gen)]
        f1_values = np.array(f1_values)
        f2_values = [0.0 for i in range(self.max_gen)]
        f2_values = np.array(f2_values)
        front1_values = [0.0 for i in range(self.max_gen)]
        front1_values = np.array(front1_values)
        front2_values = [0.0 for i in range(self.max_gen)]
        front2_values = np.array(front2_values)
        f1_min = [0.0 for i in range(self.max_gen)]
        f1_min = np.array(f1_min)
        f3_min = [0.0 for i in range(self.max_gen)]
        f3_min = np.array(f3_min)
        max_time = [0.0 for i in range(self.max_gen)]
        max_time = np.array(max_time)
        success = [0.0 for i in range(self.max_gen)]
        success = np.array(success)
        HV = [0.0 for i in range(self.max_gen)]
        HV = np.array(HV)
        pm_trigger = 0
        pc_trigger = 0
        last_front = []
        cnt = 0
        pm = self.Pm
        pc = self.Pc

        hv = 0.0
        gen = 0
        plt.ion()
        p_choice = np.zeros(self.NP)
        p_choice = np.array(p_choice)
        p_choice = self.parent_choice(0, 2, self.NP)
        p_pos = np.zeros(self.NP)
        p_pos = np.array(p_pos)
        for i in range(self.NP):
            p_pos[i] = i

        for j in tqdm(range(self.max_gen)):
            plt.clf()
            if (pm_trigger == 0):
                pm = self.Pm + (self.max_Pm - self.Pm) * (gen / self.max_gen)
            #    print("第 ",gen," 代交叉前个体 =",parenttwo[0])

            # 这里等于是错误的把parent2的值改变了，变成了x1，然后等于把2个一样的种群合一起了
            # 现在x1和parent2都指向了同一处内存空间
            # 使用拷贝，初步解决这个问题
            # 现在已经实现了父代个体不会丢失的功能

            pr2 = copy.copy(parenttwo)  # 交叉前的种群
            child = copy.copy(parenttwo)  # 交叉前的种群

            #    print("第 ", gen, " 代交叉前pr2 =", pr2[0])
            for i in range(0, self.NP, 2):
                if (pc_trigger == 0):
                    father1 = i  # 随机选择一个（0-NP）之间的整数
                    father2 = i + 1  # 随机选择一个（0-NP）之间的整数
                # 种群内个体集中已经达到一定数值
                if (pc_trigger == 1):
                    father1 = random.randint(0, self.NP - 1)
                    father2 = random.randint(0, self.NP - 1)
                    # 重新选择第2个个体来进行交叉变异过程
                    while (np.array_equal(pr2[father1], pr2[father2])):
                        father2 = random.randint(0, self.NP - 1)

                p1 = np.random.random()
                p2 = np.random.random()
                if (p1 < pc):
                    child[i], child[i + 1] = self.crossover(pr2[father1], pr2[father2])
                if (p2 > 1 - pm):
                    child[i] = self.Mutation(pr2[father1], 0.1)
                    child[i + 1] = self.Mutation(pr2[father2], 0.1)
            #    X1 = self.Crossover(parenttwo, self.Pc)  # 交叉操作 X1为交叉后群体
            #    print("第 ", gen, " 代x1操作后个体 =", X1[0])
            #    print("第 ", gen, "代x1后parent =", parenttwo[0])
            #    X2 = self.mutation(X1, self.Pm)  # 变异 变异后为子代群体
            # 检查一下交叉后个体是否丢失
            #    print("第 ",gen,"代交叉后个体 =", parenttwo[0])
            #    print("第 ", gen, " 代交叉后pr2 =", pr2[0])
            #    print("第 ", gen, "代交叉变异后个体 =", X2[0])
            parentchild2 = np.concatenate([child, parenttwo], axis=0)  # 合并父子代
            #    print((len(parentchild2)))
            #    print(parentchild2[99])
            #    parentchild2 = np.concatenate([parenttwo, X2], axis=0) #合并父子代
            #    print("第 ", gen, "代合并后个体0 =", parentchild2[0])
            #    print("第 ", gen, "代合并后子代个体0 =", parentchild2[0+self.NP])

            for i in range(len(parentchild2)):
                parentchild10[i] = self.decode(parentchild2[i])
            parentchild10 = np.asarray(parentchild10, dtype=int)
            #    print(parentchild10[99])

            values1 = np.zeros(shape=len(parentchild10), )
            for i in range(len(parentchild10)):  # 遍历每一个个体
                values1[i] = self.f1(parentchild10[i], threat, map,1)

            if (min(values1) < min_f1):
                min_f1 = min(values1)

            #    valuescost = np.zeros(shape=len(parentchild10), )
            #    for i in range(len(parentchild10)):  # 遍历每一个个体
            #         valuescost[i] = self.f2(parentchild10[i],threat,map)

            values2 = np.zeros(shape=len(parentchild10), )
            for i in range(len(parentchild10)):  # 遍历每一个个体
                # values2[i] = self.f2(parentchild10[i],threat,map)
                values2[i] = self.f3(parentchild10[i])

            values3 = np.zeros(shape=len(parentchild10), )
            for i in range(len(parentchild10)):  # 遍历每一个个体
                # values2[i] = self.f2(parentchild10[i],threat,map)
                values3[i], max_time[gen], success[gen] = self.f4(parentchild10[i], threat, map, timewindow)

            f1_values[gen] = np.mean(values1)
            f2_values[gen] = np.mean(values2)
            f1_min[gen] = np.min(values1)
            f3_min[gen] = np.min(values3)

            values = [values1, values2, values3]
            values = np.array(values)
            # 到这里，的确是算出了100个数值

            front = self.fast_non_dominated_sort(values)
            ans = front[0]  # 保存这代最佳个体
            front_0 = np.unique(front[0])

            if ((len(ans) / self.NP) >= 0.9357):
                pc_trigger = 1
            if ((len(ans) / self.NP) < 0.9357):
                pc_trigger = 0

            sol = []
            for i in range(len(front_0)):
                v1 = values[0, front_0[i]]
                v2 = values[1, front_0[i]]
                v3 = values[2, front_0[i]]
                sol.append([v1, v2,v3])

            front_0_sol = np.array(sol)
            front_0_val = np.unique(front_0_sol, axis=0)
            front_0_sorted = sorted(front_0_val, key=lambda x: (x[1], x[0]), reverse=True)
            front_0_sorted = np.array(front_0_sorted)
            if (gen == 0):
                last_front = front_0_sorted
                last_front = np.array(last_front)
            if (gen >= 1):
                if (np.array_equal(front_0_sorted, last_front) == True):
                    cnt += 1

                if (np.array_equal(front_0_sorted, last_front) == False):
                    cnt = 0
                    pm = self.Pm + (self.max_Pm - self.Pm) * (gen / self.max_gen)
                    last_front = front_0_sorted
                    last_front = np.array(last_front)
                    pm_trigger = 0
            if (cnt >= 5):
                pm = self.Pm + (self.max_Pm - self.Pm) * (gen / self.max_gen) + 0.1
                if (pm >= self.max_Pm):
                    pm = self.max_Pm
                pm_trigger = 1

           # HV[gen] = self.HV(sol, len(sol))

            distanceA = self.crowding_distance(values, front, 2 * self.NP)
            #    print("distanceA =",distanceA)
            # 目前的逻辑截止到这里都正确

            X3 = self.elitism(front, distanceA, parentchild2)

            parenttwo = np.array(X3)
            parenttwo = parenttwo.reshape(self.NP, self.Len)

            # 画图并动态更新
            resultf1 = [0.00 for i in range(0, len(front[0]))]
            resultf1 = np.array(resultf1)
            resultf2 = [0.00 for i in range(0, len(front[0]))]
            resultf2 = np.array(resultf2)
            for i in range(0, len(front[0])):
                resultf1[i] = values1[ans[i]]
                resultf2[i] = values3[ans[i]]
            val = [resultf1, resultf2]

            #    hv = self.HV(val, len(val[0]))
            #    print(gen + 1, " ", hv)
            #    print("gen #",gen," 's value =",val)
            """
            plt.xlim(16.5, 22.5)  # 20目标6uav适用
            #    plt.xlim(12,17.5)
        #    plt.ylim(-0.5, 10.5)  # 20目标6uav
            plt.ylim(0,5)  # 20目标6uav
            #    plt.ylim(-0.5,8.5)
            plt.scatter(resultf1, resultf2, s=20, marker='o')
            plt.pause(0.01)
            """

            gen = gen + 1

        # 循环结束
        # 对最后一代的2进制种群解码
        m_time = 0.0
        suc = 0.0

        for i in range(self.NP):
            parent[i] = self.decode(parenttwo[i])
        parent = np.asarray(parent, dtype=int)

        # 解码后计算适应值
        ordtotal = []
        for i in range(len(parent)):  # 遍历每一个个体
            values1[i] = self.f1(parent[i], threat, map, 1)
            if (i <= 10):

                ord = self.f1(parent[i], threat, map, 0)
                ord = np.array(ord)
                #    print("ord ",i," = ",ord)
                ordtemp = ordtotal
                ordtemp = np.array(ordtemp)
                if (i == 0):
                    ordtotal = ord
                if (i > 0):
                    ordtotal = np.concatenate((ordtemp, ord), axis=0)

        out = pd.DataFrame(ordtotal)
        writer = pd.ExcelWriter('improved NSGA-II dset2 solution 15.xlsx')
        out.to_excel(writer, sheet_name='example1', float_format='%.4f')
        writer._save()
        writer.close()

        values1 = np.zeros(shape=len(parent), )
        for i in range(len(parent)):  # 遍历每一个个体
            values1[i] = self.f1(parent[i], threat, map,1)

        values2 = np.zeros(shape=len(parent), )
        for i in range(len(parent)):  # 遍历每一个个体
            #    values2[i] = self.f2(parent[i], threat, map)
            values2[i] = self.f3(parent[i])

        values3 = np.zeros(shape=len(parent), )
        for i in range(len(parent)):  # 遍历每一个个体
            values3[i], m_time, suc = self.f4(parentchild10[i], threat, map, timewindow)

        values = [values1, values2, values3]
        #    print(values)
        front = self.fast_non_dominated_sort(values)
        ans = front[0]
        #    print(front)

        resultf1 = [0.00 for i in range(0, len(front[0]))]
        resultf1 = np.array(resultf1)
        resultf2 = [0.00 for i in range(0, len(front[0]))]
        resultf2 = np.array(resultf2)
        resultf3 = [0.00 for i in range(0, len(front[0]))]
        resultf3 = np.array(resultf3)
        for i in range(0, len(front[0])):
            resultf1[i] = values1[ans[i]]
            resultf2[i] = values2[ans[i]]
            resultf3[i] = values3[ans[i]]
        val = [resultf1, resultf2, resultf3]
        print(ans)

        Ans = []
        Ans.append(resultf1)
        Ans.append(resultf2)
        Ans.append(resultf3)
      #  Ans.append(HV)

        print("ans = ", Ans)
        output = pd.DataFrame(Ans)
        writer = pd.ExcelWriter('improved NSGA-II timewindow dset2 final 15.xlsx')
        output.to_excel(writer, sheet_name='example2', float_format='%.4f')
        writer._save()
        writer.close()

        #    igd = self.IGD(val, min_f1, min_f2)
        #    HV = self.HV(val, len(front[0]))

        #    print("IGD = ",igd)
        #    print("HV =",HV)
        plt.pause(0)
    #    plt.scatter(resultf1, resultf2, s=20, marker='o')
    #    plt.scatter(values1, values2, s=20, marker='o')
    #    for i in range(len(front[0])):
    #        plt.annotate(i, xy=(values1[i], values2[i]), xytext=(values1[i] - 0.05, values2[i] - 0.05), fontsize=7)
    #        plt.annotate(i, xy=(values1[i], values2[i]), s=20,marker='o')
    #    plt.xlabel('f1 - time_cost')
    #    plt.ylabel('f2 - avg_mission')
    #    plt.title('f1-f2 pareto等级1的所有解')


if __name__ == "__main__":
    ga1 = GaMultiobjective()
    ga1.main()