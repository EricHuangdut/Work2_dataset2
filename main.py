import math
import queue

import numpy as np
import matplotlib
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib
import copy
import operator
import requests
import openpyxl
from openpyxl import Workbook
import csv
import pandas as pd



class MOEAD:
    # 算法基本参数和全局变量
    def __init__(self):
        self.EP_x_id = [] #存储Pareto前沿中解的id
        self.EP_x_val = [[]]  #存储Pareto前沿中解的数值，问题中是2维的
        self.Pop = [] #存储种群中的每个个体，2进制
        self.Pop_val = [] #存储种群中每个个体的适应度值
        self.W = [] #存储均值向量
        self.W_Bi_T = [] #存储均值向量的邻居，数量为T个
        self.T = 5 #5个邻居
        self.NP = 500 #种群个体数
        self.max_gen = 500 #迭代次数
        self.Z = [0,0,0] #理想点，这里设定为0，0
        self.Pc = 0.9 #交叉率
        self.max_num_f = 2 #待优化的目标个数
        self.H = 31 #目标方向个数 = 20
        self.m = 3 #2维空间

        #场景参数
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
        self.max_mission = 14


        #涉及到目标函数计算的一些参数
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
        self.threat = []  # 存放目标的威胁程度
        self.ex = [2, 4, 7, 12, 16, 18, 26, 31, 34]

        gen = 0 #迭代代数
    """
    第1部分，这里使用NSGA-2算法中所有的模型代码
    模型部分最终可输出的东西：
    1.编码前的无人机分配方案,2进制数字串，其长度为（4+6）*num_mission
    2.通过编码，计算得到的f1，f2值
    注意这里要做一个对应
    """
    def dis(self, x1, x2, y1, y2):
        dist = np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))
        return dist

    def cal_distance(self, pos1):
        map = [[0 for i in range(0, self.num_target)] for i in range(0, self.num_uav+self.num_target)]
        map = np.array(map)
        for i in range(0, self.num_uav+self.num_target):
            for j in range(0, self.num_target):
                map[i, j] = self.dis(pos1[i, 0], pos1[j + self.num_uav, 0], pos1[i, 1], pos1[j + self.num_uav, 1])
        return map

    #和上面那个函数不一样的是，这个函数是对任意数据集的距离计算
    def CAL_distance(self, pos1):
        map = [[0 for i in range(0, len(pos1))] for i in range(0, len(pos1))]
        map = np.array(map)
        for i in range(0, len(pos1)):
            for j in range(0, len(pos1)):
                map[i, j] = self.dis(pos1[i, 0], pos1[j, 0], pos1[i, 1], pos1[j, 1])
        return map

    #初始数据集
    def getpos(self):
        #生成目标点的随机位置
        pos = [[0,0] for i in range(0, self.num_target + self.num_uav)]
                #这里的目的是去掉一些目标点距离出发位置过近的极端情况。
        pos = np.array(pos)
        #生成3种无人机的初始位置
        num = int(self.num_uav/3)  #3种无人机
        for i in range(0,num):
            pos[i,0] = 0
            pos[i,1] = self.ymax  #侦察无人机在（0，ymax）
        for i in range(num,2 * num):
            pos[i,0] = 0
            pos[i,1] = 0  #打击无人机在（0，0）
        for i in range(2 * num, 3 * num):
            pos[i,0] = self.xmax
            pos[i,1] = 0
        init = True
        if(init):
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

    #数据集的第二部分，初始威胁半径的生成
    def threat_initialize(self, pos):
        threat_radius = [0.00 for i in range(0, self.num_target)]
        #threat_radius = [10 + np.random.random() * 40 for i in range(0, self.num_target)]
        #一开始是用这个生成的 ↑  后面跑实验的时候都在使用同一个数据集
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

    def getmap(self, pos):
        self.map = self.cal_distance(pos)
        return self.map

    #这里调用的是第二个计算距离的函数，其他的都和第一个一样，仅作为更改数据集的情况下测试用
    def GETmap(self, pos):
        self.map = self.CAL_distance(pos)
        return self.map

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

    #任务序列重排序
    #解码函数
    def decode(self,f):
    #    print("f =",f)
        L1 = self.num_target * 6 * 2#前半部分长度
        L2 = self.num_target * 4 * 2#后半部分长度
        mission = [[0 for i in range(0, 2)] for i in range(0, 2*self.num_target)] #2维，形式是[任务index，优先级]
        order = []
        allo = [[0 for i in range(0, 2)] for i in range(0, 2*self.num_target)] #2维，形式是[任务index，无人机编号]
        allo = np.array(allo)
        mission = np.array(mission)
        order = np.array(order)
        pos = 0  # 指示此时mission的位置
        # 对任务进行排序得到一个序列
        count = 0
        m = 0

        f = np.array(f)
    #    print("f=", f)


        for j in range(0,L1):
        #    print("j=",j)
            if(count < 6):
                m = f[j] * np.power(2,count) + m#转化成10进制,这里m可以正常计算了
                count = count + 1
                if(count == 6):  # 此时需要计算下个任务
                    count = 0
                    mission[pos,0] = pos # 任务的实际编号
                    mission[pos,1] = m # 第pos+1个任务的优先级是m
                #    print(mission[pos])
                    m = 0
                    pos = pos + 1
     #   print(mission)
        taskord = self.reorder(mission)   #需要写一个reorder函数对mission排序
        #task的输出结果为一个1维的量，每个位置上的值代表任务序列

        # 排除掉不需要干扰的目标
        # 这里这个数据集要改的
        ex = self.ex #属于目标2，4，7，12，16，18 // 1,3,6,11,15,17
    #    ex = [3, 7, 13, 21, 27]  # 属于目标2，4，7，11，14 // 1,3,6,10,13
        task = [0 for i in range(0,2*self.num_target-len(ex))]
        task = np.array(task)
        ind = 0
        conf = 0
        for i in range(0,len(taskord)):
            for j in range(0,len(ex)):
                if(taskord[i] != ex[j]): #全检索
                    conf = conf
                if(taskord[i] == ex[j]):
                    conf = 1
            if(conf == 0):
                task[ind] = taskord[i]
                ind = ind + 1
            conf = 0

    #    print("task =",task)
        pos = 0
        count = 0
        n = 0
        cnt = 0
        # 为无人机分配任务,这里先得到一个初始仅约束无人机任务种类的序列，这里逻辑和已经排好序的task相关
        for i in range(L1,L1+L2-len(ex)*4):
            if(cnt < 4):
                n = f[i] * np.power(2,cnt) + n #转化成10进制
                cnt = cnt + 1
                if(cnt == 4):
                    cnt = 0
                    # %2=0意味着这是个侦察任务
                    # 对9个无人机，选能执行的6个，即编号1-3和编号7-9，数组里存储的数值是-1的
                    if((task[count]%2) == 0):
                        num_u = n % (self.num_uav/3*2)
                        if(num_u >=self.num_uav/3): # 选中了一体型无人机
                            num_u = num_u + self.num_uav/3 #对应一体型无人机的编号，+3的3 = self.num_uav/3
                        allo[pos,0] = num_u
                        allo[pos,1] = task[count]
                    #    print("allo =",allo)
                        pos = pos + 1
                        n = 0


                    # %2=1代表干扰任务
                    # 编号4-9
                    # num_u在这里=3的时候会跳过一组i值导致out of index
                    if((task[count]%2) == 1):
                        num_u = n % (self.num_uav/3*2) + self.num_uav/3 #取值范围是3-8
                        allo[pos, 0] = num_u
                        allo[pos, 1] = task[count]
                    #    print("allo =", allo)

                        pos = pos + 1

                        n = 0
                    count = count + 1
        Allo = [[0 for i in range(0, 2)] for i in range(0, 2*self.num_target-len(ex))]
        Allo = np.array(Allo)
        for i in range(0, len(Allo)):
             Allo[i,0] = allo[i,0]
             Allo[i,1] = allo[i,1]


        #感觉这步可能有点多余，因为要去除不执行的任务，保留一部分
        allo = self.reallo(Allo,len(ex)) #需要写一个reallo函数对allo调整顺序，这里涉及到无人机执行任务数量约束
        #allo的输出结果为一个2维变量，格式为[任务编号，执行的无人机编号]

        Finalorder = [[0 for i in range(0, 2)] for i in range(0, 2 * self.num_target - len(ex))]
        Finalorder = np.array(Finalorder)
        for i in range(0, len(task)):
            for j in range(0, len(Finalorder)):
                if(allo[j,1] == task[i]):
                    Finalorder[i,0] = allo[j,0]
                    Finalorder[i,1] = allo[j,1]


    #    print("Order = ",Finalorder)
        return Finalorder

    def reorder(self,task):
    #    print(task)
    #    temp = [[]] #2维，存放排序后的task
        ord = [0 for i in range(2*self.num_target)] #1维 把排序后的task只提取优先级返回
    #    ord = np.array(ord)
        temp = sorted(task,key=operator.itemgetter(1))  #按照key=2也就是优先级来排序
        temp = np.array(temp)
    #    print(temp)

        for i in range(0,(2* self.num_target)):
            ord[i] = temp[i,0] #根据key=1，也就是任务序号，排出ord

    #    print("ordinary ord=",ord)

    # 排序后，对具体的任务进行重排，确保对同一个任务，其侦察任务在干扰任务之前完成
        index = 0 #标记当前任务
        id = 0 #侦察任务下标
        ic = 0 #干扰任务下标
        for i in range(0, self.num_target):
            for j in range(0, 2 * self.num_target):
                if(ord[j] == index):
                    id = j
                if(ord[j] == index+1):
                    ic = j
            if(id>ic):
                t = ord[id]
                ord[id] = ord[ic]
                ord[ic] = t
            index = index + 2

    #    print("ord= ", ord)

        return ord

    def targetseq(self, order):
        for i in range(len(order)):
            order[i, 1] = order[i, 1] // 2  # 向下取整即可得到目标点的定位，下标从0开始，比如任务13对应目标6，任务1对应目标0

        return order

    def submap(self,seq,uavnum,n):
        sub = [0 for i in range(0,n+1)] #存储无人机所要完成的目标序号，这里是map上的序号，转化为任务序号需要-self.num_uav
        sub = np.array(sub)
        p = self.getpos() #获取初始数据集
        pointer = 1
        sub[0] = uavnum #从0开始的无人机序号下标
        #提取1个无人机的工作序列并加入pos
        for i in range(len(seq)):
            if(seq[i,0] == uavnum): #序号就是要排的这个无人机的序号
                sub[pointer] = seq[i,1] + self.num_uav #这里加num_uav是为了检索任务点对应位置的下标
                pointer = pointer + 1

        pos = [[0 for i in range(0,2)]for i in range(len(sub))]
        pos = np.array(pos)

        # 对应数据集中的位置点，输入sub数组中
        for i in range(len(sub)):
            pos[i,0] = p[sub[i],0]
            pos[i,1] = p[sub[i],1]

        m = self.GETmap(pos)  #GETmap实现正确，已经验证
    #    print("m=",m)
        ans = self.subseq(m,sub,n)
        ans = np.array(ans)

        index = 0
        for i in range(len(seq)):
            if (seq[i, 0] == uavnum):
                seq[i,1] = ans[index]
                index = index + 1

    #    print("seq =",seq)

        return seq

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

        #    print("s=",seq)
        return seq

    #目标函数部分
    #F1：
    def f1(self,order,radius,map):
        #先按照order排序，以便后续对单独一个无人机数值的静态计算，涉及时间窗口的f3不需要这么做。
        #这里的问题是，在第2次及以后调用f1的时候，传进来的参数出现了变化？
    #    print("------function F1 is used------")
    #    print("ord = ",order)
        # 这里传进来是1d的 在val里面应该先整理一下

        ord = sorted(order,key=operator.itemgetter(0))
        ord = np.array(ord)

    #    print("map =",map)
    #    print(ord)
        #这里radius传进来是空？
    #    print("radius =",radius)

        seq = ord
        seq = np.array(seq) #临时数组，保证对seq进行操作不会影响ord内的值，万一有逻辑错误可以随时修改的版本

        # 区分每个无人机的目标号
        #for i in range(len(seq)):
        #    for j in range(self.num_uav):
        #        if(ord[i,0] == j):
        #            seq[i,0] = ord[i,0]
        #            seq[i,1] = ord[i,1]//2
        for i in range(len(seq)):
            seq[i, 1] = seq[i, 1] // 2
        # 这里调用提取子map的函数

    #    print("seq0 =",seq)
        n = 0
        for i in range(self.num_uav):
            for j in range(len(seq)):
                if(seq[j,0] == i):
                    n = n + 1
            seq = self.submap(seq,i,n)
            n = 0
    #    print("seq1 =", seq)

        ord = seq

        route = [0 for i in range(self.num_uav)]
        route = np.array(route)
    #    print(len(ord))
        for i in range(0,len(ord)):
            ind0 = int(ord[i-1,1]/2) #上一个任务的位置索引
            ind1 = int(ord[i,1]/2) #当前任务的位置索引
        #    ind0 = int(ord[i - 1, 1])  # 上一个任务的位置索引
        #    ind1 = int(ord[i, 1])  # 当前任务的位置索引
        #    print("ind1 = ",ind1)
            if(i == 0): #0
                route[0] =route[0] + map[0,ind1] + 2 * math.pi * radius[ind1]
            if(((i > 0) and (i < len(ord)-1)) and (ord[i,0] == ord[i+1,0])): #不是0，但不换无人机
                route[ord[i,0]] = route[ord[i,0]] + \
                                  map[self.num_uav+ind0,ind1] + 2 * math.pi * radius[ind1]
            if (((i > 0) and (i < len(ord) - 1)) and (ord[i, 0] != ord[i + 1, 0])): #不是0，换无人机
            #    ind2 = int(ord[i + 1, 1] / 2)  # 下一个任务的位置索引
                ind2 = int(ord[i + 1, 1])
                route[ord[i,0]] = route[ord[i,0]] + map[self.num_uav+ind0,ind1] +\
                                  2 * math.pi * radius[ind1] + map[0,ind1]
                route[ord[i+1,0]] = route[ord[i+1,0]] + map[0,ind2]
            if(i == len(ord)-1): #最后一个
                route[ord[i,0]] = route[ord[i,0]] + map[self.num_uav+ind0,ind1] +\
                                  2 * math.pi * radius[ind1] + map[0,ind1]
    #    print(route)
        # 到这里，已经计算出了每个无人机的总路程
        time = [0.00 for i in range(self.num_uav)]
        time = np.array(time)
        timecost = 0.00
        for i in range(0,self.num_uav):
            if(i<self.num_uav/3):
                time[i] = route[i] / self.velocityA
            #    timecost = time[i]  / self.maxtime + timecost
                timecost = time[i]  + timecost
            if ((i >= self.num_uav / 3) and (i < self.num_uav / 3 * 2)):
                time[i] = route[i] / self.velocityB
            #    timecost = time[i] / self.maxtime + timecost
                timecost = time[i]  + timecost
            if (i >= self.num_uav / 3 * 2):
                time[i] = route[i] / self.velocityC
            #    timecost = time[i] / self.maxtime + timecost
                timecost = time[i]  + timecost
    #    timecost = timecost / (self.num_target+self.singletasktarget)
    #    print(time)
    #    print("f1 =",timecost)
    #    print("time =",time)

        Ecost = self.f2(ord,radius,map)
        Ecost = np.array(Ecost)
        cost = [0.00 for i in range(self.num_uav)]
        cost = np.array(cost)
        c = 0
    #    print(cost)

        for i in range(len(cost)):
            cost[i] = 0.02*Ecost[i] + time[i]
            c = c + cost[i]

        c = c / self.num_uav

        return c

    #F1的能耗部分
    def f2(self,order,radius,map):
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
        for i in range(0,len(ord)):
            ind1 = int(ord[i, 1] / 2)  # 当前任务的位置索引

            if(ord[i,1]%2 == 0): #侦察任务
                Ecost[ord[i,0]] = 2 * math.pi * radius[ind1] * self.Dcost + Ecost[ord[i,0]]
            if (ord[i,1]%2 == 1):  # 干扰任务
                Ecost[ord[i,0]] = 2 * math.pi * radius[ind1] * self.Ccost + Ecost[ord[i,0]]

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

    def f3(self,order):
        num = 0
        for i in range(0,len(order)):
            if(order[i,0] >= self.num_uav/3*2):
                num = num + 1

        avg_num = num / (self.num_uav/3)

        return avg_num

    def settimewindow(self, timewindow):
        initT = True

        if (initT):

            for i in range(self.num_target):
                timewindow[i, 0] = 5
                timewindow[i, 1] = 9999
                if i not in self.ex:
                    timewindow[i, 1] = 10

        return timewindow

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
        return ev

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



    """
    第2部分，遗传算法相关操作，交叉 & 变异
    此部分要写5-6个函数
    function 1:crossover,实现2个个体之间的交叉功能，交换2个2进制个体的信息
    function 2:mutation,实现1个个体的变异
    function 3:cross_mutation,输入2个个体，按照Pc决定其是交叉还是变异，这里输入的个体注意拷贝问题
    function 4:generate,进化产生下一代个体 *这部分最重点的函数之一
    function 5:e0函数，这里看看怎么优化，需要先搞定function 4
    function 6:evo,进化函数,这个部分最关键也是最核心的函数，涉及到MOEAD算法主循环
    """

    # function 1
    # 输入：个体p1,个体p2，形式为2进制
    # 输出：交叉后的个体p1，p2
    def crossover(self,f1,f2):

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

        return f1,f2

    # function 2
    # 输入：个体p1
    # 输出：变异后的个体p1
    def mutation(self,f1):
        m = 0.3
        for i in range(int(self.Len)):  # 遍历整条染色体
            p = np.random.random()  # 生成一个0-1之间的随机数
            if (p < m):
                f1[i] = np.abs(1 - f1[i])  # 将该元素取反
        return f1

    # function 3
    # 输入：个体p1，p2
    # 输出：根据Pc，决定这俩个体是交叉还是变异，返回操作后的p1，p2
    # 这里操作之后的结果不变？
    def cross_mutation(self,p1,p2):
        f1 = np.copy(p1)
        f2 = np.copy(p2)

        p = np.random.random()
        if (p < self.Pc):
            f1, f2 = self.crossover(f1, f2)

        else:
            f1 = self.mutation(f1)
            f2 = self.mutation(f2)

        return f1,f2

    # function 4
    # 输入：需要被进化的个体序号pi，2进制序列形式的个体p0，2进制序列形式，个体p0的邻居p1，p2
    # 注意这里p0就是编号为pi个体的2进制形式
    # 输出：进化后的，2进制序列形式的个体y
    # 简单来说 现在的问题是 可以交叉变异，但是交叉变异之后留下的个体都是更差的
    def generated_next(self,pi,p0,p1,p2,threat,map,timewindow):
        #print("threat=",threat)
        #print("map=",map)

    #    pi = np.array(pi)
    #    p0 = np.array(p0)
    #    p1 = np.array(p1)
    #    p2 = np.array(p2)
    #    print("pi =", pi)
    #    print("p0 =", p0)
    #    print("p1 =", p1)
    #    print("p2 =", p2)
        #第1步：对3个2进制个体，分别调用计算Tchbycheff距离的函数，计算其切比雪夫距离
        #这里不用输入值，因为解码和函数值的计算在切比雪夫计算式里面有
        qbxf_p0 = self.Tchbycheff(p0, pi, threat, map,timewindow)
        qbxf_p1 = self.Tchbycheff(p1, pi, threat, map,timewindow)
        qbxf_p2 = self.Tchbycheff(p2, pi, threat, map,timewindow)
    #    print("Tch0 =",qbxf_p0)
    #    print("Tch1 =", qbxf_p1)
    #    print("Tch2 =", qbxf_p2)
        #第2步：将3个计算结果合成为一个array，并用一个变量best标记其最小值，选中切比雪夫距离最小的个体作为y1，y1是2进制个体序列
        qbxf = np.array([qbxf_p0,qbxf_p1,qbxf_p2])
        best = np.argmin(qbxf) #argmin是检索最小值下标的
    #    print("best =",best)
        #第3步：把p0，p1，p2都copy成独立的一份，然后对copy后的3个个体调用cross_mutation函数
        #这里交叉的时候，先把newp0和newp1进行一次操作，然后用操作后的newp1和操作前的newp2再操作一次
        #这一步结束，得到新的newp0，newp1，newp2

        np0,np1,np2 = np.copy(p0),np.copy(p1),np.copy(p2)
        y1 = [p0,p1,p2][best]

        np0,np1=self.cross_mutation(np0,np1)
        np1,np2 = self.cross_mutation(np1,np2)
    #    print("y1=",y1)
        #到这里都对
        #第4步：对newp0~p2，重复步骤1和2，这次将6个个体中切比雪夫距离最小的个体作为y2，y2也是2进制个体序列

        qbxf_np0 = self.Tchbycheff(np0, pi, threat, map,timewindow)
        qbxf_np1 = self.Tchbycheff(np1, pi, threat, map,timewindow)
        qbxf_np2 = self.Tchbycheff(np2, pi, threat, map,timewindow)
        qbxf = np.array([qbxf_p0, qbxf_p1, qbxf_p2, qbxf_np0, qbxf_np1, qbxf_np2])
    #    print("qbxf 0=", qbxf_np0,qbxf_np1,qbxf_np2)
    #    print("qbxf =",qbxf)
        best = np.argmin(qbxf)  # argmin是检索最小值下标的
        y2 = [p0, p1, p2, np0, np1, np2][best]
        #第5步，对待优化的目标，随机选择其中的1个，例如待优化的目标函数为2个，则随机生成一个0或1的整数r，即随机选择一个目标更小的返回
        #默认情况下，这里返回的是y2，因为是6个里面切比雪夫距离最小的
        #但是如果y1在函数r上的表现好于y2，说明y1y2之间互不支配，只是y2切比雪夫距离小，这里允许其以50%的概率击败y2，因此追加第6步判断
        fm = np.random.randint(0,self.max_num_f)
    #    print("y1=", y1)
    #    print("y2=", y2)
        # 第6步，如果在第r个函数上，y1真的比y2表现更好，允许其以50%的概率保留下来。
        if np.random.rand()<0.5:
            dy1 = self.decode(y1)
            dy2 = self.decode(y2)
        #    print("dy1 =",dy1)
            fxy1 = self.f1(dy1,threat,map)
            fyy1 = self.f3(dy1)
            fzy1 = self.f4(dy1,threat,map,timewindow)
            fy1 = [fxy1,fyy1,fzy1]
            fxy2 = self.f1(dy2,threat,map)
            fyy2 = self.f3(dy2)
            fzy2 = self.f4(dy2,threat,map,timewindow)
            fy2 = [fxy2,fyy2,fzy2]
        #    print("fy1,fy2 =",fy1,fy2)
            if(fy2[fm] < fy1[fm]):
                return y2
            else:
                return y1
        return y2

    # function 5
    # 遗传算法主循环，需要进化max_gen轮
    # 输入：null （这里存疑，很可能需要根据模型的实际情况做更改）
    # 输出：Pareto非劣解解集中每个个体的ID集合Front_ID
    # threat map可以正确传进来
    # Pop传进来的是整个种群
    def evolution(self,threat,map,Pop,timewindow):
    #    print("threat =",threat)
    #    print("map =",map)
        #第1步 for循环，进化max_gen轮结束
    #    print(Pop)
        f = [0.0 for i in range(len(Pop))]
        f = np.array(f)
        f1 = [0.0 for i in range(self.max_gen)]
        f1 = np.array(f1)
        HyperV = [0.0 for i in range(self.max_gen)]
        HyperV = np.array(HyperV)
        self.W_Bi_T = np.array(self.W_Bi_T)
    #    print("wbit =",self.W_Bi_T)

        for gen in range(self.max_gen):
            print("gen =",gen)


        #第2步 在每次循环中，遍历种群中的每个个体，以序号pi作为迭代下标
        #问题1 这里的Pop传进来是个空集？
            for pi in enumerate(Pop):
        #        print("pi =", pi)
            #enumerate是个集合，输出的是个2维形式，第1维是序号，第2维才是个体2进制
        #        print("pi =",pi)
        #第3步 获取每个个体pi的邻居集合Bi，邻居集合的大小T为一个常量，已知
                Bi = self.W_Bi_T[pi[0]]
        #        print("Bi =",Bi)
        #第4步 随机选取2个范围为0至T的数，也就是从pi的邻居中，随机选2个，这里原算法没有去重，也就是一定概率会选择重复的
        #被遍历的个体是种群中的第pi个，这里通过Bi获取被选中的2个个体的下标k和l，然后从种群中找到对应的个体p1和p2
                a = np.random.randint(self.T)
                b = np.random.randint(self.T)
                c = np.random.randint(self.T)
                #去重，确保不会选到一个邻居
                if(a == b):
                    b = (b+c)%self.T
        #        print("a,b=",a,b)
        #第5步 对pi，p1，p2调用function 4，获得pi进化后的新个体y，当然这里有一定概率y还=pi，也就是这轮进化不出东西
                ai = Bi[a]
                bi = Bi[b]
        #        print("ai,bi=", ai, bi)
                Xp = Pop[pi[0]]
            #    print("Xp =", Xp)
                Xa = Pop[ai]
        #        print("Xa =", Xa)
                Xb = Pop[bi]
        #        print("Xb =", Xb)
        #直到这里的逻辑正确
        #第6步 分别对编号为pi的原个体，和进化操作后得到的y个体计算其切比雪夫距离
                #730行左右 输入：需要被进化的个体序号pi，2进制序列形式的个体p0，2进制序列形式，个体p0的邻居p1，p2
                Y = self.generated_next(pi[0],Xp,Xa,Xb,threat,map,timewindow)
        #        print("Y=",Y)
                qbxf_p = self.Tchbycheff(Xp,pi[0],threat,map,timewindow)
        #        print("Tch p=", qbxf_p)
                qbxf_y = self.Tchbycheff(Y,pi[0],threat,map,timewindow)
        #        print("Tch y=",qbxf_y)
        #第7步 如果y的切比雪夫距离比pi的小，也就是进化出了更好个体的情况下，走这一步，并进行一系列更新操作
                if(qbxf_y < qbxf_p):
                    # （实际上根据function 4，这里y至少也是≤pi的，但是等于的情况就跳过步骤7）
                    # 7.1 计算个体y在所有目标函数上的值
                    dy = self.decode(Y)
                    fxy = self.f1(dy, threat, map)
                    fyy = self.f3(dy)
                    fzy = self.f4(dy,threat,map,timewindow)
                    fy = [fxy, fyy, fzy]
                #    print("pi2 =",pi)
                    # 7.2 将这个值更新到EP_By_id中，EP是一个外部的全局变量，EP_By_id记录的是支配前沿的函数值
                    self.update_EP_by_ID(pi[0],fy)
                    # 7.3 随后，根据Y，更新理想点Z的值
                    # 7.4 更新EP_by_y，也就是Pareto Front中的个体。
                    self.update_EP_by_y(pi[0])
            xval = np.array(self.EP_x_val)
            #for i in range(len(xval)):
            #    front1[gen] = front1[gen] + xval[i,0]
            #front1[gen] = front1[gen] / len(xval)
            #HyperV[gen] = self.HV(self.EP_x_val,len(self.EP_x_val))

        #第8步 不管y的切比雪夫距离是不是小于pi的，都需要更新邻居
        #print("HV =",HyperV)



        return self.EP_x_id,f1

    """
    实际写函数的时候这个部分其实是需要先写的
    第3部分，MOEA/D算法相关
    """

    # function 7 & 8 & 9
    # 调用的时候注意格式
    # 生成均值向量,例如对一个2维的问题，生成20等分的向量，那么输出的时候就会输出一个长度为20×2的数组
    # 其中数组中每个元素长度为2，包含的是向量的x和y值
    # 例如[[1.00,0.00],[0.95,0.05],...,[0.00,1.00]]
    def perm(self, sequence):
        # ！！！ 序列全排列，且无重复
        l = sequence
        if (len(l) <= 1):
            return [l]
        r = []
        for i in range(len(l)):
            if i != 0 and sequence[i - 1] == sequence[i]:
                continue
            else:
                s = l[:i] + l[i + 1:]
                p = self.perm(s)
                for x in p:
                    r.append(l[i:i + 1] + x)
        return r

    def get_mean_vectors(self):
        H = self.H #这里H直接定义在主类里就行，目前使用的值为20
        m = self.m #m定义在主类里，目前使用的值为2
        sequence = []
        for ii in range(H):
            sequence.append(0)
        for jj in range(m - 1):
            sequence.append(1)
        ws = []

        pe_seq = self.perm(sequence)
        for sq in pe_seq:
            s = -1
            weight = []
            for i in range(len(sq)):
                if sq[i] == 1:
                    w = i - s
                    w = (w - 1) / H
                    s = i
                    weight.append(w)
            nw = H + m - 1 - s
            nw = (nw - 1) / H
            weight.append(nw)
            if weight not in ws:
                ws.append(weight)

        return ws

    #F9，这里直接调用F7和F8，生成一个均值向量序列。
    def generate(self):
        m_v = self.get_mean_vectors()
        return m_v
    #    print(m_v)

    #function 10
    #调用F9，得到一个序列，返回的是均值向量数组W，为权重
    def Load(self):
        mv = self.generate()
        self.W = mv
    #    Pop_size = len(mv) #这里，种群个体的规模居然是取决于被分段的数量？？
        # 这里问题解决不了的话可能开始要写死了，注意一下
        return mv

    #function 11
    #初始化Z集合
    #其实鉴于这里已经出现过f2=0的情况，所以先把Z写成0，0看一下。
 #   def cpt_Z(self):

    #function 12
    #计算初始化种群的Pareto前沿
    #输入：null（后面有逻辑问题的话需要更新一下这个地方）
    #输出：null
    def init_EP(self):
        #第1步 用一层for循环遍历种群中的每个个体pi，记录变量cnt = 0
        for pi in range(self.NP):
            cnt = 0
            f_val = self.Pop_val[pi]

        #第2步 第二层循环再遍历每个个体ppi，目的是比较
            for ppi in range(self.NP):
                f_ppval = self.Pop_val[ppi]
        #第3步，如果ppi支配pi，则cnt++，内层循环结束后，如果cnt=0表示没有解支配pi
                if(pi != ppi):
                    if(self.is_dominate_by_val(f_ppval,f_val)):
                        cnt += 1
        #第4步，如果没有解支配pi，把pi的编号加入EP_x_id，把值加入EP_x_fv
            if(cnt == 0):
                self.EP_x_id.append(pi)
                self.EP_x_val.append(f_val)
        return

    #function 13*
    #计算每个权重的t个邻居，权重由F9求得并输出
    #其实这里邻居确定后就不变了，但是初始化要弄好
    def cpt_W_Bi_T(self):
        #第1步 遍历W中每个个体
        for bi in range(len(self.W)):
        #第2步 从W中计算距离，选择最近的5个加入集合
            Bi = self.W[bi]
            Bi = np.array(Bi)
            self.W = np.array(self.W)
            dis = np.sum((self.W - Bi) ** 2, axis=1)
            bt = np.argsort(dis)
            bt = bt[1:self.T + 1]
            near = 0
           # print("lenBT", len(bt))
            for i in range(len(bt)):
                if(bt[i] < self.NP):
                    near = bt[i]
            for j in range(len(bt)):
                if(bt[j] >= self.NP):
                    bt[j] = near
            print("bi= ",bi,bt)
            self.W_Bi_T.append(bt)
        #    print("WbiT = ",self.W_Bi_T)
        return

    #function 14
    #输入：2进制个体x，2进制个体y，注意输入是有先后顺序的
    #输出：T or F，x是否支配y
    def is_dominate(self,x,y,threat):
        #第1步 解码x，y，计算适应值
        dx = self.decode(x)
        fxx = self.f1(dx, threat, map)
        fyx = self.f3(dx)
        fx = [fxx, fyx]
        dy = self.decode(y)
        fxy = self.f1(dy, threat, map)
        fyy = self.f3(dy)
        fy = [fxy, fyy]
        #第2步 判断x是否支配y
        i = 0 #x在i个维度上的表现优于y，注意是优于，而不是不差于
        for j in range(len(fx)):
            if(fx[j] <= fy[j]):  #至少在1个变量上fx更好，相等i不加
                i += 1
            if(fx[j] > fy[j]): #任何一个变量上fy更好，都说明fx不支配fy
                return False
        if(i== 2): #能走到这一步，说明fy的所有维度都不比fx好，这里追加判断fx是不是完全等于fy，不是则true
            return True
        return False

    #function 14.1
    #输入：x的适应值数组，y的适应值数组,注意这里有先后顺序
    #输出：T or F，x是否支配y
    def is_dominate_by_val(self,x,y):
        #print("x y=",x,y)
        i = 0  # x在i个维度上的表现优于y，注意是优于，而不是不差于
        for j in range(len(x)):
            if (x[j] <= y[j]):  # 至少在1个变量上fx更好，相等i不加
                i += 1
            if (x[j] > y[j]):  # 任何一个变量上fy更好，都说明fx不支配fy
                return False
    #    print("i=",i)
        if (i == 3):  # 能走到这一步，说明fy的所有维度都不比fx好，这里追加判断fx是不是完全等于fy，不是则true
            return True
        return False

    #function 15
    #输入：2进制个体x
    #输出：x点到Z的距离

    def cpt_to_Z_dist(self,x,threat):
        #第1步 解码x，y，计算适应值
        dx = self.decode(x)
        fxx = self.f1(dx, threat, map)
        fyx = self.f3(dx)
        fx = [fxx, fyx]
        #第2步 计算x到Z的距离并返回距离值
        d = 0.0
        for i in range(len(fx)):
            d = d + ((fx[i] - self.Z[i]) ** 2)
        d = np.sqrt(d)
        return d


    #function 16
    #切比雪夫距离计算
    #切比雪夫距离的定义，在2维平面上是国王移动问题，也就是x，y里较大的那个差值
    #输入
    def cal_Tchebycheff(self,w,f,z):
        #return w * abs(f-z)

        return abs(f - z)

    #function 17
    #计算一个个体的切比雪夫距离
    #输入：需要被计算的个体x，以及其位置索引idx

    def Tchbycheff(self,x,idx,threat,map,timewindow):
        #这个函数可以被反复调用，而且确认了x可以变化

        x = np.array(x)
        max = self.Z[0]
        #第1步 解码x，并计算其f1x和f2x，统合成一个2，1数组
        Bi = self.W[idx]
        dx = self.decode(x)
        fxx = self.f1(dx, threat, map)
        fyx = self.f3(dx)
        fzx = self.f4(dx, threat, map, timewindow)
        fx = [fxx, fyx, fzx]
    #    print("fxx=",fxx)
    #    print("Bi=",Bi)
    #    print("fx=", fx)
    #    print("fyx=",fyx)
    #    print("Tchb")
        #第2步 提取idx所对应的W，也就是参考向量值，以及原点值

        for i in range(len(fx)):
            fi = self.cal_Tchebycheff(Bi[i],fx[i],self.Z[i])
            if(fi > max):
                max = fi
        #第3步，调用F16，传参顺序为W[idx]，f[x]，z[x]
        #这3个函数都是2，1的shape，第1个数代表x值，第2个数代表y值，分别计算一次，然后取较大的那个作为max返回即可
    #    print("max=",max)
        return max

    #function 18
    #更新外部存档EP中的id，也就是把新的非支配个体更新进去
    #
    def update_EP_by_ID(self,id,y):
        #如果id对应的个体已经在EP中，那么找到其索引，并直接更新其函数值即可
        if(id in self.EP_x_id):
            pos = self.EP_x_id.index(id)
            self.EP_x_val[pos][:] = y  #这里传进来的y是个值组合，所以可以直接赋值
        return

    #function 19
    #根据y的id，更新EP
    #这个函数只有在出现了疑似更好的个体时才会调用
    #这里传进来的idy是个2进制个体？？
    def update_EP_by_y(self,idy):
        donminate = 0
    #    print("Popval=",self.Pop_val)
        #这里没有把种群的初始函数值计算进去
        #根据idy这个索引，找到其对应的函数值
        #所以这里不该是个个体
    #    print("idy =",idy)
    #    print("EPxVal =",self.EP_x_val)
    #    print("PopVal =", self.Pop_val)
    #    print("PopVal 0 =", self.Pop_val[0,idy])
        fy_1= self.Pop_val[0,idy]
        fy_2= self.Pop_val[1,idy]
        fy_3= self.Pop_val[2,idy]
        fy = [fy_1,fy_2,fy_3]
    #    print("fy1 =", fy_1)
    #    print("fy2 =", fy_2)
    #    print("fy =", fy)

        #fy需要是一个函数值，格式为[f1,f2]

        #创建一个集合，保存需要被删除的个体
        delset = []
        #非劣解的数量
        L = len(self.EP_x_val)
        #循环，遍历这个非劣解集合，逐一判断y和集合内所有个体的支配关系
        for pi in range(L):
        #    print("pi = ",pi)
        #如果y支配了某个个体，则将其加入待删除集合delset中，表明这个个体不被需要
        #这里的支配关系判断其实是正常的，问题是这里传进来的确实他一个都不支配
            if(self.is_dominate_by_val(fy,self.EP_x_val[pi])):
        #        print("dominated")
                delset.append(pi)
        #        print("delset =", delset)
                break
        #如果y被某个个体支配，则donminate+1
        #注意这里，如果y支配了x1，由于x1本来和其他的个体互不支配，所以y不可能被其他个体再支配了
        #相应的，如果y被x1支配，那么y和其他个体的关系要么无关，要么也是被支配
        #换句话说，一次遍历过后，要么donminate =0，要么delset是空集，不会出现d>0且delset非空的情况

            if(donminate != 0):
                break
            if(self.is_dominate_by_val(self.EP_x_val[pi],fy)):
                donminate += 1
        #        print(" non - dominated")
    #    print("delset =",delset)
        #用来暂存新的EP集合
        new_EP_x_id = []
        new_EP_x_val = []
        #第2个循环，遍历原来的非劣解解集，判断个体是否在delset中，不在的话，加入new解集
        for save in range(L):
    #        print("save = ",save)
            if save not in delset:
                new_EP_x_id.append(self.EP_x_id[save])
                new_EP_x_val.append(self.EP_x_val[save])
        #循环结束后，new解集的值赋给非支配解集，这样就产生了不包含y的新非支配解集
        self.EP_x_id = new_EP_x_id
        self.EP_x_val = new_EP_x_val
        #然后单独对于y进行操作，如果donminate = 0，则：
        if(donminate == 0):
        #如果y的id不在支配前沿里，那么直接在前沿里加入y就可以了
            if idy not in self.EP_x_id:
                self.EP_x_id.append(idy)
                self.EP_x_val.append(fy)
        #如果y的id已经在支配前沿里，则直接更新y的函数值即可
            else:
                Y = self.EP_x_id.index(idy)
                self.EP_x_val[Y] = fy[:]

        #这里return 2个东西，一个是新的支配前沿的个体id EP_x_id，另一个是对应的函数值集合EP_x_val
        return self.EP_x_id,self.EP_x_val

    #function 21
    #计算种群的初始值
    #输入：初始种群
    #根据初始种群计算函数值
    #输出：最佳前沿和最佳值2个数组
    #这个函数的输出最后会给Evo函数
    def init_Pop(self,Pop,threat,map,timewindow):
        val1 = np.zeros(shape=len(Pop),)
        val2 = np.zeros(shape=len(Pop),)
        val3 = np.zeros(shape=len(Pop),)
        val1 = np.array(val1)
        val2 = np.array(val2)
        val3 = np.array(val3)

        for i in range(len(Pop)):
            dP = self.decode(Pop[i])
            val1[i] = self.f1(dP,threat,map)
            val2[i] = self.f3(dP)
            val3[i] = self.f4(dP,threat,map,timewindow)

        val = [val1,val2,val3]

        val = np.array(val)
        self.Pop_val = val


        self.EP_x_id = self.fast_non_dominated_sort(val)
        print("Ep init id =", self.EP_x_id)
    #    Tid = self.decode(Pop[self.EP_x_id[1]])
    #    print("Tid =", Tid)
    #    self.EP_x_val = np.array(self.EP_x_val)
        for j in range(len(self.EP_x_id)):
        #    print("j = ",j)
            Xid = self.decode(Pop[self.EP_x_id[j]])
            Xid = np.array(Xid)
        #    print("Xid = ", Xid)
        #    self.EP_x_val[j,0] = self.f1(Xid,threat,map)
            f1 = self.f1(Xid,threat,map)
            #这里可以正确定位内容
        #    self.EP_x_val[j,1] = self.f3(Xid)
            f2 = self.f3(Xid)
            f3 = self.f4(Xid,threat,map,timewindow)
            f = [f1,f2,f3]
            self.EP_x_val.append(f)
        print("init val =",self.EP_x_val)
        return

    #function 22
    #非支配排序，返回非支配解，用于计算种群的初始Pareto前沿，送到self.Pop函数里去
    def fast_non_dominated_sort(self, values):
            values11 = values[0]  # 函数1解集嘛6
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

            return front[0]

    def HV(self, front, num):
        # 这里需要排序


        front = sorted(front,key=lambda x:x[0])
        front = np.array(front)



        hv = 0.0
        volume = 0.0
        for i in range(0, num):
            if (i == 0):
                volume = (24-front[i,0])  * (15-front[i,1])
            if (i > 0):
                volume = (24-front[i,0])  * (front[i-1,1] - front[i,1])

            hv = hv + volume

        return hv

    #function 20
    #MOEA/D程序主函数
    #使用MOEA/D算法的思想进行循环并得出解集
    #实际上循环在evolution那里完成了
    def main(self):
        pos = self.getpos()  # 生成目标点初始位置和无人机初始位置
        map = self.getmap(pos)  # 计算距离
    #    print("init map =",self.map)
        threat = self.threat_initialize(pos)
        parent = np.zeros((self.NP, self.num_target + self.singletasktarget, 2))
        parent = np.random.randint(0,2,(self.NP, self.Len))
    #    self.EP_x_val = [[0.0]*2 for i in range(self.NP)]
        self.EP_x_val = []
    #    self.EP_x_val = np.array(self.EP_x_val)
    #    print(self.EP_x_val)
    #    print(len(parent))
    #    print("parent=",parent)
        self.Pop = np.array(parent)
        Pop = copy.copy(self.Pop)
        timewindow = np.zeros((self.num_target, 2))
        timewindow = np.array(timewindow)
        timewindow = self.settimewindow(timewindow)
        self.init_Pop(Pop,threat,map,timewindow)




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


    #    print("pop =",self.Pop)
        self.Load() # 逻辑正确，可以正常划分不同的搜索方向
    #    print("W =",self.W)
        self.cpt_W_Bi_T() # 逻辑正确，可以找到每个方向的邻居
        print("WbiT = ", self.W_Bi_T)

        result,f1_values = self.evolution(threat,map,Pop,timewindow)
        #这里暂时无法输出正确的结果，整合后的代码存在逻辑上的bug和一些语法问题，目前在改的是这个部分
        print(result)
        #这里为什么还会输出互相支配的解
        #交叉变异过程没有问题
        print(self.EP_x_val)
        Ans = []

        Ans.append(self.EP_x_val)
       # Ans.append(front1_values)



        print("ans = ", Ans)
        output = pd.DataFrame(Ans)
        writer = pd.ExcelWriter('moead dset2 15 .xlsx')
        output.to_excel(writer, sheet_name='example1', float_format='%.4f')
        writer._save()
        writer.close()


        return



if __name__ == '__main__':
    moead = MOEAD()
    moead.main()

