#!/usr/bin/env python
# coding=utf-8
'''
@Author: lzwgiter & JK211
@Email: float311@163.com & jerryren2884@gmail.com
@Date: 2022-07-1 11:09:49
@LastEditor: JK211
LastEditTime: 2022-08-01 11:10:11
@Discription: This .py is the gossip environment of NS3 inherit from gym.Env
@Environment: python 3.8
'''
import socket
import time

import gym
import torch
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete
import numpy as np

import logging

import ns.applications as NSApplications
import ns.core as NSCore
import ns.internet as NSInternet
import ns.network as NSNetwork
import ns.wifi as NSWifi
import ns.mobility as NSMobility

# 开启日志记录
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)
#NSCore.LogComponentEnable("wifi-adhoc-app", NSCore.LOG_PREFIX_TIME + NSCore.LOG_LEVEL_INFO)


class GossipEnvironment(gym.Env):
    def __init__(self, argv):
        """
        初始化NS-3环境以及无人机gossip数据传输环境
        :param argv: NS-3参数
        """
        self.argv = argv
        self.cmd = NSCore.CommandLine()
        self.cmd.distance = 20
        self.cmd.numNodes = 5
        self.cmd.simuTime = 20
        self.cmd.is_reset = True
        self.start_time=None

        # 读取命令行参数
        self.cmd.AddValue("distance", "size of map")
        self.cmd.AddValue("numNodes", "number of uav nodes")
        self.cmd.AddValue("simuTime", "time of simulation")
        self.cmd.Parse(self.argv)

        self.cmd.distance = int(self.cmd.distance)

        self.cmd.numNodes = int(self.cmd.numNodes)
        self.cmd.simuTime = int(self.cmd.simuTime)

        high = int(self.cmd.numNodes)
        self.observation_space = Dict(
            {
                "is_received_mi": MultiBinary(self.cmd.numNodes*self.cmd.numNodes),
                "message_redundancy_Ri": Box(low=0, high=high, shape=(1,), dtype=int),
                "filling_percent_Bi": Box(low=0, high=1.0, shape=(1,), dtype=float),
                "prob":Box(low=0, high=10.0, shape=(1,), dtype=float)
            }
        )
        self.action_space = Dict(
            {
                "forwarding_probability_Pg": Box(low=0, high=10, shape=(self.cmd.numNodes,), dtype=int),
                "forward_neighbor_Nnb": MultiBinary(self.cmd.numNodes)
            }
        )

    def step(self, action=None, is_reset=True, snap_bitgrahp=None,is_first=False):
        """
        执行动作
        :param episode:
        :param action: 要执行的动作 numpy.ndarray
        :return: 返回当前状态、奖励以及一个指示当前episode是否结束的布尔值done
        """

        phy_mode = "DsssRate1Mbps"
        propagation_loss_model = "ns3::LogDistancePropagationLossModel"
        propagation_delay_model = "ns3::ConstantSpeedPropagationDelayModel"
        mobility_model = "ns3::ConstantPositionMobilityModel"

        ######################################################################
        #                                                                    #
        #                         init the UAVNET                            #
        #                                                                    #
        ######################################################################
        # 创建节点
        uav_nodes = NSNetwork.NodeContainer()
        uav_nodes.Create(self.cmd.numNodes)

        # 配置WIFI
        wifi_channel = NSWifi.YansWifiChannelHelper()
        wifi_phy = NSWifi.YansWifiPhyHelper()

        wifi_channel.SetPropagationDelay(propagation_delay_model)
        wifi_channel.AddPropagationLoss(propagation_loss_model)
        wifi_phy.SetChannel(wifi_channel.Create())

        wifi_mac = NSWifi.WifiMacHelper()
        wifi_mac.SetType("ns3::AdhocWifiMac")

        wifi = NSWifi.WifiHelper()
        wifi.SetStandard(NSWifi.WIFI_PHY_STANDARD_80211b)
        wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                     "DataMode", NSCore.StringValue(phy_mode),
                                     "ControlMode", NSCore.StringValue(phy_mode))

        wifi_devices = wifi.Install(wifi_phy, wifi_mac, uav_nodes)
        wifi_phy.EnablePcap("uav-gossip", wifi_devices)

        # 安装网络模型
        stack = NSInternet.InternetStackHelper()
        stack.Install(uav_nodes)
        address = NSInternet.Ipv4AddressHelper()
        address.SetBase(NSNetwork.Ipv4Address("192.168.1.0"), NSNetwork.Ipv4Mask("255.255.255.0"))
        address.Assign(wifi_devices)

        # 分配随机位置
        rand_pos_locator = NSMobility.RandomBoxPositionAllocator()
        play_ground = "ns3::UniformRandomVariable[Min=0.0|Max=" + str(self.cmd.distance) + "]"
        rand_pos_locator.SetAttribute("X", NSCore.StringValue(play_ground))
        rand_pos_locator.SetAttribute("Y", NSCore.StringValue(play_ground))

        # 安装移动模型
        mobility = NSMobility.MobilityHelper()
        mobility.SetPositionAllocator(rand_pos_locator)
        mobility.SetMobilityModel(mobility_model)
        mobility.Install(uav_nodes)

        ######################################################################
        #                                                                    #
        #                  init gossip applications                          #
        #                                                                    #
        ######################################################################
        for i in range(self.cmd.numNodes):
            # 创建gossip应用，包括一个发送端和一个接收端，用于处理发包、收包
            sender = NSApplications.AppSender()
            receiver = NSApplications.AppReceiver()



            # 判断是否需要建立邻居列表
            sender.SetReset(is_reset)

            # 为接收端设置网络节点个数，用于判断是否完成来数据共享
            receiver.SetReset(self.cmd.is_reset)
            receiver.SetNumNodes(self.cmd.numNodes)

            uav_nodes.Get(i).AddApplication(sender)
            uav_nodes.Get(i).AddApplication(receiver)

            receiver.SetStartTime(NSCore.Seconds(0))
            sender.SetStartTime(NSCore.Seconds(1 + 0.0001*i))

            receiver.SetStopTime(NSCore.Seconds(self.cmd.simuTime))
            sender.SetStopTime(NSCore.Seconds(self.cmd.simuTime))

            receiver.SetSenderApp(sender)

            if not is_reset:
                sender.SetRecvMsgDRL(snap_bitgrahp[i])
                receiver.SetRecvMsgDRL(snap_bitgrahp[i])
                sender.SetProbForwardbyDRL(
                    [action[i][0]]*10)
                neighberList = []
                if is_first:
                    if (i == 0):
                        for k in range(self.cmd.numNodes):
                            if ((action[i][1] % 2) == 1):
                                neighberList.append(NSNetwork.Ipv4Address("192.168.1." + str(k + 1)))
                            action[i][1] = int(action[i][1] / 2)
                else:
                    for k in range(self.cmd.numNodes):
                        if ((action[i][1] % 2) == 1):
                            neighberList.append(NSNetwork.Ipv4Address("192.168.1." + str(k + 1)))
                        action[i][1] = int(action[i][1] / 2)
                    if (len(snap_bitgrahp[i]) == 0):
                        neighberList = []

                sender.SetNeighberListbyDRL(neighberList)
            else:
                sender.SetRecvMsgDRL([])
                receiver.SetRecvMsgDRL([])

        NSCore.Simulator.Stop(NSCore.Seconds(self.cmd.simuTime))

        NSCore.Simulator.Run()

        uav_counterBitgraph_status = []
        uav_communicable_Adj = np.zeros((1, self.cmd.numNodes, self.cmd.numNodes))
        done = True
        # 收集数据
        for i in range(self.cmd.numNodes):
            counterBitgraph = list(uav_nodes.Get(i).GetApplication(1).GetRecvMessage())
            #
            if len(counterBitgraph) < 1:  # 判断是否所有无人机都收齐了数据包
                done = False
            uav_counterBitgraph_status.append(counterBitgraph)

            for j in range(len(counterBitgraph)):
                adj = counterBitgraph[j][0].split('.')
                index_adj = int(adj[-1]) - 1
                uav_communicable_Adj[0][i][index_adj] = 1

        NSCore.Simulator.Destroy()

        uav_obervations = self.get_observation(uav_counterBitgraph_status,action)
        if is_reset:

            return torch.Tensor(uav_obervations), uav_communicable_Adj, uav_counterBitgraph_status,
        else:
            uav_rewards = self.reward(uav_counterBitgraph_status, snap_bitgrahp)

        return torch.Tensor(uav_obervations), torch.Tensor(uav_communicable_Adj).squeeze(0), uav_rewards, done, uav_counterBitgraph_status

    def get_observation(self, status,action):
        uav_observations = []
        tmp = [0] * self.cmd.numNodes * self.cmd.numNodes
        for i in range(len(status)):
            i_stat=status[i]
            for j in i_stat:
                adj = j[0].split('.')
                index_adj = int(adj[-1]) - 1
                tmp[i*self.cmd.numNodes+index_adj] = 1
        for  i in range(len(status)):
            i_stat=status[i]
            observation=tmp.copy()
            num_ReduntMess = 0
            Bit_Percent = len(i_stat) / self.cmd.numNodes
            observation.append(Bit_Percent)
            for j in i_stat:
                num_ReduntMess = num_ReduntMess + j[1]
            observation.append(num_ReduntMess)
            if (action == None):

                observation.append(10)
            else:
                observation.append(action[i][0])
            uav_observations.append(observation)


        return uav_observations

    def reward(self, status, snap):
        uav_rewards = []
        is_ok=True
        reward=0

        for i in range(len(status)):
            i_stat = status[i]
            i_snap = snap[i]
            Bitpercent_snap = len(i_snap) / self.cmd.numNodes
            Bitpercent_stat = len(i_stat) / self.cmd.numNodes
            if Bitpercent_stat - Bitpercent_snap > 0:
                reward_1 = (Bitpercent_stat - Bitpercent_snap)
            else:
                reward_1 = 0

            if(len(i_stat)!= 1):
                is_ok=False

        if is_ok:
            end_time=time.time()
            print("time is {}".format(time.time() - self.start_time))

        return [reward]*len(status)

    def reset(self):
        """
        重置环境,让无人机重新发现邻居
        :return:
        """
        obs, adj, snap = self.step()


        self.start_time=time.time()
        return torch.Tensor(obs), torch.Tensor(adj).squeeze(0), snap

    def render(self, mode="human"):
        pass
