from SingleLaneIDMAge.SimulatorCode.tfLight import RandomIntervalGenerator
import random
from gym.envs.registration import EnvSpec
from gym.spaces import Discrete, Box
import gym
import yaml
from collections import deque
import math
import copy
import sys
import pickle
import time
import pylab
import pygame
import matplotlib.patches as mpatches
import matplotlib.backends.backend_agg as agg
import matplotlib
import numpy as np
import os
import sys
import cv2
sys.path.append(os.getcwd() + "/")

matplotlib.use("Agg")

# Action Maping
# action 0 - Accelerate
# action 1 - Decelerate
# action 2 - Do Nothing


class TrafficSim(gym.Env):

    def __init__(self, config):

        self.enable_seed = config["enable-seed"]
        if self.enable_seed:
            np.random.seed(1)
            random.seed(1)

        self.render = config["render"]
        self.enable_frame_capture = config["enable-frame-capture"]
        self.time_period = config["time-period"]
        self.time_elapsed = 0.0
        self.trajec_file = config["trajec-file-path"]
        self.read_trajectory()
        self.max_cars = 31
        self.num_cars = np.zeros(3, dtype=np.int32)
        self.random_trajec = np.zeros(1, dtype=np.int32)
        self.agent_lane = None
        self.lane_radius = [233]
        self.car_radius = 2  # in m
        self.car_length = 2*self.car_radius  # in m
        self.min_car_distance = 2  # in m
        self.x_pixel_one_metre = 6
        self.delta_theta = [np.rad2deg(
            ((self.car_length + self.min_car_distance) * self.x_pixel_one_metre) / self.lane_radius[0])]
        self.render_grid = config["render-grid"]
        self.cell_size = config["cell-size"]
        self.view_size = config["view-size"]
        self.polygon_points = 10
        #self.comm_mode = config["comm-mode"]
        self.comm_mode = True
        self.frac_cells = config["frac-cells"]
        self.regions_width = config["region-width"]
        self.update_graphs = config["update-graphs"]
        self.update_graphs_method = config["update-graphs-method"]
        self.collision_cost = config["collision-cost"]
        self.reward_alpha = config["reward-alpha"]
        #self.acc_noise = config["acc-noise"]
        self.external_controller = config["external-controller"]
        self.headway_thershold = config["headway-thershold"]
        self.enable_traffic_light = config["enable-traffic-light"]
        self.horizon = config["horizon"]
        self.test_mode = config["test-mode"]
        self.test_mode_trajec_file = config["test-file-path"]
        self.config_file = config
        # print(config)
        if self.enable_traffic_light:
            self.readTFConfig()
            self.genObj = RandomIntervalGenerator(int(self.min_duration / self.time_period), int(
                self.max_duration/self.time_period), self.num_stops, 1, self.horizon, self.num_tries)

        self.findLocalview()
        print("Local View for cars : ", self.localView_in_m)

        self.view_size_theta = [np.rad2deg(
            (self.view_size*self.x_pixel_one_metre)/self.lane_radius[0])]

        if self.update_graphs_method not in ["vel", "occ"]:
            print("Invalid update graphs method : %s" %
                  (self.update_graphs_method))
            sys.exit(-1)

        self.densities = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        #self.densities = [0.4]

        if ((self.view_size % self.cell_size) != 0):
            print("Error : View Size should be divisible by cell size")
            sys.exit(-1)

        self.num_cols = int(2*(self.view_size/self.cell_size))
        self.occ_grid = np.zeros((1, self.num_cols), dtype=np.int32)
        self.vel_grid = np.zeros((1, self.num_cols), dtype=np.float32)

        self.extended_view_cols = int(2 * ((self.view_size - self.localView_in_m) / self.cell_size))
        self.max_age_value = 2 * self.extended_view_cols
        self.age_time_step = self.time_period / self.max_age_value

        # print(self.num_cols)
        self.observation_space = Box(-float("inf"), float("inf"),
                                     shape=((2 * self.num_cols) + self.extended_view_cols, ), dtype=np.float)
        self.region_maping = {}

        if (self.comm_mode):

            comm_cells = int(self.frac_cells*(self.num_cols/2))

            if ((comm_cells % self.regions_width) != 0):
                print('Error : Num of regions should divide the comm cells exactly')
                print('Comm Cells : %d' % (comm_cells))
                sys.exit(-1)
            else:
                print('Using %1.f metre of area as communication region.' %
                      (comm_cells*self.cell_size))

                start = 0
                num_regions = int(comm_cells/self.regions_width)

                for i in range(0, num_regions):
                    self.region_maping['reg_'+str(i)] = list(np.linspace(
                        start, start+self.regions_width-1, num=self.regions_width, dtype='int'))
                    start = start + self.regions_width

                start = (self.num_cols-1) - comm_cells + 1
                for i in range(num_regions, 2*num_regions):
                    self.region_maping['reg_'+str(i)] = list(np.linspace(
                        start, start+self.regions_width-1, num=self.regions_width, dtype='int'))
                    start = start + self.regions_width

        # print(self.region_maping)
        # ---- IDM Parameters ---- #
        self.a = 0.73
        self.b = 1.67

        if config["use-vel"]:
            max_vel = config["max-vel"]
        else:
            max_vel = self.cal_vel(self.b, self.view_size)

        self.v_not = max_vel
        self.max_vel = max_vel
        self.other_max_vel = config["other-max-vel"]

        self.T = 1.5
        self.delta = 4
        self.s_not = 2
        self.bsafe = 4*self.b

        # ----- ACTION MAP ----- #
        self.action_map = {0: "Accelerate",
                           1: "Decelerate", 2: "Do Nothing", None: "None"}

        # ------ PYGAME VARIABLES ------ #

        self.centre = (250, 250)
        self.radius = 250
        self.lane_width = 5
        self.lane_thickness = 2
        self.orig_lane_radius = [350, 252]
        self.fps = config["fps"]
        self.lane_map = {}

        self.lab2ind = {'angle': 0, 'vel': 1, 'lane': 2, 'agent': 3, "id": 4}

        self.init_action_decoder()
        self.local_indexes = self.findLocalViewIndex()
        # self.init_randomize_cars()

        if self.render:
            self.init_render()
            self.fig = pylab.figure(tight_layout=True, figsize=(4, 4))
            self.ax = self.fig.gca()
            # self.ax.grid()
            self.ax.set_title("Lidar Data")
            labels = {0: 'Ego Veh.', 1: 'Other Veh.', 2: "Unknown Area"}
            colors = ["green", "black", "red"]
            patches = [mpatches.Patch(
                color=colors[i], label=labels[i]) for i in [0, 1, 2]]
            self.ax.legend(handles=patches, loc="lower right")
            self.img = self.ax.imshow(np.random.rand(
                self.num_cols, self.num_cols, 3))

    def set_plots(self, data):

        self.img.set_data(data)
        canvas = agg.FigureCanvasAgg(self.fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()

        return raw_data, canvas.get_width_height()

    def init_action_decoder(self):

        isEmpty = bool(self.region_maping)
        self.plan_map = {"acc": 0, "dec": 1, "do-nothing": 2}
        self.plan_map_reverse = {0: "acc", 1: "dec", 2: "do-nothing"}

        self.action_map = {}

        if isEmpty == False:
            self.action_map[0] = "acc"
            self.action_map[1] = "dec"
            self.action_map[2] = "do-nothing"
        else:
            possible_query = []

            [possible_query.append(query)
             for query in self.region_maping.keys()]
            possible_query.append("NULL")

            possible_plan = ["acc", "dec", "do-nothing"]

            count = 0
            for plan in possible_plan:
                for query in possible_query:
                    self.action_map[count] = str(plan)+"&"+str(query)
                    count += 1

        self.action_space = Discrete(len(self.action_map))

    def cal_vel(self, dec, dist):
        u_max = np.sqrt(2 * dec * dist)
        time_period_dist = self.cal_dist_travelled(
            u_max, self.a, self.time_period)
        effective_distance = self.view_size - time_period_dist - \
            self.min_car_distance - (self.car_length/2)
        final_u_max = np.sqrt(2 * dec * effective_distance)
        return final_u_max

    def findLocalview(self):
        self.localView_in_m = 0.0
        if self.comm_mode:
            self.localView_in_m = self.view_size - \
                (self.view_size * self.frac_cells)
        else:
            self.localView_in_m = self.view_size

    def readTFConfig(self):
        self.max_duration = self.config_file["tf-config"]["max-dur"]
        self.min_duration = self.config_file["tf-config"]["min-dur"]
        self.num_tries = self.config_file["tf-config"]["num-tries"]
        self.num_stops = self.config_file["tf-config"]["num-stops"]

    def expandPts(self, ptslist):
        expandedPts = []
        for a in ptslist:
            expandedPts.append(a[0])
            expandedPts.append(a[0]+a[1])

        return expandedPts

    def findLocalViewIndex(self):
        all_index = list(np.arange(0, self.num_cols))
        
        for reg in self.region_maping.keys():
            cols = self.region_maping[reg]
            for col in cols:
                all_index.remove(col)

        return all_index

    def init_render(self):
        pygame.init()
        pygame.font.init()
        self.road_color = (97, 106, 107)
        self.color_background = (30, 132, 73)
        self.color_white = (255, 255, 255)
        self.color_red = (236, 112, 99)
        self.color_yellow = (247, 220, 111)
        self.color_grey = (192, 192, 192, 80)
        self.color_lime = (128, 250, 0)
        self.dark_green = (0, 120, 0)
        self.light_x_y = (2*self.radius + 10, self.radius - 110)

        self.screen = pygame.display.set_mode(
            (2*self.radius + 500, 2*self.radius))
        pygame.display.set_caption('Traffic Simulator')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Comic Sans MS', 25)
        self.font2 = pygame.font.SysFont("Comic Sans MS", 20)

        # ---- DISPLAY BOX -----#
        self.text_board = pygame.Surface((200, 200), pygame.SRCALPHA)
        self.text_board.fill((128, 128, 128))
        self.text_board.set_alpha(80)

        self.red_light = pygame.image.load(
            os.getcwd() + "/" + "SingleLaneIDMAge/SimulatorCode/images/red.png").convert_alpha()
        self.green_light = pygame.image.load(
            os.getcwd() + "/" + "SingleLaneIDMAge/SimulatorCode/images/green.png").convert_alpha()

        # --- DISPLAY BOX END ---#

    def read_trajectory(self):
        with open(self.trajec_file, 'rb') as handle:
            self.trajectories = pickle.load(handle)

    def road_boundary(self, surface, color, radius, width, pos):
        pygame.draw.circle(surface, color, pos, radius, width)

    def road(self, surface, color, radius, width, pos):
        pygame.draw.circle(surface, color, pos, radius, width)

    def metre_to_pixels(self, metre):
        return metre*self.x_pixel_one_metre

    def get_coordinates(self, angle, radius):
        angle_rad = np.deg2rad(angle)
        X = self.radius + np.cos(angle_rad)*radius
        Y = self.radius + np.sin(angle_rad)*radius
        return X, Y

    def build_lane_map(self):
        self.lane_map = {0: self.vehicles_list_lane0,
                         1: self.vehicles_list_lane1, 2: self.vehicles_list_lane2}

    def draw_car(self, centre, color):
        pygame.draw.circle(self.screen, color, centre,
                           self.metre_to_pixels(self.car_radius), 0)

    def return_arc_length(self, radius, theta_deg):
        arc_length = (np.pi*radius*theta_deg)/180
        arc_length_in_metres = arc_length * (1/self.x_pixel_one_metre)

        return arc_length_in_metres

    def cal_acc(self, s_alpha, delta_v_alpha, v_alpha, v_not):

        relative_vel = None
        bump_bump_dis = None

        if s_alpha < self.localView_in_m:
            relative_vel = delta_v_alpha
            bump_bump_dis = s_alpha
        else:
            allowed_max_lv = np.sqrt(2 * self.b * self.localView_in_m)
            relative_vel = v_alpha - allowed_max_lv
            # print(allowed_max_lv)
            bump_bump_dis = self.localView_in_m

        s_star = self.s_not + \
            max(0, ((v_alpha*self.T) + ((v_alpha*relative_vel)/(2*np.sqrt(self.a*self.b)))))
        acc = self.a * (1 - ((v_alpha/v_not) **
                             self.delta) - ((s_star/bump_bump_dis)**2))

        '''
    	if self.acc_noise:
    		noise = np.random.normal(0.0, .2)
    		return acc + noise
    	'''
        return acc

    def cal_dist_travelled(self, u, acc, time):
        dist = (u*time) + (0.5*acc*time*time)
        return dist

    def cal_new_velocity(self, u, acc, time):
        vel = u + (acc*time)
        return vel

    def get_theta(self, radius, arc_length, act):
        # S = theta * r
        theta_rad = arc_length/radius
        theta_deg = np.rad2deg(theta_rad)
        if theta_deg < 0:
            print('Negative angle caused by ', act)
            print(theta_deg)
            print(theta_rad)
            sys.exit(-1)
        return theta_deg

    def draw_graphics(self, reward, action, query):

        speed_idx = [i for i, tup in enumerate(
            self.lane_map_list[self.agent_lane]) if tup[self.lab2ind["agent"]] == 1][0]

        speed = self.lane_map_list[self.agent_lane][speed_idx][self.lab2ind['vel']]

        self.screen.fill(self.color_background)
        # for lane 1

        self.road_boundary(self.screen, self.color_white,
                           self.radius, self.lane_thickness, self.centre)
        self.road(self.screen, self.road_color, self.radius -
                  self.lane_thickness, 0, self.centre)
        self.road_boundary(self.screen, self.color_white, self.radius-self.lane_thickness -
                           self.metre_to_pixels(self.lane_width), self.lane_thickness, self.centre)

        if self.render_grid:
            self.draw_occupancy()

        x, y = self.get_coordinates(0.0, self.lane_radius[0])
        lane_string = "lane : %d" % (0)
        lane_text = self.font2.render(lane_string, False, (0, 0, 0))
        self.screen.blit(lane_text, (x+30, y))

        for lane in range(0, 1):
            for car in range(0, len(self.lane_map_list[lane])):
                x, y = self.get_coordinates(
                    self.lane_map_list[lane][car][self.lab2ind['angle']], self.lane_radius[lane])
                if (self.lane_map_list[lane][car][self.lab2ind['agent']] == 0):
                    self.draw_car((int(x), int(y)), self.color_yellow)
                elif(self.lane_map_list[lane][car][self.lab2ind['agent']] == 1):
                    self.draw_car((int(x), int(y)), self.color_lime)
                    # print(lane)
                self.screen.blit(self.text_board, (150, 150))
                speed_string = 'Agent Speed : %.2f ' % (speed*3.6)+' km/hr'
                speed_text = self.font.render(speed_string, False, ((0, 0, 0)))
                area_string = 'Visiblity : '+str(self.view_size)+' m'
                area_text = self.font.render(area_string, False, (0, 0, 0))
                reward_string = 'Agent Reward : %.2f' % (reward)
                reward_text = self.font.render(reward_string, False, (0, 0, 0))
                time_string = 'Time Elapsed : %.1f s' % (self.time_elapsed)
                time_text = self.font.render(time_string, False, (0, 0, 0))
                sample_string = 'Sampling Rate : %d Hz' % (
                    int(1/self.time_period))
                sample_text = self.font.render(sample_string, False, (0, 0, 0))
                if action == None:
                    action_string = 'Action : None'
                else:
                    action_string = 'Action : '+self.plan_map_reverse[action]
                action_text = self.font.render(action_string, False, (0, 0, 0))
                query_string = 'Query : '+str(query)
                query_text = self.font.render(query_string, False, (0, 0, 0))
                maxSpeed_string = 'Max Agent Speed : %.2f km/hr' % (
                    self.max_vel * 3.6)
                maxSpeed_text = self.font.render(
                    maxSpeed_string, False, (0, 0, 0))
                maxSpeed_nonEgostring = "Non-Ego Max: %.2f km/hr" % (
                    self.other_max_vel * 3.6)
                maxSpeed_nonEgostring_text = self.font.render(
                    maxSpeed_nonEgostring, False, (0, 0, 0))

                self.screen.blit(speed_text, (155, 155))
                self.screen.blit(area_text, (155, 180))
                self.screen.blit(time_text, (155, 205))
                self.screen.blit(sample_text, (155, 230))
                self.screen.blit(reward_text, (155, 255))
                self.screen.blit(action_text, (155, 280))
                self.screen.blit(query_text, (155, 305))
                self.screen.blit(maxSpeed_text, (155, 330))
                self.screen.blit(maxSpeed_nonEgostring_text, (155, 355))

        if self.update_graphs:
            self.get_lidar_data()
            raw_string, size = self.set_plots(self.lidar_data)
            surf = pygame.image.fromstring(raw_string, size, "RGB")
            self.screen.blit(surf, (2*self.radius+50, 50))

        if self.enable_traffic_light:
            if self.set_red_light:
                self.screen.blit(self.red_light, self.light_x_y)
            else:
                self.screen.blit(self.green_light, self.light_x_y)

            # Draw traffic Line
            x1, y1 = self.get_coordinates(0.0, self.lane_radius[0] + 20)
            x2, y2 = self.get_coordinates(0.0, self.lane_radius[0] - 20)
            pygame.draw.line(self.screen, (0, 0, 0), (x1, y1), (x2, y2), 3)

        pygame.display.flip()

        if self.enable_frame_capture:
            pass
        else:
            self.clock.tick(self.fps)

    def get_lidar_data(self):

        self.lidar_data = np.ones((self.num_cols, self.num_cols, 3))
        # print(self.lidar_data.shape)
        if self.update_graphs_method == "occ":
            half = int(self.num_cols/2)
            ahead_part1 = np.flip(self.occ_grid[0][half:], axis=0)
            #ahead_part2 = np.flip(self.occ_grid[1][half:], axis=0)
            #ahead_part3 = np.flip(self.occ_grid[2][half:], axis=0)
            ahead_part = {0: ahead_part1}

            back_part1 = np.flip(self.occ_grid[0][0:half], axis=0)
            #back_part2 = np.flip(self.occ_grid[1][0:half], axis=0)
            #back_part3 = np.flip(self.occ_grid[2][0:half], axis=0)
            back_part = {0: back_part1}

            space = int(self.num_cols/6)
            # print(space)
            line1 = space
            line2 = line1 + space
            line3 = line2 + space
            line4 = line3 + space

            lane1_mid = line1 + int((line2 - line1)/2)
            lane2_mid = line2 + int((line3 - line2)/2)
            lane3_mid = line3 + int((line4 - line3)/2)
            mid_map = {0: lane1_mid, 1: lane2_mid, 2: lane3_mid}
            #print(lane1_mid, lane2_mid, lane3_mid)
            #self.lidar_data[line1][:][:] = 0
            #self.lidar_data[line2][:][:] = 0
            #self.lidar_data[line3][:][:] = 0
            #self.lidar_data[line4][:][:] = 0

            for lane in range(0, 1):

                for i in range(0, ahead_part1.shape[0]):

                    if ahead_part[lane][i] == 1:
                        self.lidar_data[mid_map[lane]][i][0] = 0
                        self.lidar_data[mid_map[lane]][i][1] = 0
                        self.lidar_data[mid_map[lane]][i][2] = 0

                    if ahead_part[lane][i] == 2:
                        self.lidar_data[mid_map[lane]][i][0] = 0
                        self.lidar_data[mid_map[lane]][i][1] = 1
                        self.lidar_data[mid_map[lane]][i][2] = 0

                    if ahead_part[lane][i] == -1:
                        self.lidar_data[mid_map[lane]][i][0] = 1
                        self.lidar_data[mid_map[lane]][i][1] = 0
                        self.lidar_data[mid_map[lane]][i][2] = 0

            for lane in range(0, 1):

                for i in range(0, back_part1.shape[0]):

                    if back_part[lane][i] == 1:
                        self.lidar_data[mid_map[lane]][half + i][0] = 0
                        self.lidar_data[mid_map[lane]][half + i][1] = 0
                        self.lidar_data[mid_map[lane]][half + i][2] = 0

                    if back_part[lane][i] == 2:
                        self.lidar_data[mid_map[lane]][half + i][0] = 0
                        self.lidar_data[mid_map[lane]][half + i][1] = 1
                        self.lidar_data[mid_map[lane]][half + i][2] = 0

                    if back_part[lane][i] == -1:
                        self.lidar_data[mid_map[lane]][half + i][0] = 1
                        self.lidar_data[mid_map[lane]][half + i][1] = 0
                        self.lidar_data[mid_map[lane]][half + i][2] = 0

        else:
            half = int(self.num_cols/2)
            ahead_part1 = np.flip(self.vel_grid[0][half:], axis=0)
            ahead_part2 = np.flip(self.vel_grid[1][half:], axis=0)
            ahead_part3 = np.flip(self.vel_grid[2][half:], axis=0)
            ahead_part = {0: ahead_part1, 1: ahead_part2, 2: ahead_part3}

            back_part1 = np.flip(self.vel_grid[0][0:half], axis=0)
            back_part2 = np.flip(self.vel_grid[1][0:half], axis=0)
            back_part3 = np.flip(self.vel_grid[2][0:half], axis=0)
            back_part = {0: back_part1, 1: back_part2, 2: back_part3}

            space = int(self.num_cols/6)
            # print(space)
            line1 = space
            line2 = line1 + space
            line3 = line2 + space
            line4 = line3 + space

            lane1_mid = line1 + int((line2 - line1)/2)
            lane2_mid = line2 + int((line3 - line2)/2)
            lane3_mid = line3 + int((line4 - line3)/2)
            #print(lane1_mid, lane2_mid, lane3_mid)
            mid_map = {0: lane1_mid, 1: lane2_mid, 2: lane3_mid}

            #self.lidar_data[line1][:][:] = 0
            #self.lidar_data[line2][:][:] = 0
            #self.lidar_data[line3][:][:] = 0
            #self.lidar_data[line4][:][:] = 0

            for lane in range(0, 3):

                for i in range(0, ahead_part3.shape[0]):

                    if ahead_part[lane][i] > 0.0:
                        self.lidar_data[mid_map[lane]][i][0] = 0
                        self.lidar_data[mid_map[lane]][i][1] = 0
                        self.lidar_data[mid_map[lane]][i][2] = 0

                    if ahead_part[lane][i] == -1:
                        self.lidar_data[mid_map[lane]][i][0] = 1
                        self.lidar_data[mid_map[lane]][i][1] = 0
                        self.lidar_data[mid_map[lane]][i][2] = 0

            for lane in range(0, 3):

                for i in range(0, back_part1.shape[0]):

                    if back_part[lane][i] > 0.0:
                        self.lidar_data[mid_map[lane]][half + i][0] = 0
                        self.lidar_data[mid_map[lane]][half + i][1] = 0
                        self.lidar_data[mid_map[lane]][half + i][2] = 0

                    if back_part[lane][i] == -1:
                        self.lidar_data[mid_map[lane]][half + i][0] = 1
                        self.lidar_data[mid_map[lane]][half + i][1] = 0
                        self.lidar_data[mid_map[lane]][half + i][2] = 0

    def action_decoder(self, act):

        if not self.comm_mode:
            action, query = act, "NULL"
        else:
            action, query = self.action_map[act].split("&")
            action = self.plan_map[action]

        return (action, query)

    def create_dummy_traffic_tuple(self, lane):
        light_tuple = self.lane_map_list[0][0].copy()
        light_tuple[self.lab2ind["angle"]] = 0.0
        light_tuple[self.lab2ind["vel"]] = 0.0
        light_tuple[self.lab2ind["lane"]] = lane
        light_tuple[self.lab2ind["agent"]] = 2  # Denotes Traffic Light
        light_tuple[self.lab2ind["id"]] = -1

        return light_tuple

    def agevec2obs(self, array, init_t=False):

        assert array.shape[0] == self.occ_grid.shape[1]

        if init_t:
            for key in self.region_maping:
                for col in self.region_maping[key]:
                    array[col] = 0

        obs = []
        for col in range(0, array.shape[0]):
            if array[col] != -1:
                obs.append(array[col])

        obs = np.array(obs)

        assert obs.shape[0] == self.extended_view_cols

        return obs

    def reset(self, density=None):

        if density == None:
            density = self.densities[np.random.randint(0, len(self.densities))]

        self.cur_density_val = density
        #self.cur_density_val = 0.3

        self.time_elapsed = 0.0
        self.num_steps = 0
        self.num_cars = int(density*self.max_cars)
        self.random_trajec[0] = np.random.randint(
            0, self.trajectories[str(density)]['lane0']['total_count'])

        self.lane_map = {}
        self.lane_map[0] = np.array(self.trajectories[str(density)]['lane0']['data'][self.random_trajec[0]], dtype=[
                                    ('angle', 'f8'), ('vel', 'f8'), ('lane', 'i4'), ('agent', 'i4'), ('id', 'i4')])

        # print(self.lane_map[0])
        self.lane_map_list = {}
        self.lane_map_list[0] = list(self.lane_map[0])

        self.track_map_list = {}
        self.track_map_list[0] = {}
        self.track_vel_list = {}
        self.track_vel_list[0] = {}

        self.agent_lane = 0
        if not self.test_mode:
            loc = np.random.randint(0, len(self.lane_map_list[self.agent_lane]))
            self.lane_map_list[self.agent_lane][loc][self.lab2ind['agent']] = 1
        else:
            self.lane_map_list[self.agent_lane][0][self.lab2ind["agent"]] = 1

        self.agent_id = None
        self.num_cars_in_setup = len(self.lane_map_list[0])

        del self.lane_map

        for lane in range(0, 1):
            for idx in range(0, len(self.lane_map_list[lane])):
                veh_info = self.lane_map_list[lane][idx]
                veh_id = veh_info[self.lab2ind["id"]]
                veh_angle = veh_info[self.lab2ind["angle"]]
                veh_vel = veh_info[self.lab2ind["vel"]]
                if veh_info[self.lab2ind["agent"]] == 1:
                    self.agent_id = veh_id
                self.track_map_list[lane][veh_id] = veh_angle
                self.track_vel_list[lane][veh_id] = veh_vel

        self.get_occupancy_grid()

        # Create Age Vectors
        self.agent_age = -1 * np.ones(self.num_cols)
        self.true_age = -1 * np.ones(self.num_cols)

        agent_age_obs = self.agevec2obs(self.agent_age, init_t = True)
        self.agevec2obs(self.true_age, init_t=True)

        #print(agent_age_obs)
        #print(self.agent_age)
        #print(self.true_age)


        # Set traffic Lights
        if self.enable_traffic_light:
            self.set_red_light = False
            if not self.test_mode:
                self.genObj.reset()
                self.genPts = self.genObj.gen()
                # print(self.genPts)
                self.expanded_pts = self.expandPts(self.genPts)
            else:
                with open(self.test_mode_trajec_file, "rb") as handle:
                    self.expanded_pts = pickle.load(handle)
            

            #print(self.expanded_pts)

        self.occ_track = self.occ_grid.copy()
        self.vel_track = self.vel_grid.copy()

        if self.render:
            self.draw_graphics(0.0, None, None)

            if self.enable_frame_capture:
                self.curr_screen = pygame.surfarray.array3d(self.screen)
                self.curr_screen = np.flip(self.curr_screen, axis=0)
                self.curr_screen = np.rot90(self.curr_screen, k=-1)[:, :570]
                self.curr_screen = cv2.resize(self.curr_screen, dsize=(285, 250), interpolation=cv2.INTER_CUBIC)


        res = np.concatenate((self.occ_grid[0], self.vel_grid[0], agent_age_obs)).copy()
        #print("Occ : ", self.occ_grid)
        #print("Vel : ", self.vel_grid)
        # print("----------------------")
        #agent_vel = self.lane_map_list[self.agent_lane][0][self.lab2ind["vel"]]
        #agent_vel = np.array(agent_vel).reshape(1,)

        #res = np.append(self.occ_grid, agent_vel)
        #print(self.genPts)
        #print(self.random_trajec)

        return res.copy()

    def is_valid(self, lane, vech_list, angle, agent_index):

        if (agent_index < (len(vech_list[lane])-1)):
            delta = vech_list[lane][agent_index +
                                    1][self.lab2ind['angle']] - angle
        else:
            delta = vech_list[lane][0][self.lab2ind['angle']] - angle
            delta = delta % 360

        if (delta > self.delta_theta[lane]):
            return True
        else:
            return False

    def draw_occupancy(self):

        agent_lane = self.agent_lane
        agent_cell = None

        lane_boundries = [[248, 218]]
        agent_cell = [i for i, tup in enumerate(
            self.lane_map_list[self.agent_lane]) if tup[self.lab2ind["agent"]] == 1][0]

        all_lanes = [0]
        lane_mapping = {0: 0}

        lane_deltas = [np.rad2deg(
            (self.cell_size*self.x_pixel_one_metre)/self.lane_radius[0])]
        # Forward Grids

        for lane in all_lanes:

            forward_angle_curr_lane = self.lane_map_list[agent_lane][agent_cell][self.lab2ind['angle']]

            for i in range(0, int(len(self.occ_grid[0])/2)):

                old_forward_angle_curr_lane = forward_angle_curr_lane
                forward_angle_curr_lane += lane_deltas[lane]

                cx, cy = self.get_coordinates(
                    old_forward_angle_curr_lane, lane_boundries[lane][0])
                p = []

                points = list(np.linspace(old_forward_angle_curr_lane,
                                          forward_angle_curr_lane, self.polygon_points))
                for a in points:
                    x, y = self.get_coordinates(a, lane_boundries[lane][0])
                    p.append((x, y))

                points.reverse()

                for a in points:
                    x, y = self.get_coordinates(a, lane_boundries[lane][1])
                    p.append((x, y))
                p.append((cx, cy))

                array_offset = int((len(self.occ_grid[0])/2))

                if (self.occ_grid[lane_mapping[lane]][array_offset+i] == 1):
                    pygame.draw.polygon(self.screen, (0, 0, 0), p, 0)
                elif (self.occ_grid[lane_mapping[lane]][array_offset+i] == 2):
                    pygame.draw.polygon(self.screen, self.dark_green, p, 0)
                elif (self.occ_grid[lane_mapping[lane]][array_offset+i] == -1):
                    pygame.draw.polygon(self.screen, self.color_red, p, 0)
                else:
                    pygame.draw.polygon(self.screen, self.color_white, p, 1)

        # Backward Grids
        for lane in all_lanes:

            forward_angle_curr_lane = self.lane_map_list[agent_lane][agent_cell][self.lab2ind['angle']]

            for i in range(0, int(len(self.occ_grid[0])/2)):

                old_forward_angle_curr_lane = forward_angle_curr_lane
                forward_angle_curr_lane -= lane_deltas[lane]

                cx, cy = self.get_coordinates(
                    old_forward_angle_curr_lane, lane_boundries[lane][0])
                p = []

                points = list(np.linspace(old_forward_angle_curr_lane,
                                          forward_angle_curr_lane, self.polygon_points))

                for a in points:
                    x, y = self.get_coordinates(a, lane_boundries[lane][0])
                    p.append((x, y))

                points.reverse()

                for a in points:
                    x, y = self.get_coordinates(a, lane_boundries[lane][1])
                    p.append((x, y))
                p.append((cx, cy))

                half_len = int(len(self.occ_grid[0])/2)
                if (self.occ_grid[lane_mapping[lane]][half_len-1-i] == 1):
                    pygame.draw.polygon(self.screen, (0, 0, 0), p, 0)
                elif (self.occ_grid[lane_mapping[lane]][half_len-1-i] == 2):
                    pygame.draw.polygon(self.screen, self.dark_green, p, 0)
                elif (self.occ_grid[lane_mapping[lane]][half_len-1-i] == -1):
                    pygame.draw.polygon(self.screen, self.color_red, p, 0)
                else:
                    pygame.draw.polygon(self.screen, self.color_white, p, 1)

    def get_occupancy_grid(self):

        self.occ_grid = np.zeros((1, self.num_cols), dtype=np.int32)
        self.vel_grid = np.zeros((1, self.num_cols), dtype=np.float32)

        agent_lane = self.agent_lane
        agent_cell = None

        self.lane_map_list[0] = sorted(
            self.lane_map_list[0], key=lambda x: x[self.lab2ind['angle']])

        agent_cell = [i for i, tup in enumerate(
            self.lane_map_list[agent_lane]) if tup[self.lab2ind["agent"]] == 1][0]

        all_lanes = [0]
        lane_mapping = {0: 0}

        view_size_theta = [np.rad2deg(
            (self.view_size*self.x_pixel_one_metre)/self.lane_radius[0])]
        agent_angle = self.lane_map_list[agent_lane][agent_cell][self.lab2ind['angle']]

        # Forward
        for lane in all_lanes:

            shift = np.rad2deg(self.car_length/2 *
                               self.x_pixel_one_metre)/self.lane_radius[lane]
            forward_done = False
            _next_ = None

            for j in range(0, len(self.lane_map_list[lane])):
                if (self.lane_map_list[lane][j][self.lab2ind['angle']] - shift > agent_angle):
                    _next_ = j
                    break

            if (_next_ == None):
                _next_ = 0

            count_forward = 0
            copy_next = _next_

            for x in range(0, len(self.lane_map_list[lane])):
                next_car_angle = self.lane_map_list[lane][copy_next][self.lab2ind['angle']]

                if (next_car_angle <= agent_angle):
                    next_car_angle += 360

                angle_diff = next_car_angle - agent_angle - shift

                '''
                if lane == 1:
                    print(angle_diff, self.time_elapsed)
                    print(self.lane_map[1], agent_angle, next_car_angle)
                '''
                angle_diff = (angle_diff) % 360

                if(angle_diff <= view_size_theta[lane]):
                    count_forward += 1
                    copy_next += 1
                    if (copy_next > (len(self.lane_map_list[lane])-1)):
                        copy_next = 0
                else:
                    break

            for x in range(0, count_forward):
                # print('Innside qw')
                next_car_angle = self.lane_map_list[lane][_next_][self.lab2ind['angle']]
                next_car_vel = self.lane_map_list[lane][_next_][self.lab2ind['vel']]

                if (next_car_angle < agent_angle):
                    next_car_angle += 360

                angle_diff = next_car_angle - agent_angle - shift

                angle_diff = (angle_diff) % 360

                if (angle_diff <= view_size_theta[lane]):

                    distance_covered = np.deg2rad(
                        angle_diff)*self.lane_radius[lane]
                    distance_covered_metres = (
                        1/self.x_pixel_one_metre)*distance_covered
                    index = distance_covered_metres/float(self.cell_size)

                    half_len = int(len(self.occ_grid[0])/2)
                    num_indexs = math.ceil(
                        (self.car_length*0.5)/self.cell_size)
                    self.occ_grid[lane_mapping[lane]
                                  ][half_len+int(index)] = 1
                    self.vel_grid[lane_mapping[lane]
                                  ][half_len+int(index)] = next_car_vel

                    if ((_next_+1) > (len(self.lane_map_list[lane])-1)):
                        _next_ = 0
                    else:
                        _next_ += 1

        num_indexs_others = math.ceil((self.car_length)/self.cell_size)
        half = int(len(self.occ_grid[0])/2) - 1
        tmp_copy = np.copy(self.occ_grid)

        for lane in range(0, 1):

            for i in range(half, self.occ_grid[lane].shape[0]):

                if(tmp_copy[lane][i] == 1):
                    self.occ_grid[lane][i+1:i + num_indexs_others + 1] = 1
                    val = self.vel_grid[lane][i]
                    self.vel_grid[lane][i+1:i + num_indexs_others + 1] = val

        tmp = {}
        tmp[0] = sorted(self.lane_map_list[0],
                        key=lambda x: x[self.lab2ind['angle']], reverse=True)

        for lane in all_lanes:

            backward_done = False
            _prev_ = None

            for j in range(0, len(tmp[lane])):
                if (tmp[lane][j][self.lab2ind['angle']] + shift < agent_angle):
                    _prev_ = [i for i, tup in enumerate(
                        self.lane_map_list[lane]) if tup[self.lab2ind['angle']] == tmp[lane][j][self.lab2ind['angle']]][0]
                    break

            if (_prev_ == None):
                _prev_ = len(self.lane_map_list[lane])-1

            count = 0
            copy_prev = _prev_

            for x in range(0, len(self.lane_map_list[lane])):

                prev_car_angle = self.lane_map_list[lane][copy_prev]['angle'] + shift
                #prev_car_angle += np.rad2deg( (self.car_length/2 * self.x_pixel_one_metre)/ self.lane_radius[lane])
                angle_diff = (agent_angle - prev_car_angle)

                angle_diff = (angle_diff % 360)
                if (angle_diff <= view_size_theta[lane]):
                    count += 1
                    copy_prev -= 1
                    if (copy_prev < 0):
                        copy_prev = len(self.lane_map_list[lane])-1
                else:
                    break

            for x in range(0, count):
                #print('Inside 128')
                prev_car_angle = self.lane_map_list[lane][_prev_]['angle'] + shift
                #prev_car_angle += np.rad2deg( (self.car_length/2 * self.x_pixel_one_metre)/ self.lane_radius[lane])
                prev_car_vel = self.lane_map_list[lane][_prev_]['vel']

                angle_diff = (agent_angle - prev_car_angle)
                angle_diff = (angle_diff % 360)

                if (angle_diff <= view_size_theta[lane]):
                    distance_covered = np.deg2rad(
                        angle_diff)*self.lane_radius[lane]
                    distance_covered_metres = (
                        1/self.x_pixel_one_metre)*distance_covered
                    index = distance_covered_metres/float(self.cell_size)

                    #num_indexs = math.ceil((self.car_length)/self.cell_size)
                    half_len = int(len(self.occ_grid[0])/2)
                    self.occ_grid[lane_mapping[lane]
                                  ][half_len-1 - int(index)] = 1
                    self.vel_grid[lane_mapping[lane]][half_len -
                                                      1 - int(index)] = prev_car_vel
                    _prev_ -= 1

        tmp_copy = np.copy(self.occ_grid)
        half_len = int(len(self.occ_grid[0])/2)
        num_indexs = math.ceil((self.car_length*0.5)/self.cell_size)
        num_indexs_others = math.ceil((self.car_length)/self.cell_size)

        for lane in range(0, 1):
            for i in range(0, half_len):

                if (tmp_copy[lane][i] == 1):

                    if(i - num_indexs_others <= 0):
                        self.occ_grid[lane][0:i] = 1
                        val = self.vel_grid[lane][i]
                        self.vel_grid[lane][0:i] = val
                    else:
                        self.occ_grid[lane][i - num_indexs_others: i] = 1
                        val = self.vel_grid[lane][i]
                        self.vel_grid[lane][i - num_indexs_others: i] = val

        # 2 is used to identify the agent in occupancy grid. Can't do this in for loop.
        half_len = int(len(self.occ_grid[0])/2)
        num_indexs = math.ceil((self.car_length*0.5)/self.cell_size)
        self.occ_grid[lane_mapping[agent_lane]][half_len-1] = 2
        self.occ_grid[lane_mapping[agent_lane]
                      ][half_len-1+1:half_len-1+num_indexs+1] = 2
        self.occ_grid[lane_mapping[agent_lane]
                      ][half_len-1-num_indexs:half_len-1] = 2
        self.vel_grid[lane_mapping[agent_lane]][half_len -
                                                1] = self.lane_map_list[self.agent_lane][agent_cell][self.lab2ind["vel"]]
        self.vel_grid[lane_mapping[agent_lane]][half_len-1+1: half_len-1 +
                                                num_indexs+1] = self.lane_map_list[self.agent_lane][agent_cell][self.lab2ind["vel"]]
        self.vel_grid[lane_mapping[agent_lane]][half_len-1-num_indexs:half_len -
                                                1] = self.lane_map_list[self.agent_lane][agent_cell][self.lab2ind["vel"]]

        low_range = agent_angle - shift
        high_range = agent_angle + shift
        other_lanes = [0]
        other_lanes.pop(agent_lane)
        for other_lane in other_lanes:
            for car in range(len(self.lane_map_list[other_lane])):

                if (self.lane_map_list[other_lane][car][self.lab2ind['angle']] >= low_range and self.lane_map_list[other_lane][car][self.lab2ind['angle']] <= high_range):
                    view_angle = agent_angle - view_size_theta[other_lane]
                    angle_diff = self.lane_map_list[other_lane][car][self.lab2ind['angle']] - view_angle
                    angle_diff = angle_diff % 360

                    distance_covered = np.deg2rad(
                        angle_diff) * self.lane_radius[other_lane]
                    distance_covered_metres = (
                        1/self.x_pixel_one_metre) * distance_covered
                    index = distance_covered_metres / float(self.cell_size)
                    self.occ_grid[other_lane][int(index)] = 1
                    self.vel_grid[other_lane][int(
                        index)] = self.lane_map_list[other_lane][car][self.lab2ind["vel"]]
                    self.occ_grid[other_lane][int(
                        index)+1:int(index)+num_indexs+1] = 1
                    self.occ_grid[other_lane][int(
                        index)-num_indexs:int(index)] = 1
                    val = self.vel_grid[other_lane][int(index)]
                    self.vel_grid[other_lane][int(
                        index)+1:int(index)+num_indexs+1] = val
                    self.vel_grid[other_lane][int(
                        index)-num_indexs:int(index)] = val

        '''
        if (self.comm_mode):
            tmp_copy_2 = np.copy(self.occ_grid)
            vel_copy = np.copy(self.vel_grid)
            for key in self.region_maping:
                for col in self.region_maping[key]:
                    self.occ_grid[:, col] = -1
                    self.vel_grid[:, col] = -1

            if (query_action != "NULL"):
                for col in self.region_maping[query_action]:
                    self.occ_grid[:, col] = tmp_copy_2[:, col]
                    self.vel_grid[:, col] = vel_copy[:, col]
        '''

    def getAllowedLanetoChange(self):

        if self.agent_lane == 0:
            return [1]
        elif self.agent_lane == 2:
            return [1]
        else:
            return [0, 2]

    def changeLane(self, lane):

        half = int(len(self.occ_grid[0])/2) - 1
        num_indexs_others = int(math.ceil((self.car_length)/self.cell_size)/2)
        forward_sum = self.occ_grid[self.agent_lane][half +
                                                     num_indexs_others + 1:].sum()

        if forward_sum == 0:
            if len(self.lane_map_list[lane]) > 2:

                agent_idx = [i for i, tup in enumerate(
                    self.lane_map_list[self.agent_lane]) if tup[self.lab2ind["agent"]] == 1][0]
                agent_angle = self.lane_map_list[self.agent_lane][agent_idx][self.lab2ind["angle"]]

                index = None
                for i in range(0, len(self.lane_map_list[lane])):

                    if self.lane_map_list[lane][i][self.lab2ind["angle"]] > agent_angle:
                        index = i
                        break

                if index == None:
                    index = 0

                other_car_angle = self.lane_map_list[lane][index][self.lab2ind["angle"]]

                angle_diff = other_car_angle - agent_angle
                if angle_diff < 0.0:
                    angle_diff = (other_car_angle + 360) - agent_angle

                angle_diff = angle_diff % 360

                if (angle_diff > self.view_size_theta[self.agent_lane] + 10):

                    #forward_min_theta = np.rad2deg(((((self.s_not))*self.x_pixel_one_metre)/self.lane_radius[self.agent_lane]))
                    forward_min_theta = 10
                    index_forward = None

                    for j in range(0, len(self.lane_map_list[self.agent_lane])):
                        if (self.lane_map_list[self.agent_lane][j][self.lab2ind["angle"]] > other_car_angle):
                            index_forward = j
                            break

                    if (index_forward == None):
                        index_forward = 0

                    reverse_map = self.lane_map_list[self.agent_lane][::-1]
                    # print(reverse_map)
                    #print("Other Car angle : ", other_car_angle)
                    #print("Front", self.lane_map_list[self.agent_lane][index_forward])

                    index_prev = None
                    for j in range(0, len(reverse_map)):
                        if (reverse_map[j][self.lab2ind["angle"]] < other_car_angle):
                            index_prev = j
                            # print("Inside")
                            break

                    if index_prev == None:
                        index_prev = len(reverse_map) - 1

                    forward_diff = self.lane_map_list[self.agent_lane][
                        index_forward][self.lab2ind["angle"]] - other_car_angle

                    if forward_diff < 0.0:
                        forward_diff = (
                            self.lane_map_list[self.agent_lane][index_forward][self.lab2ind["angle"]] + 360) - other_car_angle

                    forward_diff = forward_diff % 360

                    back_diff = other_car_angle - \
                        reverse_map[index_prev][self.lab2ind["angle"]]

                    if back_diff < 0.0:
                        back_diff = (other_car_angle + 360) - \
                            reverse_map[index_prev][self.lab2ind["angle"]]

                    back_diff = back_diff % 360
                    # print(back_diff)

                    res1 = forward_diff > forward_min_theta
                    res2 = back_diff > forward_min_theta

                    if res1 and res2:
                        copy_lane_map = copy.deepcopy(self.lane_map_list)
                        veh = copy_lane_map[lane].pop(index)
                        veh[self.lab2ind["lane"]] = self.agent_lane
                        copy_lane_map[self.agent_lane].insert(
                            len(copy_lane_map[self.agent_lane]), veh)
                        self.lane_map_list = copy.deepcopy(copy_lane_map)
                '''
                if index != None:

                    forward_min_theta = np.rad2deg(((((self.s_not))*self.x_pixel_one_metre)/self.lane_radius[self.agent_lane]))
                    
                    index_forward = None

                    for j in range(0, len(self.lane_map_list[self.agent_lane])):
                        if (self.lane_map_list[self.agent_lane][j][self.lab2ind["angle"]] > self.lane_map_list[lane][i][self.lab2ind["angle"]]):
                            index_forward = j
                            break

                    if (index_forward == None):
                        index_forward = 0

                    forward_theta = self.lane_map_list[self.agent_lane][index_forward][self.lab2ind['angle']] - self.lane_map_list[lane][i][self.lab2ind["angle"]]
                    forward_theta %= 360

                    res2 = forward_theta > forward_min_theta
                    res1 = math.fabs(self.lane_map_list[lane][i][self.lab2ind["angle"]] - agent_angle) > self.view_size_theta[self.agent_lane]

                    if res2 and res1:
                        copy_lane_map = copy.deepcopy(self.lane_map_list)
                        veh = copy_lane_map[lane].pop(index)
                        veh[self.lab2ind["lane"]] = self.agent_lane
                        copy_lane_map[self.agent_lane].insert(len(copy_lane_map[self.agent_lane]), veh)
                        self.lane_map_list = copy.deepcopy(copy_lane_map)
                        '''

    def calculate_reward(self):

        agent_idx = [i for i, tup in enumerate(
            self.lane_map_list[self.agent_lane]) if tup[self.lab2ind["agent"]] == 1][0]
        rew = (self.lane_map_list[self.agent_lane]
               [agent_idx][self.lab2ind["vel"]])/self.max_vel
        return rew

    def occChangeDetector(self, old_occ, new_occ, prev_vel, new_vel, change_store):

        for col in range(0, old_occ.shape[1]):

            if ((new_occ[0][col] != old_occ[0][col]) or (new_vel[0][col]!=prev_vel[0][col])):
                change_store[0][col] = 1

    def toggleTrafficLight(self):
        self.set_red_light = not self.set_red_light

    def step(self, rec):

        self.num_steps += 1

        prev_occ = self.occ_grid.copy()
        prev_vel = self.vel_grid.copy()

        #print(self.expanded_pts, self.num_steps)
        if self.enable_traffic_light:
            if self.num_steps in self.expanded_pts:
                self.toggleTrafficLight()
                #print("Toggled Traffic Light , Red : ", self.set_red_light)

        '''

        #print(self.num_steps)
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.toggleTrafficLight()
                    print("Done", self.set_red_light)
        '''
        action, query = self.action_decoder(rec)
        self.decoded_action = action
        self.decoded_query = query

        self.lane_map_list[0] = sorted(
            self.lane_map_list[0], key=lambda x: x[self.lab2ind['angle']])

        reward = 0.0
        game_over = False

        agent_idx = [i for i, tup in enumerate(
            self.lane_map_list[self.agent_lane]) if tup[self.lab2ind["agent"]] == 1][0]

        agent_vel = self.lane_map_list[self.agent_lane][agent_idx][self.lab2ind['vel']]
        agent_angle = self.lane_map_list[self.agent_lane][agent_idx][self.lab2ind['angle']]
        prev_agent_angle = copy.deepcopy(agent_angle)

        if self.external_controller:
            if (action == 0):

                if (agent_vel >= self.v_not):
                    acc = 0
                else:
                    acc = self.a

                agent_distance_travelled_metre = self.cal_dist_travelled(
                    agent_vel, acc, self.time_period)
                agent_distance_travelled_pixles = self.x_pixel_one_metre * \
                    agent_distance_travelled_metre
                # print('action 0')
                agent_delta_theta = self.get_theta(
                    self.lane_radius[self.agent_lane], agent_distance_travelled_pixles, 0)
                new_agent_theta = agent_angle + agent_delta_theta
                res = self.is_valid(
                    self.agent_lane, self.lane_map_list, agent_angle, agent_idx)

                if res:
                    self.lane_map_list[self.agent_lane][agent_idx][self.lab2ind['vel']] = self.cal_new_velocity(
                        agent_vel, acc, self.time_period)
                    self.lane_map_list[self.agent_lane][agent_idx][self.lab2ind['angle']
                                                                   ] = new_agent_theta
                    #reward = self.lane_map_list[self.agent_lane][agent_idx][self.lab2ind['vel']]
                    #reward = (self.lane_map_list[self.agent_lane][agent_idx][self.lab2ind['vel']] - 0)/(self.v_not - 0)
                else:
                    #reward = self.collision_cost
                    #reward = self.lane_map_list[self.agent_lane][agent_idx][self.lab2ind['vel']]
                    game_over = True
                    self.lane_map_list[self.agent_lane][agent_idx][self.lab2ind['vel']] = 0.0
                    self.lane_map_list[self.agent_lane][agent_idx][self.lab2ind['angle']] = agent_angle

            elif (action == 1):

                if (agent_vel <= 0.0):
                    acc = 0
                else:
                    acc = -1*self.b

                agent_distance_travelled_metre = self.cal_dist_travelled(
                    agent_vel, acc, self.time_period)

                if agent_distance_travelled_metre < 0:
                    rounded_agent_dist_travelled_metre = round(
                        agent_distance_travelled_metre, 1)
                    error = max(rounded_agent_dist_travelled_metre, agent_distance_travelled_metre) - \
                        min(rounded_agent_dist_travelled_metre,
                            agent_distance_travelled_metre)
                    # print('Error : ',error)
                    agent_distance_travelled_metre = 0.0
                    # print(agent_distance_travelled_metre)

                agent_distance_travelled_pixles = self.x_pixel_one_metre * \
                    agent_distance_travelled_metre
                agent_delta_theta = self.get_theta(
                    self.lane_radius[self.agent_lane], agent_distance_travelled_pixles, 1)
                new_agent_theta = agent_angle + agent_delta_theta
                res = self.is_valid(
                    self.agent_lane, self.lane_map_list, agent_angle, agent_idx)
                new_agent_vel = self.cal_new_velocity(
                    agent_vel, acc, self.time_period)

                if new_agent_vel < 0:
                    new_agent_vel = 0

                if res:
                    #reward = agent_distance_travelled_metre
                    self.lane_map_list[self.agent_lane][agent_idx][self.lab2ind['vel']
                                                                   ] = new_agent_vel
                    self.lane_map_list[self.agent_lane][agent_idx][self.lab2ind['angle']
                                                                   ] = new_agent_theta
                    #reward = (self.lane_map_list[self.agent_lane][agent_idx][self.lab2ind['vel']] - 0)/(self.v_not - 0)
                    #reward = self.lane_map_list[self.agent_lane][agent_idx][self.lab2ind['vel']]

                else:
                    #reward = self.collision_cost
                    #reward = self.lane_map_list[self.agent_lane][agent_idx][self.lab2ind['vel']]
                    game_over = True
                    self.lane_map_list[self.agent_lane][agent_idx][self.lab2ind['vel']] = 0.0
                    self.lane_map_list[self.agent_lane][agent_idx][self.lab2ind['angle']] = agent_angle

            elif (action == 2):

                agent_distance_travelled_metre = self.cal_dist_travelled(
                    agent_vel, 0, self.time_period)
                agent_distance_travelled_pixles = self.x_pixel_one_metre * \
                    agent_distance_travelled_metre
                # print('action 2')
                agent_delta_theta = self.get_theta(
                    self.lane_radius[self.agent_lane], agent_distance_travelled_pixles, 2)
                new_agent_theta = agent_angle + agent_delta_theta
                res = self.is_valid(
                    self.agent_lane, self.lane_map_list, agent_angle, agent_idx)
                new_agent_vel = self.cal_new_velocity(
                    agent_vel, 0, self.time_period)

                if res:
                    #reward = agent_distance_travelled_metre
                    self.lane_map_list[self.agent_lane][agent_idx][self.lab2ind['vel']
                                                                   ] = new_agent_vel
                    self.lane_map_list[self.agent_lane][agent_idx][self.lab2ind['angle']
                                                                   ] = new_agent_theta
                    #reward = (self.lane_map_list[self.agent_lane][agent_idx][self.lab2ind['vel']] - 0)/(self.v_not - 0)
                    #reward = self.lane_map_list[self.agent_lane][agent_idx][self.lab2ind['vel']]
                else:
                    #reward = self.collision_cost
                    #reward = self.lane_map_list[self.agent_lane][agent_idx][self.lab2ind['vel']]
                    game_over = True
                    self.lane_map_list[self.agent_lane][agent_idx][self.lab2ind['vel']] = 0.0
                    self.lane_map_list[self.agent_lane][agent_idx][self.lab2ind['angle']] = agent_angle
        else:
            pass

        agent_idx = [i for i, tup in enumerate(
            self.lane_map_list[self.agent_lane]) if tup[self.lab2ind["agent"]] == 1][0]
        lastest_agent_angle = self.lane_map_list[self.agent_lane][agent_idx][self.lab2ind["angle"]]

        diff = lastest_agent_angle - prev_agent_angle
        if diff < 0:
            diff = (diff % 360) + 360

        self.agent_id = self.lane_map_list[self.agent_lane][agent_idx][self.lab2ind["id"]]
        self.track_map_list[self.agent_lane][self.agent_id] += diff

        if self.enable_traffic_light:
            if self.set_red_light:
                self.lane_map_list[self.agent_lane].append(
                    self.create_dummy_traffic_tuple(self.agent_lane))

        self.lane_map_list[0] = sorted(
            self.lane_map_list[0], key=lambda x: x[self.lab2ind['angle']])

        theta_diff = 0.0
        delta_v_alpha = 0.0
        v_alpha = 0.0
        car_gap_in_metre = 0.0
        s_alpha = 0.0
        idm_acc = 0.0
        dist_travelled_in_metre = 0.0
        rounded_dist_travelled_in_metre = 0.0
        error = 0.0
        new_velocity = 0.0
        dist_travelled_in_pixels = 0.0
        new_theta = 0.0
        self.new_velocity_map = {0: []}
        self.new_theta_map = {0: []}

        for lane in range(0, 1):
            for i in range(0, len(self.lane_map_list[lane])):

                check_value = None
                if self.external_controller:
                    check_value = 1
                else:
                    check_value = 3

                light_check_value = None
                if self.enable_traffic_light:
                    if self.set_red_light:
                        light_check_value = 2
                    else:
                        light_check_value = 3
                else:
                    light_check_value = 3

                if not (self.lane_map_list[lane][i][self.lab2ind['agent']] == check_value or self.lane_map_list[lane][i][self.lab2ind['agent']] == light_check_value):

                    if (i < (len(self.lane_map_list[lane]) - 1)):
                        theta_diff = self.lane_map_list[lane][i+1][self.lab2ind['angle']
                                                                   ] - self.lane_map_list[lane][i][self.lab2ind['angle']]
                        delta_v_alpha = self.lane_map_list[lane][i][self.lab2ind['vel']
                                                                    ] - self.lane_map_list[lane][i+1][self.lab2ind['vel']]
                        v_alpha = self.lane_map_list[lane][i][self.lab2ind['vel']]
                    else:
                        theta_diff = self.lane_map_list[lane][0][self.lab2ind['angle']
                                                                 ] - self.lane_map_list[lane][i][self.lab2ind['angle']]
                        theta_diff = theta_diff % 360
                        delta_v_alpha = self.lane_map_list[lane][i][self.lab2ind['vel']
                                                                    ] - self.lane_map_list[lane][0][self.lab2ind['vel']]
                        v_alpha = self.lane_map_list[lane][i][self.lab2ind['vel']]

                    car_gap_in_metre = self.return_arc_length(
                        self.lane_radius[lane], theta_diff)
                    s_alpha = car_gap_in_metre - self.car_length
                    idm_acc = self.cal_acc(
                        s_alpha, delta_v_alpha, v_alpha, self.other_max_vel)
                    dist_travelled_in_metre = self.cal_dist_travelled(
                        v_alpha, idm_acc, self.time_period)

                    if dist_travelled_in_metre < 0:
                        rounded_dist_travelled_in_metre = round(
                            dist_travelled_in_metre, 1)
                        error = max(rounded_dist_travelled_in_metre, dist_travelled_in_metre) - min(
                            rounded_dist_travelled_in_metre, dist_travelled_in_metre)
                        # print('Error in Calculation : ',error)
                        rounded_dist_travelled_in_metre = 0.0
                        dist_travelled_in_metre = rounded_dist_travelled_in_metre

                    new_velocity = self.cal_new_velocity(
                        v_alpha, idm_acc, self.time_period)
                    if new_velocity < 0:
                        new_velocity = 0

                    self.new_velocity_map[lane].append(new_velocity)

                    dist_travelled_in_pixels = self.x_pixel_one_metre*dist_travelled_in_metre
                    new_theta = self.get_theta(
                        self.lane_radius[lane], dist_travelled_in_pixels, 'idm')
                    self.new_theta_map[lane].append(new_theta)
                else:
                    # print(self.lane_map_list[lane][i])
                    self.new_velocity_map[self.agent_lane].append(
                        self.lane_map_list[self.agent_lane][i][self.lab2ind['vel']])
                    self.new_theta_map[self.agent_lane].append(0)

        for lane in range(0, 1):
            for i in range(0, len(self.new_velocity_map[lane])):
                self.lane_map_list[lane][i][self.lab2ind['vel']
                                            ] = self.new_velocity_map[lane][i]
                self.lane_map_list[lane][i][self.lab2ind['angle']
                                            ] += self.new_theta_map[lane][i]
                veh_info = self.lane_map_list[lane][i]
                veh_id = veh_info[self.lab2ind["id"]]
                if veh_id < 0:
                    pass
                else:
                    self.track_map_list[lane][veh_id] += self.new_theta_map[lane][i]

        self.copy_lane_map_list = copy.deepcopy(self.lane_map_list)

        for lane in range(0, 1):
            for i in range(0, len(self.lane_map_list[lane])):

                veh_id = self.lane_map_list[lane][i][self.lab2ind["id"]]
                veh_vel = self.lane_map_list[lane][i][self.lab2ind["vel"]]
                self.track_vel_list[lane][veh_id] = veh_vel

                if self.lane_map_list[lane][i][self.lab2ind['angle']] > 360:
                    # print("yes")
                    self.lane_map_list[lane][i][self.lab2ind['angle']] %= 360

        # print(self.lane_map_list[0])

        if self.enable_traffic_light:
            if self.set_red_light:
                light_index = [i for i, tup in enumerate(
                    self.lane_map_list[self.agent_lane]) if tup[self.lab2ind["agent"]] == 2][0]
                self.lane_map_list[self.agent_lane].pop(light_index)

        self.get_occupancy_grid()

        self.detect_change = np.zeros(self.occ_grid.shape)
        self.occChangeDetector(prev_occ, self.occ_grid, prev_vel, self.vel_grid, self.detect_change)

        # ---- Update True Age ---- #
        for col in range(0, prev_occ.shape[1]):
            if self.detect_change[0][col] == 1.0:
                self.true_age[col] = 0.0
            else:
                self.true_age[col] = min(1, self.true_age[col] + self.age_time_step)
        # ---- Update True Age ---- #

        # ---- Update Agent Age ---- #
        for col in range(0, self.agent_age.shape[0]):
            self.agent_age[col] = min(1, self.agent_age[col] + self.age_time_step)

        if query == "NULL":
            pass
        else:
            cols = self.region_maping[query]
            for col in cols:
                self.agent_age[col] = self.true_age[col]
        # ---- Update Agent Age ---- #

        self.time_elapsed += self.time_period

        self.lane_map_list[0] = sorted(
            self.lane_map_list[0], key=lambda x: x[self.lab2ind['angle']])

        comm_obs = []
        for key in self.region_maping.keys():
            for col in self.region_maping[key]:
                comm_obs.append(self.agent_age[col])

        comm_obs = np.array(comm_obs)
        
        # Copy local view first
        for index in self.local_indexes:
            self.occ_track[0][index] = self.occ_grid[0][index]
            self.vel_track[0][index] = self.vel_grid[0][index]

        if query == "NULL":
            pass
        else:
            cols = self.region_maping[query]
            for col in cols:
                self.occ_track[0][col] = self.occ_grid[0][col]
                self.vel_track[0][col] = self.vel_grid[0][col]

        reward = self.calculate_reward()

        self.before_comm_reward = reward

        if self.comm_mode:
            if (query == "NULL"):
                reward += 0.1

        res = np.concatenate((self.occ_track[0], self.vel_track[0], comm_obs))
        #print("Occ : ", self.occ_grid)
        #print("Vel : ", self.vel_grid)
        # print("----------------------")
        #agent_idx = [i for i, tup in enumerate(self.lane_map_list[self.agent_lane]) if tup[self.lab2ind["agent"]] == 1][0]
        #agent_vel = self.lane_map_list[self.agent_lane][agent_idx][self.lab2ind["vel"]]
        #agent_vel = agent_vel.reshape(1, )

        if self.render:
            self.draw_graphics(reward, action, query)

            if self.enable_frame_capture:
                self.curr_screen = pygame.surfarray.array3d(self.screen)
                self.curr_screen = np.flip(self.curr_screen, axis=0)
                self.curr_screen = np.rot90(self.curr_screen, k=-1)[:, :570]
                self.curr_screen = cv2.resize(self.curr_screen, dsize=(285, 250), interpolation=cv2.INTER_CUBIC)
                

        #res = np.append(self.occ_grid.flatten(), agent_vel)
        #res = np.array((self.occ_grid, self.vel_grid)).flatten()
        '''
        if game_over:
            res[-1] = -1 # this is done to denote terminal state
        '''
        return (res.copy(), reward, game_over, {})

    def destroy_window(self):
        if self.render:
            pygame.quit()


if __name__ == "__main__":

    with open("SimulatorCode/ppo-sim-config.yaml") as handle:
        sim_config = yaml.load(handle)

    trajec_path = "/SimulatorCode/micro.pkl"
    sim_config["config"]["trajec-file-path"] = os.getcwd() + trajec_path

    env = TrafficSim(sim_config["config"])

    print(env.action_map)
    print(env.observation_space)
    print(env.action_space.n)

    # env.reset()
    for i in range(0, 1):
        state = env.reset(random.sample(self.densities))
        print(state)
        done = False
        total_reward = 0.0

        while True:

            #state, reward, done, _ = env.step(np.random.randint(0, 3))
            state, reward, done, _ = env.step(2)
            total_reward += reward
            # print(state)
            # print(reward)
            if done:
                # print("Inside")
                print(env.agent_id)
                print(env.track_map_list[env.agent_lane])
                break
                # pass

        # print(total_reward)
