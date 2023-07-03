import traci
import numpy as np
import random
import timeit
import os

# 根據 environment.net.xml 定義的相位編碼
NS_GREEN = 0
NS_YELLOW = 1
NSL_GREEN = 2
NSL_YELLOW = 3
EW_GREEN = 4
EW_YELLOW = 5
EWL_GREEN = 6
EWL_YELLOW = 7


class Simulation:

    def __init__(self, Model, Buffer, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, num_states, num_actions, training_epochs):
        self.Model = Model
        self.Buffer = Buffer
        self.TrafficGen = TrafficGen
        self.gamma = gamma
        self.step = 0
        self.sumo_cmd = sumo_cmd
        self.max_step = max_steps
        self.green_duration = green_duration
        self.yellow_duration = yellow_duration
        self.num_states = num_states
        self.num_actions = num_actions
        self._reward_store = []
        self._cumulative_wait_store = []
        self._avg_queue_length_store = []
        self.training_epochs = training_epochs


    # 執行模擬並開始訓練
    def run(self, episode, epsilon):
        start_time = timeit.default_timer()

        # 生成路由文件並設置sumo
        # self.TrafficGen.generate_routefile(seed=episode)
        traci.start(self.sumo_cmd)
        print("Simulating...")

        self.step = 0
        self.waiting_times = {}
        self.total_reward = 0
        self.queue_length = 0
        self.sum_waiting_time = 0
        
        old_total_wait = 0
        old_state = -1
        old_action = -1

        while self.step < self.max_step:

            # 獲取路口當前狀態
            current_state = self.get_state()

            # 計算之前動作的獎勵（兩個動作之間的累積等待時間變化）
            current_total_wait = self.collect_waiting_times()
            reward = (old_total_wait - current_total_wait) / 100

            # 將數據保存到buffer中
            if self.step != 0:
                self.Buffer.add_sample((old_state, old_action, reward, current_state))

            # 根據路口當前狀態選擇要激活的紅綠燈
            action = self.choose_action(current_state, epsilon)

            # 綠燈紅燈之間記得要加黃燈
            if self.step != 0 and old_action != action:
                self.set_yellowPhase(old_action)
                self.simulate(self.yellow_duration)

            # 執行之前選擇的action
            self.set_greenPhase(action)
            self.simulate(self.green_duration)

            # 儲存state, action，並累積獎勵
            old_state = current_state
            old_action = action
            old_total_wait = current_total_wait

            # 只儲存有意義的獎勵，以更好地觀察代理行為是否正確
            if reward < 0:
                self.total_reward += reward

        self.save_episode_stats()
        print("Total reward:", self.total_reward, "- Epsilon:", round(epsilon, 2))
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        print("Training...")
        start_time = timeit.default_timer()
        for _ in range(self.training_epochs):
            self.replay()
        training_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time, training_time


    def simulate(self, steps_todo):
        # 在收集統計數據的同時執行 SUMO 的步驟
        if (self.step + steps_todo) >= self.max_step:
            steps_todo = self.max_step - self.step

        while steps_todo > 0:
            traci.simulationStep()
            self.step += 1
            steps_todo -= 1
            queue_length = self.get_queue_length()
            self.queue_length += queue_length
            self.sum_waiting_time += queue_length


    def collect_waiting_times(self):
        # 檢索進入道路上每輛車輛的等待時間
        incoming_roads = ["E1", "N1", "W1", "S1"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)
            if road_id in incoming_roads:
                self.waiting_times[car_id] = wait_time
            else:
                if car_id in self.waiting_times:
                    del self.waiting_times[car_id]
        total_waiting_time = sum(self.waiting_times.values())
        return total_waiting_time


    def choose_action(self, state, epsilon):
        # 根據 epsilon-greedy 策略，決定是執行探索性還是最佳化的動作
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return np.argmax(self.Model.predict_one(state))


    def set_yellowPhase(self, old_action):
        yellow_phase_code = old_action * 2 + 1
        traci.trafficlight.setPhase("Light", yellow_phase_code)


    def set_greenPhase(self, action_number):
        if action_number == 0:
            traci.trafficlight.setPhase("Light", NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("Light", NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase("Light", EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase("Light", EWL_GREEN)


    def get_queue_length(self):
        halt_N = traci.edge.getLastStepHaltingNumber("N1")
        halt_S = traci.edge.getLastStepHaltingNumber("S1")
        halt_E = traci.edge.getLastStepHaltingNumber("E1")
        halt_W = traci.edge.getLastStepHaltingNumber("W1")
        queue_length = halt_N + halt_S + halt_E + halt_W
        return queue_length


    def get_state(self):
        state = np.zeros(self.num_states)
        # car_list 儲存SUMO所有車輛的 ID
        car_list = traci.vehicle.getIDList()

        # 取得該車在車道中的位置 (lane_pos) 和該車當前所在的車道 ID (lane_id)
        for car_id in car_list:
            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)
            lane_pos = 750 - lane_pos

            # 將 lane_pos 分成 10 個 cell，根據車輛在車道中的位置來決定
            if lane_pos < 7:
                lane_cell = 0
            elif lane_pos < 14:
                lane_cell = 1
            elif lane_pos < 21:
                lane_cell = 2
            elif lane_pos < 28:
                lane_cell = 3
            elif lane_pos < 40:
                lane_cell = 4
            elif lane_pos < 60:
                lane_cell = 5
            elif lane_pos < 100:
                lane_cell = 6
            elif lane_pos < 160:
                lane_cell = 7
            elif lane_pos < 400:
                lane_cell = 8
            elif lane_pos <= 750:
                lane_cell = 9

            # 根據 lane_id 分類車道
            if lane_id == "W1_0" or lane_id == "W1_1" or lane_id == "W1_2":
                lane_group = 0
            elif lane_id == "W1_3":
                lane_group = 1
            elif lane_id == "N1_0" or lane_id == "N1_1" or lane_id == "N1_2":
                lane_group = 2
            elif lane_id == "N1_3":
                lane_group = 3
            elif lane_id == "E1_0" or lane_id == "E1_1" or lane_id == "E1_2":
                lane_group = 4
            elif lane_id == "E1_3":
                lane_group = 5
            elif lane_id == "S1_0" or lane_id == "S1_1" or lane_id == "S1_2":
                lane_group = 6
            elif lane_id == "S1_3":
                lane_group = 7
            else:
                lane_group = -1

            if lane_group >= 1 and lane_group <= 7:
                car_position = int(str(lane_group) + str(lane_cell))
                valid_car = True
            elif lane_group == 0:
                car_position = lane_cell
                valid_car = True
            else:
                valid_car = False

            if valid_car:
                state[car_position] = 1
        return state


    def replay(self):
        # 從緩衝區中擷取一組樣本，對每個樣本更新學習方程式，然後進行訓練
        batch = self.Buffer.get_samples(self.Model.batch_size)

        if len(batch) > 0:
            states = np.array([val[0] for val in batch])
            next_states = np.array([val[3] for val in batch])

            Qt = self.Model.predict_batch(states)
            Qts = self.Model.predict_batch(next_states)

            x = np.zeros((len(batch), self.num_states))
            y = np.zeros((len(batch), self.num_actions))

            for i, bat in enumerate(batch):
                state, action, reward, _ = bat[0], bat[1], bat[2], bat[3]
                current_q = Qt[i]
                current_q[action] = reward + self.gamma * np.amax(Qts[i])  # update Q(state, action)
                x[i] = state
                y[i] = current_q
            self.Model.train_batch(x, y)


    def save_episode_stats(self):
        self._reward_store.append(self.total_reward)
        self._cumulative_wait_store.append(self.sum_waiting_time)
        self._avg_queue_length_store.append(self.queue_length / self.max_step)

    @property
    def reward_store(self):
        return self._reward_store


    @property
    def cumulative_wait_store(self):
        return self._cumulative_wait_store


    @property
    def avg_queue_length_store(self):
        return self._avg_queue_length_store

