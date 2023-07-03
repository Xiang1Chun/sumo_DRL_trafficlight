import traci
import numpy as np
import random
import timeit
import os

NS_GREEN = 0
NS_YELLOW = 1
NSL_GREEN = 2
NSL_YELLOW = 3
EW_GREEN = 4
EW_YELLOW = 5
EWL_GREEN = 6
EWL_YELLOW = 7


class Simulation:
    def __init__(self, Model, TrafficGen, sumo_cmd, max_steps, green_duration, yellow_duration, num_states, num_actions):
        self.Model = Model
        self.TrafficGen = TrafficGen
        self.step = 0
        self.sumo_cmd = sumo_cmd
        self.max_step = max_steps
        self.green_duration = green_duration
        self.yellow_duration = yellow_duration
        self.num_states = num_states
        self.num_actions = num_actions
        self._reward_episode = []
        self._queue_length_episode = []


    def run(self, episode):
        start_time = timeit.default_timer()

        # self.TrafficGen.generate_routefile(seed=episode)
        traci.start(self.sumo_cmd)
        print("Simulating...")

        # inits
        self.step = 0
        self.waiting_times = {}
        old_total_wait = 0
        old_action = -1

        while self.step < self.max_step:
            current_state = self.get_state()
            current_total_wait = self.collect_waiting_times()
            reward = old_total_wait - current_total_wait
            action = self.choose_action(current_state)

            if self.step != 0 and old_action != action:
                self.set_yellow_phase(old_action)
                self.simulate(self.yellow_duration)

            self.set_green_phase(action)
            self.simulate(self.green_duration)
            old_action = action
            old_total_wait = current_total_wait
            self.reward_episode.append(reward)

        print("Total reward:", np.sum(self._reward_episode))
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)
        return simulation_time


    def simulate(self, steps_todo):
        if (self.step + steps_todo) >= self.max_step:
            steps_todo = self.max_step - self.step

        while steps_todo > 0:
            traci.simulationStep()
            self.step += 1
            steps_todo -= 1
            queue_length = self.get_queue_length()
            self.queue_length_episode.append(queue_length)


    def collect_waiting_times(self):
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


    def choose_action(self, state):
        return np.argmax(self.Model.predict_one(state))

    def set_yellow_phase(self, old_action):
        yellow_phase_code = old_action * 2 + 1
        traci.trafficlight.setPhase("Light", yellow_phase_code)

    def set_green_phase(self, action_number):
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
        car_list = traci.vehicle.getIDList()

        for car_id in car_list:
            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)
            lane_pos = 750 - lane_pos

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


    @property
    def queue_length_episode(self):
        return self._queue_length_episode


    @property
    def reward_episode(self):
        return self._reward_episode



