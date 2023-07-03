from __future__ import absolute_import
from __future__ import print_function
import os
import datetime
from shutil import copyfile
from train_simulation import Simulation
from generator import Generator
from buffer import Buffer
from model import TrainModel
from visualization import Visualization
from utils import importTrain, set_sumo, set_trainPath

if __name__ == "__main__":

    # 參數設置
    config = importTrain(config_file='training_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    path = set_trainPath(config['models_path_name'])

    Model = TrainModel(
        config['num_layers'], 
        config['width_layers'], 
        config['batch_size'], 
        config['learning_rate'], 
        input_dim=config['num_states'],
        output_dim=config['num_actions']
    )

    buffer = Buffer(
        config['buffer_max'],
        config['buffer_min']
    )

    TrafficGen = Generator(
        config['max_steps'], 
        config['car_number']
    )

    Visualization = Visualization(
        path, 
        dpi=96
    )
        
    Simulation = Simulation(
        Model,
        buffer,
        TrafficGen,
        sumo_cmd,
        config['gamma'],
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states'],
        config['num_actions'],
        config['epochs']
    )
    
    episode = 0
    timestamp_start = datetime.datetime.now()

    # 開始訓練
    while episode < config['episodes']:
        print('\n----- Episode', str(episode+1), 'of', str(config['episodes']))
        epsilon = 1.0 - episode / config['episodes']
        simulation_time, training_time = Simulation.run(episode, epsilon)
        print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:', round(simulation_time+training_time, 1), 's')
        episode += 1

    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
    print("----- Session info saved at:", path)


    # 儲存資料
    Model.save_model(path)

    copyfile(src='training_settings.ini', dst=os.path.join(path, 'training_settings.ini'))

    Visualization.save_data_and_plot(data=Simulation.reward_store, filename='reward', xlabel='Episode', ylabel='Cumulative negative reward')
    Visualization.save_data_and_plot(data=Simulation.cumulative_wait_store, filename='delay', xlabel='Episode', ylabel='Cumulative delay (s)')
    Visualization.save_data_and_plot(data=Simulation.avg_queue_length_store, filename='queue', xlabel='Episode', ylabel='Average queue length (vehicles)')
