import configparser
from sumolib import checkBinary
import os
import sys

# .ini文件的讀取、設定 SUMO 參數，以及創建模型和測試路徑

# 讀取train.ini文件的內容，並存放到config dictionary
def importTrain(config_file):

    # 解析train.ini文件的內容
    text = configparser.ConfigParser()
    text.read(config_file)

    config = {}
    config['gui'] = text['simulation'].getboolean('gui')
    config['episodes'] = text['simulation'].getint('episodes')
    config['max_steps'] = text['simulation'].getint('max_steps')
    config['car_number'] = text['simulation'].getint('car_number')
    config['green_duration'] = text['simulation'].getint('green_duration')
    config['yellow_duration'] = text['simulation'].getint('yellow_duration')
    config['num_layers'] = text['model'].getint('num_layers')
    config['width_layers'] = text['model'].getint('width_layers')
    config['batch_size'] = text['model'].getint('batch_size')
    config['learning_rate'] = text['model'].getfloat('learning_rate')
    config['epochs'] = text['model'].getint('epochs')
    config['buffer_min'] = text['buffer'].getint('buffer_min')
    config['buffer_max'] = text['buffer'].getint('buffer_max')
    config['num_states'] = text['agent'].getint('num_states')
    config['num_actions'] = text['agent'].getint('num_actions')
    config['gamma'] = text['agent'].getfloat('gamma')
    config['models_path_name'] = text['dir']['models_path_name']
    config['sumocfg_file_name'] = text['dir']['sumocfg_file_name']
    return config

# 讀取test.ini文件的內容，並存放到config dictionary
def importTest(config_file):

    text = configparser.ConfigParser()
    text.read(config_file)
    config = {}
    config['gui'] = text['simulation'].getboolean('gui')
    config['max_steps'] = text['simulation'].getint('max_steps')
    config['car_number'] = text['simulation'].getint('car_number')
    config['episode_seed'] = text['simulation'].getint('episode_seed')
    config['green_duration'] = text['simulation'].getint('green_duration')
    config['yellow_duration'] = text['simulation'].getint('yellow_duration')
    config['num_states'] = text['agent'].getint('num_states')
    config['num_actions'] = text['agent'].getint('num_actions')
    config['sumocfg_file_name'] = text['dir']['sumocfg_file_name']
    config['models_path_name'] = text['dir']['models_path_name']
    config['model_to_test'] = text['dir'].getint('model_to_test')
    return config

# 設定sumo參數
def set_sumo(gui, sumocfg_file_name, max_steps):
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    if gui == False:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # setting the cmd command to run sumo at simulation time
    sumo_cmd = [sumoBinary, "-c", os.path.join('intersection', sumocfg_file_name), "--no-step-log", "true", "--waiting-time-memory", str(max_steps)]

    return sumo_cmd


# 創建用於存儲訓練模型的路徑
def set_trainPath(models_path_name):
    models_path = os.path.join(os.getcwd(), models_path_name, '')
    os.makedirs(os.path.dirname(models_path), exist_ok=True)

    dir_content = os.listdir(models_path)
    if dir_content:
        previous_versions = [int(name.split("_")[1]) for name in dir_content]
        new_version = str(max(previous_versions) + 1)
    else:
        new_version = '1'

    data_path = os.path.join(models_path, 'model_'+new_version, '')
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    return data_path 


def set_testPath(models_path_name, model_n):
    model_folder_path = os.path.join(os.getcwd(), models_path_name, 'model_'+str(model_n), '')

    if os.path.isdir(model_folder_path):    
        plot_path = os.path.join(model_folder_path, 'test', '')
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        return model_folder_path, plot_path
    else: 
        sys.exit('The model number specified does not exist in the models folder')
