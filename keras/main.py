# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 14:27:37 2020

@author: custodio
 """
from types import SimpleNamespace
import tensorflow as tf
import logging 
# from settings import Settings
from dss_engine import DSSEngine
from dqn_agent import DQNAgent
import os 
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from datetime import timedelta, datetime, date 
import pytz
import numpy as np 
import pickle 
import time 
import h5py 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# import plotter
import dask.delayed as delay
import dask
from statistics import mean

VERYLARGENUMBER = np.inf
GND = 0 
NEUTRAL = 4
OUTP_NAMES = [
    'Offline Episodes Reward History',
    'Offline Episodes Average Loss History',
    'Offline Solution Time (h)',
    'Online Episodes Reward History',
    'Online Episodes Average Loss History',
    'Online Solution Time (h)',
    'Online Episodes Violation History',
    'Online Maximum Voltages',
    'Online Minimum Voltages',
    'Online Tap Positions',
    'No Control Maximum Voltages',
    'No Control Minimum Voltages',
    'No Control Tap Positions',
    'Maximum Iterations',
    'Learning Rate',
    'Discount Factor',
    'L2 Lambida',
    'Activation Function',
    'Memory Capacity',
    'Time Step',
    'Epsilon',
    'Minimum Epsilon',
    'Maximum Epsilon',
    'Epsilon Decay',
    'Target Network Update Frequency',
    'Batch Size', 
    'PV Penetration',
    'Network Name', 
    'Training Cycles',
    'Testing Cycles'
]

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='[%(asctime)s] %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

# tf.debugging.set_log_device_placement(True) 

def run_no_control_simulation(dss, cycles, h5file):
    """Run simulation without control

    Args:
        dss (class): OpenDSS 
        cycles (class): selected days to be simulated
        h5file (h5file): h5 file that stores voltages
    
    Returns:
            (dict): summary of results  
    """
    
    init_time = time.time()
    # dss.reset_env()
    max_voltages = []
    min_voltages = []
    taps = []

    for cycle in cycles:
        
        logging.info (f'No Control, cycle {cycle}')
        dss.set_env(cycle=cycle, mode=0) # 0: snap, 2: time 
        h5cycle = h5file.create_group(f"{cycle}")

        for time_step in range(dss.npts):
            
            logging.info (f'No Control, cycle {cycle} step {time_step}')
            dss.circuit.Solution.Solve()
            # __, __, __, __ = dss.dss_feedback(0, settings.criticalLoads)

            # no_control_info['time_step'].append(time_step)
            # no_control_info['state'].append(state)
            # no_control_info['noviol'].append(noviol)
            
            # dss.circuit.RegControls.First
            # gcycle['CONTROL_ACTIONS'][time_step] = dss.circuit.RegControls.TapNumber
            
        data, cycle_max_voltages, cycle_min_voltages, cycle_tap_position = dss.extractMonitorData()
        h5cycle.create_dataset('VOLTAGES', data = data)  
        max_voltages.extend(cycle_max_voltages)
        min_voltages.extend(cycle_min_voltages)
        taps.extend(cycle_tap_position)

    final_time = time.time()
    total_time = final_time - init_time
    conversion = timedelta(seconds=total_time)
    converted_total_time = str(conversion)
    logging.info(f'Total time: {converted_total_time}')
    
    OUTP_VALUES = (
            max_voltages,
            min_voltages,
            taps
        )
    outp = dict(zip(OUTP_NAMES[10:], OUTP_VALUES))
    
    return outp 

def run_online_training(case, dss, h5file,control, cycles):
    """Run online training (test the DQN control)

    Args:
        case (int): defines the context of the reward function (c0 and nv)
        dss (class): opendss
        h5file (file): results are going to be stored in this file
        control (class): control (DQN agent)
        cycles (list): testing cycles
    """

    def control_trigger(time_step, check_frequency): # TODO: use this function later
        """Trigger control

        Args:
            time_step (int): simulation time
            check_frequency ([type]): frequency that the control should be triggered

        Returns:
            activate (boolean): indicate whether the control can act or not
        """
        activate = not(time_step+1)%check_frequency
        
        return activate

    init_time = time.time()
    dss.reset_env()
    dss.putMonitors()
    control.settings.epsilon = 0.1
    episodes_reward_history = []
    episodes_avg_loss_history = []
    episodes_noviol_history = []
    max_voltages = []
    min_voltages = []
    taps = []
    # min_voltages = h5file.create_dataset('min_voltages', data = [])  
    # max_voltages = h5file.create_dataset('max_voltages', data = [])  
    # try: # just in case of something triggering an error
    # for c,cycle in enumerate(cycles): #TODO: change this after finding the problem
    for c,cycle in enumerate(cycles):
        logging.info (f'Online training, cycle {cycle} ({c+1}/{len(cycles)})')
        h5cycle = h5file.create_group(f"{cycle}")
        dss.set_env(cycle=cycle, mode=0) # 0: snap, 2: time 
        dss.circuit.Solution.SolveSnap()
        __, reward, noviol = dss.dss_feedback(case, False) 

        for episode in range(dss.npts):
            logging.info (f'Online training, cycle {cycle} ({c+1}/{len(cycles)}) step {episode}')
            avg_loss = None

            if not noviol:
                state, reward, noviol = dss.dss_feedback(case, True)
                action = control.act(state)
                next_state, reward, noviol = dss.update_controls(case, True, 'on', action)
                control.memorize(state, action, reward, next_state, noviol)
                state = next_state
            else:
                dss.circuit.Solution.Solve()
                __, reward, noviol = dss.dss_feedback(case, False)

            episodes_reward_history.append(reward)
            episodes_noviol_history.append(noviol)
            action = control.act(state)
            next_state, reward, noviol = dss.update_controls(case, 'on', action)
            episodes_reward_history.append(reward)
            next_state = np.reshape(next_state, [1, control.state_size])
            control.memorize(state, action, reward, next_state, noviol)
            state = next_state

            
            if episode % control.settings.update_target_network == 0:
                # update the the target network with new weights
                control.update_target_model()
                    
            if len(control.memory) > control.settings.batch_size:
                avg_loss = control.replay()
                episodes_avg_loss_history.append(avg_loss)

            logging.info("episode: {}/{}, reward: {}, loss: {}, e: {}"
            .format(episode, 
                dss.npts,
                reward,
                avg_loss, 
                control.settings.epsilon))

        data, cycle_max_voltages, cycle_min_voltages, cycle_tap_position = dss.extractMonitorData()
        h5cycle.create_dataset('VOLTAGES', data = data)  
        max_voltages.extend(cycle_max_voltages)
        min_voltages.extend(cycle_min_voltages)
        taps.extend(cycle_tap_position)
       
    final_time = time.time()
    total_time = final_time - init_time
    conversion = timedelta(seconds=total_time)
    converted_total_time = str(conversion)
    logging.info(f'Total time: {converted_total_time}')

    OUTP_VALUES = (
            episodes_reward_history,
            episodes_avg_loss_history,
            converted_total_time,
            episodes_noviol_history,
            max_voltages,
            min_voltages,
            taps
        )
    outp = dict(zip(OUTP_NAMES[3:], OUTP_VALUES))
    
    return outp 

def run_offline_training(case, dss, control, cycles):
    """This function executes the offline training

    Args:
        case (int): defines the context (nv and c0) used
        dss (class): opendss
        control (class): control agent (DQN)
        cycles (array): cycles that are going to be used

    Returns: 
        episodes_reward_history (dict): cumulative and averaged rewards  
    """
    def train_control(state, reward, noviol):
        episode_reward = []
        episode_avg_loss = []
        if noviol == False:
            state, reward, noviol = dss.dss_feedback(case, True) 
            vreg = dss.store_vreg()
            for attempt in range(control.settings.max_iterations):
                action = control.act(state)
                next_state, reward, noviol = dss.update_controls(case, True, 'off', action)
                episode_reward.append(reward)
                control.memorize(state, action, reward, next_state, noviol)
                state = next_state
                
                if noviol:
                    control.update_target_model()
                    dss.restore_vreg(vreg)
                    break
                
                if attempt % control.settings.update_target_network == 0:
                    # update the the target network with new weights
                    control.update_target_model()
                        
                if len(control.memory) > control.settings.batch_size:
                    avg_loss = control.replay()
                    episode_avg_loss.append(avg_loss)
        
        else:
            episode_reward.append(reward)
            if len(control.memory) > control.settings.batch_size:
                avg_loss = control.replay()
                episode_avg_loss.append(avg_loss)
        
        ep_cum_r = np.sum(np.array(episode_reward))/len(episode_reward)
        ep_avg_r = np.mean(np.array(episode_reward))
        if len(episode_avg_loss)>0:
            ep_avg_l = np.mean(np.array(episode_avg_loss))
        else:
            ep_avg_l = np.nan
            
        return ep_cum_r, ep_avg_r, ep_avg_l
        
    init_time = time.time()
    episodes_reward_history = {'cumulative': [], 'avg':[]}   
    episodes_avg_loss_history = []
    # try: # just in case of something triggering an error
    for c,cycle in enumerate(cycles): 
        logging.info (f'Offline training, cycle {cycle} ({c+1}/{len(cycles)})')
        dss.set_env(cycle=cycle, mode=0) # 0: snap, 2: time 

        for episode in range(dss.npts):
            # logging.info (f'Offline training, cycle {cycle} ({c+1}/{len(cycles)}) step {episode}')
            dss.circuit.Solution.Solve()
            state, reward, noviol = dss.dss_feedback(case, False) 
            ep_cum_r, ep_avg_r, ep_avg_l = train_control(state, reward, noviol)
            episodes_reward_history['cumulative'].append(ep_cum_r)
            episodes_reward_history['avg'].append(ep_avg_r)
            episodes_avg_loss_history.append(ep_avg_l)
            logging.info("cycle: {}/{}, episode: {}/{}, reward: {}, loss: {}, e: {}"
            .format(c+1,
                len(cycles),
                episode, 
                dss.npts,
                ep_cum_r,
                ep_avg_l,
                control.settings.epsilon))
            control.tensorboard.update_stats(reward_avg=ep_avg_r, reward_cum=ep_cum_r, epsilon=control.settings.epsilon)
    
    final_time = time.time()
    total_time = final_time - init_time
    conversion = timedelta(seconds=total_time)
    converted_total_time = str(conversion)
    logging.info(f'Total time: {converted_total_time}')

    OUTP_VALUES = (
        episodes_reward_history,
        episodes_avg_loss_history,
        converted_total_time,
    )
    outp = dict(zip(OUTP_NAMES[:3], OUTP_VALUES))
    
    return outp 

def train_test_cycles(dss, control):
    """Define which cycles are going to be used during off and on training

    Args:
        dss (class): opendss
        control (class): control (dqn)

    Returns:
        trainingCyles (array): cycles of the offline training
        testingCycles (array): [cycles of the online training
    """
    
    cycles = int(dss.settings.days/dss.settings.simulationCycleinDays)  
    tsh = int(365/(5*dss.settings.simulationCycleinDays)) # five groups
    num_elms = round(cycles/5) 
    numtrainingCyclesByGroup = round(control.settings.ratioTrainTest*cycles/5)
    trainingCyclesList = []
    testingCyclesList = []
    
    for i in range(5):
        selected_cycles = np.random.choice(np.arange(tsh*i+1,tsh*(i+1)), num_elms) 
        trainingCyclesList.extend(selected_cycles[:numtrainingCyclesByGroup])
        testingCyclesList.extend(selected_cycles[numtrainingCyclesByGroup:])
        
    trainingCycles = np.array(trainingCyclesList)
    testingCycles = np.array(testingCyclesList)

    return trainingCycles, testingCycles

def save_results(outp, file_path):
    counter=1
    while True:
        if os.path.isfile(f'{file_path}.pkl'):
            file_path = f'{file_path}({counter})'
            counter+=1
            continue
        else:
            with open(f'{file_path}.pkl', 'wb') as f:
                    pickle.dump(outp, f, pickle.HIGHEST_PROTOCOL)
            break

def run(inp):         
    """Run OFF,ON, NO CONTROL simulations

    Args:
        inp (tuple): input -> (Network, PV penetration, scenario, dss_settings, control_settings)

    Returns:
        dict: summary of results from training
    """
    outp = {}
    dss = DSSEngine(inp[0:4])
    control = DQNAgent(inp[4], dss.state_size, dss.action_size)
    training_cycles, testing_cycles = train_test_cycles(dss,control)
    num_training_cycles = len(training_cycles)
    num_testing_cycles = len(testing_cycles)
    logging.info(f'Days used in offline training: {num_training_cycles}')    
    logging.info(f'Days used in online training: {num_testing_cycles}')    
  
    for i in range(len(dss.settings.nv)):
        name = f'{dss.settings.days}_{dss.settings.vs[i]}_nv{dss.settings.nv[i]}_c0{dss.settings.c0[i]}_20200805'
        outp[f'{name}'] = {}
        #HINT *Offline Training 
        if control.settings.offline_training:
            results = run_offline_training(i, dss, control, training_cycles)
            # results = run_offline_training(i, dss, control, [0])
            control.save(f'{name}_OFF')
            outp[f'{name}'].update(results)
        else:
            control.load(f'{name}_OFF')
        #HINT *Online Training
        if control.settings.online_training:
            h5file_name = rf"{control.settings.outp_folder}\online_info_{name}_spyder2.h5"
            h5file = h5py.File(h5file_name, "w")
            logging.info('Online training')
            episodes_reward_history, avg_loss_history = run_online_training(i, dss, h5file, control, testing_cycles)
            control.save(f'{name}_ON')
            kVBases = dss.get_kvbases()
            h5file['violations'] = dss.get_cycle_info(h5file)
            h5file.close()

    #HINT *No Control
    if control.settings.no_control:
        dss = DSSEngine(inp[0:4])
        h5file_name = rf"{control.settings.outp_folder}\nocontrol_info_spyder2.h5"
        h5file = h5py.File(h5file_name, "w")
        logging.info('No Control')
        results = run_no_control_simulation(dss, testing_cycles, h5file)
        # results = run_no_control_simulation(dss, [300], h5file)
        outp.update(results)
        outp.update({'No Control H5file Name': h5file_name})
        h5file['VIOLATIONS'] = dss.get_cycle_info(h5file)
        h5file.close()
        
    OUTP_VALUES = (
       control.settings.max_iterations,
       control.settings.learning_rate,
       control.settings.gamma,
       control.settings.l,
       control.settings.afun,
       control.settings.memory_capacity,
       control.settings.time_step,
       control.settings.epsilon,
       control.settings.epsilon_min,
       control.settings.max_epsilon,
       control.settings.epsilon_decay,
       control.settings.update_target_network,
       control.settings.batch_size,
       inp[1],
       inp[0],
       training_cycles,
       testing_cycles
    )
    outp['info'] = dict(zip(OUTP_NAMES[13:], OUTP_VALUES))
    save_results(outp, f'{control.settings.outp_folder}/outp_{num_training_cycles}OFF_{num_testing_cycles}ON')
    
    return outp 

if __name__ == "__main__":
    int_d0 = 1
    int_d1 = 365
    d0 = datetime(2018,1,1,0)+timedelta(days=int_d0-1)
    d0 = d0.strftime('%Y-%m-%d')
    d1 = datetime(2018,1,1,0)+timedelta(days=int_d1-1)
    d1 = d1.strftime('%Y-%m-%d')
    days_used_in_on_off_training = 60 #*defines the length of the dataset
    # Run the simulation
    control_settings = { 
        'max_iterations': 100, 
        'learning_rate': 0.0001,
        'frequency': 15, #min
        'gamma': 0.99, 
        'l': 0.1,
        'afun': 'tanh',
        'memory_capacity': 300,
        'time_step':300, 
        'epsilon': 1, 
        'epsilon_min': 0.1, 
        'max_epsilon': 1, 
        'epsilon_decay': 0.9995,
        'offline_training': True,
        'online_training': False, 
        'without_control': False,
        'plotter': False,
        'ratioTrainTest': 0.7,
        'update_target_network': 50,
        'batch_size':16,
        'outp_folder' :  r'F:\Users\custo\OneDrive\Msc\BEPE\Deep Reinforcement Learning\DQN\keras\results'
        }

    dss_settings = {
        'create_pv_file': False,
        'find_critical_loads': False,
        'voltage_scalers': [0.8,1.2],                                              # voltage range used to scale data (input neural network)
        'networkName': 'TAQ-ETR103',                                              
        'neutralNum': 4,
        'Sbase': 100e3,
        'ZIP_P':(0.0, 1.0, 0.0),                                                  # Coefficients of the ZIP model - active power
        'ZIP_Q':(1.0, 0.0, 0.0),                                                  # Coefficients of the ZIP model - reactive power
        'enableCapControls':True,                                                 # If False, the simulations are performed with the CapControls blocked
        'enableCapacitors':True,                                                 
        'enableRegControls':True,                                                 # If False, the simulations are performed with the RegControls blocked
        'timeStep':300, # step_time in qlearning 
        'd0': d0,
        'd1': d1,
        'days': days_used_in_on_off_training,
        'simulationCycleinDays': 1,
        'SourceVoltage':1.0,                                                      # Voltage at the HV bus of the simulated substation transformer (pu)
        'loadsPowerFactor':0.85,                                                  # Default power factor used in all loads
        'pvsPowerFactor':1.0,                                                     # Default power factor used in all PV systems
        'enableSyntheticLoadShapesB1':True,
        'limitResidentialPV':100,
        'bandVR':dict(zip([30, 300], [1, 1])),
        'OnOffCap':dict(zip([30, 300], [1, 1])),
        'customMeter':True,
        '# hdf_file_cuca_a' : 'Loadshapes_16-07-2019_PAULISTA_2018',
        'hdf_file_cuca_a' : 'Loadshapes_22-08-2019_PAULISTA',
        'hdf_file_cuca_a_keys' : 'loads_and_keys_20191016',
        'hdf_file_b1' : 'Curvas_de_carga300s_compact_vs8',
        'hdf_file_b3' : 'CuCa_B3_300s_v2',
        'pv_shapes_file' : 'pv.pkl',
        'ls_folder' :  r'C:\Users\custo\Documents\temp_data',
        'volt_var_folder' :  r'\\le41-16\temp\Vinicius',  
        'task_folder' :  r'C:\temp\PA3047',  
        'network_folder' :  r'C:\Users\custo\OneDrive\Msc\BEPE\rl\Network',  
        'network_version' : '20190620-1',
        'master' : 'vs2',
        'simStartDblHour':0,
        'outp_folder' :  r'C:\Users\custo\OneDrive\Msc\BEPE\rl\res_best_rf',
        'voltageLV' : {'min':0.92,'max':1.05,
                     'nrp_under': {'min': 0.87, 'max': 0.92}, 
                     'nrp_over': {'min': 1.05, 'max': 1.06}, 
                     'nrc_under': 0.87, 
                     'nrc_over': 1.06 }, 
        'voltageMV' : {'min':0.93,'max':1.05,
                     'nrp_under': {'min': 0.90, 'max': 0.93}, 
                     'nrc_under': 0.90, 
                     'nrc_over': 1.05}, 
        'LVMVlimiar' : 1000,
        'find_critical_loads': False,
        'pvIrradValues' : np.arange(0.1,1.1,0.1),
        'vs': ['r0'], # used to test just one version
        # 'vs': ['r0', 'r1', 'r2'], # used to test all versions
        'nv': [100], # same as above
        # 'nv': [1,10,100,10000,100,10000,10000],
        'c0': [1]
        # 'c0': [1,1,1,1,10,10,100]
        }
    
    dss_settings['d0'] = pd.to_datetime(dss_settings['d0']).tz_localize('Brazil/East').tz_convert(pytz.utc).strftime('%Y-%m-%d %H:%M:%S%z')
    dss_settings['d1'] = pd.to_datetime(dss_settings['d1']).tz_localize('Brazil/East').tz_convert(pytz.utc).strftime('%Y-%m-%d %H:%M:%S%z')
    # dss_settings['days'] = (pd.to_datetime(dss_settings['d1']) - pd.to_datetime(dss_settings['d0'])).days + 1
    logging.info(f'Simulation with {dss_settings["days"]} days')
    dss_settings = SimpleNamespace(**dss_settings)
    control_settings = SimpleNamespace(**control_settings)
    inp = ('TAQ-ETR103', 60, 0, dss_settings, control_settings)
    outp = run(inp)
    

    # *Plot Results 
    outp = pd.read_pickle(f'{control_settings.outp_folder}/outp_40OFF_20ON.pkl')
    case = list(outp.keys())
    # h5file = h5py.File(outp[case[0]]['H5file Name'], "r")
    # plotter.barplot(h5file)
    # fig, ax = plt.subplots(4,1, figsize=(13, 8), dpi=100, sharey = False, sharex=True)
    # xlim_tsh = len(outp[case[0]]['Online Maximum Voltages'])
    # ax[0].plot(outp['No Control Maximum Voltages'], 'black',lw=1.5)
    # ax[0].plot(outp['No Control Minimum Voltages'], 'black',lw=1.5, label='Basic Control')
    # ax[0].plot([0,xlim_tsh],[0.92,0.92], ls=":", color='r', lw=2.5)
    # ax[0].plot([0,xlim_tsh],[1.05,1.05], ls=":", color='r', lw=2.5)
    # ax[1].plot(outp[case[0]]['Online Maximum Voltages'], 'blue',lw=1.5)
    # ax[1].plot(outp[case[0]]['Online Minimum Voltages'], 'blue',lw=1.5, label='Online DQN Control')
    # ax[1].plot([0,xlim_tsh],[0.92,0.92], ls=":", color='r', lw=2.5)
    # ax[1].plot([0,xlim_tsh],[1.05,1.05], ls=":", color='r', lw=2.5)
    # ax[2].plot(outp[case[0]]['Online Maximum Voltages'], 'blue',lw=1.5, label='Online DQN Control')
    # ax[2].plot(outp[case[0]]['Online Minimum Voltages'], 'blue',lw=1.5)
    # ax[2].plot(outp['No Control Maximum Voltages'], 'black',lw=1.5, label='Basic Control')
    # ax[2].plot(outp['No Control Minimum Voltages'], 'black',lw=1.5)
    # ax[2].plot([0,xlim_tsh],[0.92,0.92], ls=":", color='r', lw=2.5)
    # ax[2].plot([0,xlim_tsh],[1.05,1.05], ls=":", color='r', lw=2.5)
    # ax[3].plot(outp[case[0]]['Online Tap Positions'], 'blue',lw=1.5, label='Online DQN Control')
    # ax[3].plot(outp['No Control Tap Positions'], 'black',lw=1.5, label='Basic Control')
    # # ax[2].plot(outp[case[0]]['Online Episodes Reward History'])

    # # ax[3].plot(outp[case[0]]['Online Episodes Average Loss History'])
    # ax[0].set_title('Minimum and Maximum Voltages', fontdict={'fontsize':10})
    # ax[1].set_title('Minimum and Maximum Voltages', fontdict={'fontsize':10})
    # ax[2].set_title('Minimum and Maximum Voltages', fontdict={'fontsize':10})
    # ax[3].set_title('Tap Positions (Basic control and Online DQN Control)', fontdict={'fontsize':10})
    
    # ax[0].set(ylabel='Voltage (pu)')
    # ax[1].set(ylabel='Voltage (pu)')
    # ax[2].set(ylabel='Voltage (pu)')
    # ax[3].set_ylabel('Tap (pu)')
    # # ax[3].set_ylabel('Value')
    # # ax[0].set_xlabel('Time (days)')
    # # ax[1].set_xlabel('Time (days)')
    # # ax[2].set_xlabel('Time (days)')
    # ax[3].set_xlabel('Time (days)')
    # # ax[2].set_xlabel('episodes/snapshots')
    # # ax[3].set_xlabel('episodes/snapshots')
    # #Settings
    # ax[1].legend(loc="lower left", 
    #                 fontsize= 'medium', 
    #                 fancybox = False,
    #                 # markerscale = 35,
    #                 handletextpad=-2.0, 
    #                 handlelength=0,
    #                 frameon = False, 
    #                 bbox_to_anchor=(0.15,-0.05),
    #                 # borderpad=0.1
    #                 )
    # # ax[0].grid(False)
    # ax[0].set_ylim([0.85,1.12])
    # ax[1].set_ylim([0.85,1.12])
    # ax[2].set_ylim([0.85,1.12])
    # ax[3].set_ylim([0.85,1.12])
    # ax[0].set_xlim([0,xlim_tsh])
    # ax[1].set_xlim([0,xlim_tsh])
    # ax[2].set_xlim([0,xlim_tsh])
    # ax[3].set_xlim([0,xlim_tsh])
    # # ax[3].set_xlim([0,xlim_tsh])
    # ax[1].tick_params(direction='in',color='#595959', grid_alpha=0.5)
    # # a = [0,48,96,144,192,240]
    # # b = [0,4,8,12,16,20]
    # a = np.arange(0,288*len(outp['info']['Testing Cycles']), 288)
    # b = [f'day {i+1}' for i in range(len(outp['info']['Testing Cycles']))]
    # ax[0].set_xticks(a)
    # ax[1].set_xticks(a)
    # ax[2].set_xticks(a)
    # ax[3].set_xticks(a)
    # ax[0].set_xticklabels(b)
    # ax[1].set_xticklabels(b)
    # ax[2].set_xticklabels(b)
    # ax[3].set_xticklabels(b)
    # # ax[0].set_xticklabels(b)
    # # ax[1].set_xticks(a)
    # # ax[1].set_xticklabels(b)
    # ax[0].legend(loc="lower left", ncol=2, 
    #               fontsize= 'small', 
    #               fancybox = False,
    #               # markerscale = 35,
    #               frameon = False, 
    #               borderpad=0.1)

    # ax[1].legend(loc="lower left", ncol=2, 
    #                fontsize= 'small', 
    #               fancybox = False,
    #               # markerscale = 35,
    #               frameon = False, 
    #               borderpad=0.1)
    # ax[2].legend(loc="lower left", ncol=2, 
    #               fontsize= 'small', 
    #               fancybox = False,
    #               # markerscale = 35,
    #               frameon = False, 
    #               borderpad=0.1)

    # ax[3].legend(loc="lower left", ncol=2, 
    #                fontsize= 'small', 
    #               fancybox = False,
    #               # markerscale = 35,
    #               frameon = False, 
    #               borderpad=0.1)
    # plt.tight_layout()
    
    #Second Plot
    
    fig, ax = plt.subplots(2,1, figsize=(10, 8), dpi=100, sharey = False, sharex=True)
    ax[0].plot(outp[case[0]]['Offline Episodes Reward History']['cumulative'], 'black')
    ax[1].plot(outp[case[0]]['Offline Episodes Average Loss History'], 'black')
    # ax[0].set_title('Scenario Reward', fontdict={'fontsize':10})
    # ax[1].set_title('Scenario Average Loss', fontdict={'fontsize':10})
    a = np.arange(0,288*len(outp['info']['Training Cycles']), 288*5)
    b = [f'day {i}' for i in np.arange(1,len(outp['info']['Training Cycles']),5)]
    xlim_tsh = 288*len(outp['info']['Training Cycles'])
    ax[0].set_xticks(a)
    ax[1].set_xticks(a)
    ax[0].set_xticklabels(b)
    ax[1].set_xticklabels(b)
    ax[0].set_ylim([0,110])
    ax[1].set_ylim([-5,110])
    ax[0].set_xlim([0,xlim_tsh])
    ax[1].set_xlim([0,xlim_tsh])
    ax[0].set_ylabel('Reward by Scenario')
    ax[1].set_ylabel('Average Loss')
    ax[1].set_xlabel('Time (days')
    plt.tight_layout()
    
    plt.show()