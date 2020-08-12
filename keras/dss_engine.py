'''
Created on May 7, 2019

@authors: Guilherme Custodio
'''
from win32com.client import makepy
from math import sqrt, pi, cos, sin
import win32com.client
import networkx as nx
import pandas as pd
import numpy as np
import decimal
import sys
import os
import csv
import pickle
from operator import itemgetter
import logging
from random import randint, shuffle, choice, sample
import pytz
import h5py
import itertools
import networkx as nx
import random
from dss import DSS, enums
from numba import jit, jitclass 
import dask
from sklearn.preprocessing import MinMaxScaler
GND = 0
VERYLARGENUMBER = np.inf


class DSSEngine(object):
    def __init__(self, inp):  # Runs when instance of the class is created

        DSS.Start(0)
        # Assign a variable to each of the interfaces for easier access
        self.dssObj = DSS
        self.text = self.dssObj.Text
        self.circuit = self.dssObj.ActiveCircuit
        self.solution = self.circuit.Solution
        self.cktElement = self.circuit.ActiveCktElement
        self.bus = self.circuit.ActiveBus
        self.meters = self.circuit.Meters
        self.loads = self.circuit.Loads
        self.lines = self.circuit.Lines
        self.transformers = self.circuit.Transformers
        self.regControls = self.circuit.RegControls
        self.activeClass = self.circuit.ActiveClass

        self.PVPenetration = inp[1]
        self.Scenario = inp[2]
        self.settings = inp[3]
        self.neutralNum = self.settings.neutralNum
        self.Transformer = inp[0]
        self.Substation = self.Transformer.split('-')[0]
        # self.npts = 3 
        self.npts = int(86400/self.settings.timeStep)*self.settings.simulationCycleinDays
        # Clear memory and compile the network
        self.dssObj.ClearAll
        logging.info('Running non-reduced version of the network')
        self.text.Command = rf"Compile {self.settings.network_folder}/{self.Transformer.split('-')[0]}/Master_{self.Transformer}_{self.settings.master}.dss"

        if not self.settings.enableCapacitors:
            self.text.Command = f"BatchEdit Capacitor..* Enabled=False"

        self.Feeders = [item.upper() for item in self.circuit.Lines.AllNames if item.find(rf"{self.Substation.lower()}") != -1]

        # Enable PVs for the current scenario
        pvs = pd.read_pickle(rf"{self.settings.ls_folder}\9.Volt_var\scripts/_____scenarios_pv/scenarios_pv_{self.Transformer}.pkl")
        number_of_pvs = int((self.PVPenetration/100) * pvs.shape[1])
        pvs = set(pvs.loc[self.Scenario].iloc[0:number_of_pvs])
        self.PVs = pvs

        LoadShapeResidentialLV = pd.read_pickle(rf"{self.settings.ls_folder}/_____scenarios_dem_prof/scenarios_demand_profiles_{self.Transformer}.pkl")
        LoadShapeResidentialLV = LoadShapeResidentialLV.transpose()[self.Scenario]
        self.LoadShapeResidentialLV = LoadShapeResidentialLV

        store_PV = pd.read_pickle(rf"{self.settings.ls_folder}\\{self.settings.pv_shapes_file}")[0] #TO DO: salvar pvsProfile.pkl como Series e não como DataFrame
        temp_pv = pd.DataFrame([store_PV.values[-1]], index=[store_PV.index[-1] + pd.Timedelta('1 minute')])
        store_PV = pd.concat([store_PV, temp_pv], axis=0)
        store_PV.index = store_PV.index.tz_convert(pytz.utc)
        self.store_PV = store_PV

        self.store_A = h5py.File(rf"{self.settings.ls_folder}/{self.settings.hdf_file_cuca_a}.hdf5", "r")
        self.keys_A = pd.read_pickle(rf"{self.settings.ls_folder}/{self.settings.hdf_file_cuca_a_keys}.pkl")
        self.store_B1 = h5py.File(rf"{self.settings.ls_folder}/LVSyntheticShapes/{self.settings.hdf_file_b1}.h5", "r")
        self.store_B3 = h5py.File(rf"{self.settings.ls_folder}/{self.settings.hdf_file_b3}.hdf5", "r")

        # Set the load model
        self.text.Command = f"BatchEdit Load._ip_.* Enabled=False"
        self.text.Command = f"Makebuslist"
        # self.text.Command = f"BatchEdit Load..* Vminpu=0.92 Vmaxpu=1.05"
        self.text.Command = f"BatchEdit Load..* PF={self.settings.loadsPowerFactor}"
        self.text.Command = 'CalcVoltageBases'

        if self.settings.ZIP_P == (1.0, 0.0, 0.0) and self.settings.ZIP_Q == (1.0, 0.0, 0.0):
            self.text.Command = f"BatchEdit Load..* model=2"
        
        else:
            self.text.Command = f"BatchEdit Load..* model=8 zipv={self.settings.ZIP_P + self.settings.ZIP_Q + (0.0,)}"

        # self.makeVsourceAnInfinityBus()
        # self.solution_settings(0) # time=2
        # self.total_pv_capacity = self.enable_pvs()

        if self.settings.create_pv_file:
            self.total_pv_capacity = self.install_pvs()
        else:
            self.text.Command = rf'Redirect {self.settings.network_folder}\\PVs.dss'

        if self.settings.find_critical_loads:
            logging.info('Finding Critical Loads')
            self.critical_loads = self.find_critical_loads()
        
        else:
            self.critical_loads = pd.read_pickle(rf'{self.settings.outp_folder}/crit_lds_{self.PVPenetration}_pl.pickle')
            # self.critical_loads = pd.read_pickle(rf'{self.settings.outp_folder}/crit_lds_60_pl.pickle')
        
        logging.info(f'Num critical loads:{len(self.critical_loads)}')
        self.state_size = len(self.critical_loads)
        self.action_size, self.actions = self.action_space()
                        
    def getDblHour(self):
    
        self.set_loadshapes(0)
        # npts = int(86400/self.settings.timeStep)#*settings.simulationCycleinDays
        npts = int(86400/self.settings.timeStep)#*settings.simulationCycleinDays
        aggreg_demand = np.zeros(npts)
        
        for name in self.circuit.Loads.AllNames:
            
            if not "_b1" in name:
                continue 
            
            self.circuit.Loads.Name = name 
            self.circuit.LoadShapes.Name = self.circuit.Loads.daily
            load_dem = self.circuit.Loads.kW*np.array(self.circuit.LoadShapes.Pmult)
            try:
                aggreg_demand += load_dem[:288]
            except:
                None
           
        sunrise = int(6*3600/self.settings.timeStep)
        sundown = int(18*3600/self.settings.timeStep)
        dblHourMin = (np.argmin(aggreg_demand[sunrise:sundown])+sunrise+1)*self.settings.timeStep/3600
        dblHourMax = (np.argmax(aggreg_demand)+1)*self.settings.timeStep/3600
        
        return dblHourMin, dblHourMax    
    
    def setPVIrradValue(self,irrad):
        self.circuit.LoadShapes.Name = "PVdaily"
        self.circuit.LoadShapes.Pmult = np.full(self.circuit.LoadShapes.Npts, irrad)
     

    def find_critical_loads(self):
        ''' Find PQ buses to represent the distribution network inside Q-learning  
        
        Find using a low/null demans profile and a range of values for the injected 
        active power of each distributed generator, the first buses that will 
        present voltage problems 
  
        Return 
        ------
        - criticalPQbuses: list, 
            name of the loads (PQ buses) that will first sense the impact of 
            the distributed generation.
        
        '''
        def find_allowed_loads():
            '''Create a list of the allowed loads (loads which the peak demand is within the max and min allowed values)'''
            allowed_loads = []
            total_PVs_power = 0 #kW
            for lname in self.loads.AllNames:
               if '_b1' in lname or '_a' in lname and self.circuit.ActiveCktElement.Enabled:
                    allowed_loads.append(lname)
            
            return allowed_loads

        def voltage_violations():

            all_crit_loads = []
            for lname in self.circuit.Loads.AllNames:
                
                self.loads.Name = lname
                
                if not '_b1' in lname and not '_a' in lname or not self.circuit.ActiveCktElement.Enabled:
                    continue
                
                self.circuit.SetActiveBus(self.cktElement.BusNames[0])
                VmodArray = np.array(self.bus.puVmagAngle[::2])
                
                try:
                    if 1000*self.bus.kVBase <= self.settings.LVMVlimiar/(3**0.5):
                        if min(VmodArray[VmodArray > 0.1]) < self.settings.voltageLV['min']:# the 0.1 is to exclude the neutral node Vmag
                            all_crit_loads.append(lname)
                        elif max(VmodArray) > self.settings.voltageLV['max']:
                            all_crit_loads.append(lname)
                    else:
                        if min(VmodArray[VmodArray > 0.1]) < self.settings.voltageMV['min']:
                            all_crit_loads.append(lname)
                        elif max(VmodArray) > self.settings.voltageMV['max']:
                            all_crit_loads.append(lname)
                except:
                    None
                    
            return all_crit_loads
    
        self.solution.ControlMode = 0 #static
        dblHourMin, dblHourMax = self.getDblHour()
        self.solution.dblHour = dblHourMin
        critical_loads = []
        # undervoltage_cases = None    
        #Overvoltage
        for irrad in self.settings.pvIrradValues:
            self.setPVIrradValue(irrad)
            self.solution.SolveSnap()
            feedback = voltage_violations()
            if len(feedback) != 0:
                critical_loads += feedback
                break
        
        #Undervoltage
        self.solution.dblHour = dblHourMax
        self.circuit.RegControls.First
        self.circuit.ActiveCktElement.Enabled = False
        self.circuit.Transformers.Name = self.circuit.RegControls.Transformer
        min_tap_pu = self.circuit.Transformers.MinTap
        max_tap_pu = self.circuit.Transformers.MaxTap
        num_taps = self.circuit.Transformers.NumTaps
        all_taps_pu = np.linspace(min_tap_pu,max_tap_pu,num_taps+1)
        self.setPVIrradValue(0)
        
        for tap in all_taps_pu:
            self.circuit.Transformers.Tap = tap 
            self.solution.SolveSnap()
            feedback = voltage_violations()

            if len(feedback) == 0:
                critical_loads += undervoltage_cases[0] #-9
                break
            else:
                undervoltage_cases = feedback
        
        #*Overvoltage
        set_ld_kW(uc_kW, 0.001)
        overvoltage_cases = []
        self.solution.dblHour = 0#round(dblHourMin, 2)
        # self.text.Command = "Set mode=snapshot"
        # self.solution.ControlMode = 0
        # self.circuit.RegControls.First #! problem here with opendss version
        self.circuit.RegControls.Name = self.circuit.RegControls.Name
        self.circuit.ActiveCktElement.Enabled = True
        for irrad in self.settings.pvIrradValues:
            # self.setPVIrradValue(irrad)
            set_pv_kW(uc_kW, irrad) 
            self.solution.Solve()
            feedback = voltage_violations()
            if len(feedback) == 0:
                critical_loads += overvoltage_cases[0]
                break
            else:
                overvoltage_cases.append(feedback)

        if len(critical_loads) == 0:        
            logging.info('No critical loads found')
        else:
            critical_loads = list(set(critical_loads))
        
        critical_loads = find_allowed_loads()
        logging.info(f'Num critical loads:{len(critical_loads)}')

        with open(rf'{self.settings.outp_folder}/crit_lds_{self.PVPenetration}_pl.pickle', 'wb') as handle:
            pickle.dump(critical_loads, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return critical_loads    
    
    def action_space(self):
       '''Define the number of size of states and actions'''
       
       def reg_settings():
           all_names = self.circuit.RegControls.AllNames
           num_regs = self.circuit.RegControls.Count
           num_taps = np.zeros(num_regs, dtype=int)
           all_taps_pu = []
           for i,name in enumerate(all_names):
               
               self.circuit.RegControls.Name = name 
               self.circuit.Transformers.Name = self.circuit.RegControls.Transformer
               self.circuit.Transformers.Wdg = self.circuit.RegControls.TapWinding 
               min_tap_pu = self.circuit.Transformers.MinTap
               max_tap_pu = self.circuit.Transformers.MaxTap
               num_taps[i] = self.circuit.Transformers.NumTaps
               all_taps_pu.append(np.linspace(min_tap_pu,max_tap_pu,num_taps[i]+1))
               
           return num_regs,num_taps,all_taps_pu
             
       num_regs, num_taps, all_taps_pu  = reg_settings()
       
       # action_space_size = int(np.prod(num_taps))
       # state_space_size = self.num_states**len(self.critical_loads)
        
       #All possible actions 
       all_list_actions = [all_taps_pu[i] for i in range(num_regs)]
       actions = list(itertools.product(*all_list_actions))
       # all_list_states = [list(np.arange(0,self.num_states)) for i in range(len(self.critical_loads))]
       # states = list(itertools.product(*all_list_states))
       
       return np.sum(num_taps)+1, actions 
   

    def makeVsourceAnInfinityBus(self):
        '''Author: Pedro Pato'''

        self.circuit.Vsources.First # selects the circuit source as the active element

        oldBus = self.cktElement.Properties('bus1').Val
        r0_old = float(self.cktElement.Properties('r0').Val)
        x0_old = float(self.cktElement.Properties('x0').Val)
        r1_old = float(self.cktElement.Properties('r1').Val)
        x1_old = float(self.cktElement.Properties('x1').Val)

        # approximate the voltage source to an infinity bus (Zp = 0 + j0.0001, Zm = 0)
        self.cktElement.Properties('Bus1').Val = 'newSlackBus.1.2.3'
        self.cktElement.Properties('r0').Val = 0
        self.cktElement.Properties('x0').Val = str(0.0001)
        self.cktElement.Properties('r1').Val = 0
        self.cktElement.Properties('x1').Val = str(0.0001)

        #create a new line to represent the source old Thevenin Impedances
        self.text.Command = f'New Line.SourceLink Bus1=newSlackBus.1.2.3 bus2={oldBus} phases=3 C0=0 C1=0 Xg=0 rg=0'
        self.text.Command = f'r0={1e4*r0_old} x0={1e4*x0_old} r1={1e4*r1_old} x1={1e4*x1_old} length={1e-4}'

        self.text.Command = 'calc voltageBases'

        return
    
    def solution_settings(self, mode):

        ''' Define solution settings in OpenDSS'''

        self.text.Command = f"Edit Vsource.Source pu={self.settings.SourceVoltage}"
        self.solution.MaxControlIterations = 50
        self.solution.MaxIterations = 20
        # self.dssObj.ActiveCircuit.Settings.LoadsTerminalCheck = False
        self.text.Command = 'Reset'
        self.text.Command = "Set mode=daily"
        self.solution.StepSize = int(self.settings.timeStep/self.settings.simulationCycleinDays)
        self.solution.Number = 1
        self.solution.dblHour = 0
        self.solution.Tolerance = 5e-3
        #self.text.Command = "Set SampleEnergyMeters=NO" -- note: with our custom loop, value is not used
        self.text.Command = "Set DefaultBaseFreq=60"
        # self.text.Command = "Set ControlMode=time"
        self.solution.ControlMode = mode # 0: static, 1: time
         
    def findBusesWithLoads(self):

        '''Find b1 and a customers'''

        ldBuses = []

        for ld in self.circuit.Loads.AllNames:

            self.circuit.Loads.Name = ld

            if '_a' in ld or '_b1' in ld:
                ldBuses.append(self.circuit.ActiveCktElement.BusNames[0].split('.')[0])

        ldBuses = list(set(ldBuses))

        return ldBuses

    def getAggregateBusDemandGeneration(self):

        def getDemand():
            demand = {}
            for ld in self.circuit.Loads.AllNames:

                self.circuit.Loads.Name = ld

                busName = self.circuit.ActiveCktElement.BusNames[0].split('.')[0]

                if demand.get(busName, VERYLARGENUMBER) == VERYLARGENUMBER:
                    demand.update({f'{busName}': self.circuit.Loads.kva})
                else:
                    demand[f'{busName}']+= self.circuit.Loads.kva

            return demand

        def getGeneration():

            generation = {}
            for pv in self.circuit.PVSystems.AllNames:

                self.circuit.PVSystems.Name = pv

                busName = self.circuit.ActiveCktElement.BusNames[0].split('.')[0]

                if generation.get(busName, VERYLARGENUMBER) == VERYLARGENUMBER:
                    generation.update({f'{busName}': self.circuit.PVSystems.kVArated})
                else:
                    generation[f'{busName}']+= self.circuit.PVSystems.kVArated

            return generation

        return getDemand(), getGeneration()

    def install_pvs(self):
        
        ''' Deploy PV systems using Generator element '''

        def find_allowed_loads():
            '''Create a list of the allowed loads (loads which the peak demand is within the max and min allowed values)'''
            allowed_loads = []
            total_PVs_power = 0 #kW
            for load_name in self.loads.AllNames:
                if '_b1' in load_name or '_a' in load_name:
                    allowed_loads.append(load_name)
            
            return allowed_loads

        def install(chosen_loads, outputFile):
            """Install pvs (write .dss file)

            Args:
                chosen_loads (list): chosen points to install pvs
                outputFile ([type]): file that will describe pvs in opendss

            Returns:
                int: total pv installed power
            """
            total_PVs_power = 0
            for load_name in chosen_loads:
                self.loads.Name = load_name
                pv_name = 'PV_' + load_name.split("_")[-1]
                outputFile.write(f'New Load.{pv_name} ')
                outputFile.write(f'phases = {self.cktElement.NumPhases} ')
                outputFile.write(f'bus1 = {self.cktElement.BusNames[0]} ')
                outputFile.write(f'kV = {self.circuit.Loads.kV} ')
                outputFile.write(f'kW = {-self.circuit.Loads.kW/(0.2*0.8)} ') #! this is not realistic
                outputFile.write(f'PF={self.settings.pvsPowerFactor} Daily=PVdaily model=1 ')
                outputFile.write('vminpu=0.0 vmaxpu=3.0 \n')
                total_PVs_power += -self.circuit.Loads.kW
            
            return total_PVs_power
            
        outputFile=open(self.settings.network_folder + '\\PVs.dss','w')
        outputFile.write(f"New LoadShape.PVdaily Pmult=[]\n")        
        allowed_loads = find_allowed_loads()
        num_PVs_to_install = int(self.PVPenetration*len(allowed_loads)/100)
        chosen_loads = random.sample(allowed_loads, num_PVs_to_install)
        total_PVs_power = install(chosen_loads, outputFile)
        outputFile.close()
        self.text.Command = rf'Redirect {self.settings.network_folder}\\PVs.dss'
        logging.info(f'Total PV intalled capacity: {total_PVs_power} kW')
        
        return total_PVs_power

    
    def enable_pvs(self):

        ''' Deploy PV systems using Generator element '''

        def calc_pmpp():

            eff = 0.97
            temp_factor = 0.88
            pmpp = abs(self.circuit.Loads.kW)/(eff*temp_factor)

            return pmpp

        def change_to_pv_system_model():

              self.text.Command = f'New PVSystem.{self.circuit.Loads.Name}'
              self.text.Command = 'irrad = 1'
              self.text.Command = 'temperature = 25'
              self.text.Command = 'kvar=0'
              self.text.Command = r'%cutin = 0 '
              self.text.Command = r'%cutout = 0 '
              self.text.Command = 'effcurve = Myeff '
              self.text.Command = 'P-TCurve = MyPvsT '
              self.text.Command = f'Pmpp = {calc_pmpp()} '
            #  self.text.Command = 'pctPmpp = 100'
              self.text.Command = 'vmaxpu=1.2 '
              self.text.Command = 'vminpu=0.0 '
              self.text.Command = 'daily = PVdaily'
            #  self.text.Command = 'TDaily = MyTemp'
              self.text.Command = f'kVA = {abs(self.circuit.Loads.kW)} '
              self.text.Command = f'phases = {self.circuit.ActiveCktElement.NumPhases}'
              self.text.Command = f'bus1 = {self.circuit.ActiveCktElement.BusNames[0]}'
              self.text.Command = f'kV = {self.circuit.Loads.kV}'

        total_pv_capacity = 0 # Installed PV capacity of the scenario

        self.text.Command = "New LoadShape.PVdaily Pmult=[]\n"
        self.text.Command = "New XYCurve.MyPvsT npts=4  xarray=[0  25  75  100] yarray=[1.0 1.0 1.0 1.0]"
        self.text.Command = "New XYCurve.MyEff npts=4  xarray=[.1 .2 .4 1.0]  yarray=[1.0 1.0 1.0 1.0]"
        self.text.Command = f"New Tshape.MyTemp npts=5760 sinterval={self.settings.timeStep} csvfile={self.settings.network_folder}\\Temperature_shape.csv"

        if self.PVPenetration == 0:
            return total_pv_capacity

        # Include all PVs (enabled=False)
        self.text.Command = rf"Redirect {self.settings.ls_folder}\scripts/_____generators_load_model/Generators_disabled_{self.Transformer}.dss"
        generator_all_names = set([x for x in self.circuit.Loads.AllNames if x[:2] == 'pv'])
        loads_all_names = []
        loads_all_classes = []
        loads_all_buses = []
        for item in self.circuit.Loads.AllNames:
            if item[:2] == 'pv': continue
            self.circuit.Loads.Name = item
            loads_all_names.append(item.split('_')[-1])
            loads_all_classes.append(item.split('_')[-2])
            loads_all_buses.append(self.circuit.ActiveCktElement.BusNames[-1])

        loads_all_buses = dict(zip(loads_all_names, loads_all_buses))
        loads_all_classes = dict(zip(loads_all_names, loads_all_classes))

        for item in self.PVs:
            flag_disable_pv = False
            item = f'{int(item):010d}'
            if f"pv{item}" in generator_all_names and f"{item}" in loads_all_names:
                self.circuit.Loads.Name = f"PV{item}"
                pv_bus = self.circuit.ActiveCktElement.BusNames[-1]
                load_bus = loads_all_buses[item]
                if pv_bus.split('.')[0] == load_bus.split('.')[0]:
                    for pv_phase in pv_bus.split('.')[1:]:
                        if pv_phase not in load_bus.split('.')[1:]:
                            flag_disable_pv = True
                            break
                    if flag_disable_pv or self.circuit.Loads.kW >= self.settings.limitResidentialPV:# and loads_all_classes[item] == 'res':
                        self.circuit.ActiveCktElement.Enabled = False
                        if flag_disable_pv:
                            logging.warning(rf"O SISTEMA FV {item} NÃO FOI HABILITADO POIS SUAS FASES ESTÂO EM DESACORDO COM AS FASES DA CARGA REPRESENTANDO O CONSUMIDOR CORRESPONDENTE.")
                        else:
                            logging.warning(rf"O SISTEMA FV {item} NÃO FOI HABILITADO POIS TRATA-SE DE SISTEMA {loads_all_classes[item].upper()} COM CAPACIDADE SUPERIOR A {self.settings.limitResidentialPV} kW.")
                    else:
                        # change_to_pv_system_model()
                        self.circuit.ActiveCktElement.Enabled = True
                        # total_pv_capacity += self.circuit.PVSystems.kVArated
                        self.circuit.Loads.kW = self.circuit.Loads.kW*8
                        total_pv_capacity += self.circuit.Loads.kW
                        # self.text.Command = f"New Monitor.MonitorPV{item}PQ Element=Load.PV{item} Terminal=1 Mode=1 ppolar=0"
            else:
                logging.warning(rf"O SISTEMA FV {item} NÃO FOI HABILITADO POIS A CARGA REPRESENTANDO O CONSUMIDOR CORRESPONDENTE NÃO FOI LOCALIZADA.")


        self.text.Command = rf"New LoadShape.PVdaily Pmult=[1] Qmult=[0]"
        self.text.Command = rf"BatchEdit Load.pv.* PF={self.settings.pvsPowerFactor} Daily=PVdaily"

        return total_pv_capacity

    def set_loadshapes_PVs(self, cycle):

        # Set irrandiance loadshape
        irrad = self.store_PV

        temp = np.diff(self.store_PV.index.values)/1e9

        pv_resolution_in_seconds = list(set(temp))

        if len(pv_resolution_in_seconds) != 1:
            logging.error("Os perfis de geração FV utilizados não possuem intervalos regulares...")
        else:
            pv_resolution_in_seconds = pv_resolution_in_seconds[0]

        d0 = rf"{(self.settings.d0).replace('2018', '2018').replace('2017', '2017')}"
        d1 = pd.to_datetime(rf"{(self.settings.d1).replace('2018', '2018').replace('2017', '2017')}") + pd.Timedelta("1 day")
        d0_cycle = pd.to_datetime(d0) + pd.Timedelta(rf"{int(cycle * self.settings.simulationCycleinDays)} days")
        d1_cycle = d0_cycle + pd.Timedelta(rf"{self.settings.simulationCycleinDays} days")

        if d1 < d1_cycle:
            d1_cycle = d1 - pd.Timedelta("1 day")
            npts = int((d1_cycle - d0_cycle).days * 86400 / self.settings.timeStep)
        else:
            npts = int(self.settings.simulationCycleinDays * 86400 / self.settings.timeStep)

        irrad = irrad[d0_cycle.strftime("%Y-%m-%d %H:%M:%S") : d1_cycle.strftime("%Y-%m-%d %H:%M:%S")]
        if self.settings.timeStep != pv_resolution_in_seconds:
            irrad = irrad.resample(f"{self.settings.timeStep}s").interpolate(method='linear')
        irrad = irrad.values[0:-1]
        irrad = np.round(irrad,4)
        irrad = [x[0] if x>0 else 0 for x in irrad]

        self.circuit.LoadShapes.Name = "PVdaily"
        self.circuit.LoadShapes.Npts = npts
        self.circuit.LoadShapes.Sinterval = self.settings.timeStep
        # self.circuit.LoadShapes.sInterval = self.settings.timeStep
        self.circuit.LoadShapes.Pmult = irrad
        # self.circuit.LoadShapes.Pmult = [0.547]*npts


    def set_loadshapes_B1(self, cycle):

        logging.info(f"Início da alocação das curvas de carga do grupo B1 (ciclo {cycle})...")

        # if cycle == 0:
        self.text.Command = f"Compile {self.settings.ls_folder}/LoadShape_B1.dss"

        loads_with_synthetic_curve = [rf"_b1_{str(item).zfill(10)}" for item in self.LoadShapeResidentialLV.index.tolist()]

        d0_cycle = pd.to_datetime(self.settings.d0) + pd.Timedelta(rf"{int(cycle * self.settings.simulationCycleinDays)} days")
        d1_cycle = d0_cycle + pd.Timedelta(rf"{self.settings.simulationCycleinDays} days")
        d1 = pd.to_datetime(self.settings.d1) + pd.Timedelta("1 day")

        if d1 < d1_cycle:
            d1_cycle = d1
            days = (d1_cycle - d0_cycle).days
        else:
            days = self.settings.simulationCycleinDays

        if d0_cycle == d1_cycle:
            days = 1

        d0_range = d0_cycle.dayofyear - 1
        if d1_cycle.strftime('%Y-%m-%d') == '2019-01-01':
            d1_range = d1_cycle.dayofyear + 365 - 1
        else:
            d1_range = d1_cycle.dayofyear - 1

        npts = int(days * 86400 / self.settings.timeStep)
        sinterval = int(self.settings.timeStep)

        for lname in self.circuit.Loads.AllNames:

            self.circuit.Loads.Name = lname

            lkV = self.circuit.Loads.kV

            skip = lname not in loads_with_synthetic_curve or lkV == 0.115 or lkV == 0.230

            if skip:
                continue

            lsname = self.LoadShapeResidentialLV.loc[lname.split('_')[-1]]
            house, nphases, faixa = lsname

            daily = rf"faixa_{faixa}___phases_{nphases}___house{house}"

            Ploadshape = []
            Qloadshape = []
            if d1_range == d0_range:
                d1_range += 1
            for day in range(d0_range, d1_range):
                Pkey = rf"/fase_{nphases}/faixa_{faixa}/P/dia_{day}/casa_{house}"
                Qkey = rf"/fase_{nphases}/faixa_{faixa}/Q/dia_{day}/casa_{house}"
                Ploadshape.append(np.asarray(self.store_B1[Pkey]))
                Qloadshape.append(np.asarray(self.store_B1[Qkey]))

            Ploadshape = np.concatenate(Ploadshape)
            Qloadshape = np.concatenate(Qloadshape)

            if len(Ploadshape) < int(days * 86400 / self.settings.timeStep):
                logging.error("ATENÇÃO: AS CURVAS SINTÉTICAS DOS CONSUMIDORES B1 POSSUEM RESOLUÇÃO MAIOR DO QUE O PASSO DE SIMULAÇÃO E O PROCESSO SERÁ INTERROMPIDO.")
                sys.exit()
            elif len(Ploadshape) > int(days * 86400 / self.settings.timeStep):
                if cycle == 0:
                    logging.warning("ATENÇÃO: AS CURVAS SINTÉTICAS DOS CONSUMIDORES B1 POSSUEM RESOLUÇÃO MENOR DO QUE O PASSO DE SIMULAÇÃO. PARA MELHOR DESEMPENHO, UTILIZAR OUTRO CONJUNTO DE CURVAS B1.")
                needs_reshape = True
            else:
                needs_reshape = False

            self.circuit.Loads.Name = lname
            self.circuit.Loads.kW = 1.0
            self.circuit.Loads.kvar = 1.0
            self.circuit.Loads.daily = daily

            self.circuit.LoadShapes.Name = daily
            self.circuit.LoadShapes.Npts = npts
            # self.circuit.LoadShapes.sInterval = sinterval
            self.circuit.LoadShapes.Sinterval = sinterval

            if needs_reshape:
                current_resolution_P = 86400 * days / len(Ploadshape)
                current_resolution_Q = 86400 * days / len(Qloadshape)
                self.circuit.LoadShapes.Pmult = np.reshape(Ploadshape, (int(self.settings.timeStep / current_resolution_P), int(days * 86400 / self.settings.timeStep)), order='F').mean(axis=0)
                self.circuit.LoadShapes.Qmult = np.reshape(Qloadshape, (int(self.settings.timeStep / current_resolution_Q), int(days * 86400 / self.settings.timeStep)), order='F').mean(axis=0)
            else:
                self.circuit.LoadShapes.Pmult = Ploadshape
                self.circuit.LoadShapes.Qmult = Qloadshape

        #self.store_B1.close()

        logging.info(f"Fim da alocação das curvas de carga do grupo B1 (ciclo {cycle})...")

        return loads_with_synthetic_curve


    def set_loadshapes_A(self, cycle):

        logging.info(f"Início da alocação das curvas de carga do grupo A (ciclo {cycle})...")

        loads_all = self.circuit.Loads.AllNames
        sinterval = int(self.settings.timeStep)
        index = 1

        # if cycle == 0:
        for lname in loads_all:
            self.circuit.Loads.Name = lname
            load = self.circuit.Loads
            if lname[0:2] == '_a':
                #logging.info(f"Criando o loadshape do consumidor do grupo A {lname}...")
                self.text.Command = f"New Loadshape.{lname} Npts=1 Mult=(1)"
                load.kW = 1
                load.kvar = 1
                load.daily = lname

        d0_cycle = pd.to_datetime(self.settings.d0) + pd.Timedelta(rf"{int(cycle * self.settings.simulationCycleinDays)} days")
        d1_cycle = d0_cycle + pd.Timedelta(rf"{self.settings.simulationCycleinDays} days")
        d1 = pd.to_datetime(self.settings.d1) + pd.Timedelta("1 day")
        if d1 < d1_cycle:
            d1_cycle = d1
            npts = int((d1_cycle - d0_cycle).days * 86400 / self.settings.timeStep)
        else:
            npts = int(self.settings.simulationCycleinDays * 86400 / self.settings.timeStep)
        d0_str = d0_cycle.strftime("%Y-%m-%d %H:%M:%S")
        d1_str = d1_cycle.strftime("%Y-%m-%d %H:%M:%S")

        tt = pd.to_datetime(np.asarray(self.store_A[rf"TIME"]), unit='s')
        for lname in loads_all:

            self.circuit.Loads.Name = lname
            load = self.circuit.Loads

            if load.kV < 1:
                continue

            uc = int(lname.split('_')[-1])
            if not uc in self.keys_A.index:
                continue

            key_uc = self.keys_A['keys'].loc[uc]
            pshape = np.asarray(self.store_A[key_uc + rf"/P"])
            qshape = np.asarray(self.store_A[key_uc + rf"/Q"])
            #tt = tt + pd.Timedelta('2 hours')
            #tt = tt.tz_localize(pytz.utc)
            loadshape = pd.DataFrame(np.transpose([pshape, qshape]), index=tt, columns=['P (kW)', 'Q (kvar)'])
            pshape = loadshape['P (kW)'][d0_str : d1_str]
            qshape = loadshape['Q (kvar)'][d0_str : d1_str]
            pshape = pshape.resample(f"{self.settings.timeStep}s").interpolate(method='linear')
            qshape = qshape.resample(f"{self.settings.timeStep}s").interpolate(method='linear')
            self.circuit.LoadShapes.Name = load.daily
            self.circuit.LoadShapes.Npts = npts
            # self.circuit.LoadShapes.sInterval = sinterval
            self.circuit.LoadShapes.Sinterval = sinterval
            self.circuit.LoadShapes.Pmult = pshape.values[0:-1]
            self.circuit.LoadShapes.Qmult = qshape.values[0:-1]
            index = index + 1

        #self.store_A.close()

        logging.info(f"Fim da alocação das curvas de carga do grupo A (ciclo {cycle})...")


    def set_loadshapes_B3(self, cycle):

        logging.info(f"Início da alocação das curvas de carga do grupo B3 (ciclo {cycle})...")

        d0_cycle = pd.to_datetime(self.settings.d0) + pd.Timedelta(rf"{int(cycle * self.settings.simulationCycleinDays)} days")
        d1_cycle = d0_cycle + pd.Timedelta(rf"{self.settings.simulationCycleinDays} days")
        d1 = pd.to_datetime(self.settings.d1)# + pd.Timedelta("1 day")

        if d1 < d1_cycle:
            d1_cycle = d1
            days = (d1_cycle - d0_cycle).days
        else:
            days = self.settings.simulationCycleinDays

        if d0_cycle == d1_cycle:
            days = 1

        d0_range = d0_cycle.dayofyear - 1
        if d1_cycle.strftime('%Y-%m-%d') == '2019-01-01':
            d1_range = d1_cycle.dayofyear + 365 - 1
        else:
            d1_range = d1_cycle.dayofyear - 1

        npts = int(days * 86400 / self.settings.timeStep)
        sinterval = int(self.settings.timeStep)

        # if cycle == 0:
        logging.info(f"Criando o loadshape do consumidor do grupo B não 1...")
        self.text.Command = f"New Loadshape.loadshape_b_not_1 Npts=1 Mult=(1)"

        Ploadshape = []
        #Qloadshape = []
        if d1_range == d0_range:
            d1_range += 1
        for day in range(d0_range, d1_range):

            Ploadshape.append(np.asarray(self.store_B3[f"{day}"]))
            #Qloadshape.append(np.asarray(self.store_B3[f"{day}"]))

        Ploadshape = np.concatenate(Ploadshape)
        #Qloadshape = np.concatenate(Qloadshape)

        if len(Ploadshape) < int(days * 86400 / self.settings.timeStep):
            logging.error("ATENÇÃO: AS CURVAS SINTÉTICAS DOS CONSUMIDORES B não 1 POSSUEM RESOLUÇÃO MAIOR DO QUE O PASSO DE SIMULAÇÃO E O PROCESSO SERÁ INTERROMPIDO.")
            sys.exit()
        elif len(Ploadshape) > int(days * 86400 / self.settings.timeStep):
            if cycle == 0:
                logging.warning("ATENÇÃO: AS CURVAS SINTÉTICAS DOS CONSUMIDORES B não 1 POSSUEM RESOLUÇÃO MENOR DO QUE O PASSO DE SIMULAÇÃO. PARA MELHOR DESEMPENHO, UTILIZAR OUTRO CONJUNTO DE CURVAS B1.")
            needs_reshape = True
        else:
            needs_reshape = False

        self.circuit.LoadShapes.Name = 'loadshape_b_not_1'
        self.circuit.LoadShapes.Npts = npts
        # self.circuit.LoadShapes.sInterval = sinterval
        self.circuit.LoadShapes.Sinterval = sinterval

        if needs_reshape:
            current_resolution_P = 86400 * days / len(Ploadshape)
            #current_resolution_Q = 86400 * days / len(Qloadshape)
            self.circuit.LoadShapes.Pmult = np.reshape(Ploadshape, (int(self.settings.timeStep / current_resolution_P), int(days * 86400 / self.settings.timeStep)), order='F').mean(axis=0)
            #self.circuit.LoadShapes.Qmult = np.reshape(Qloadshape, (int(self.settings.timeStep / current_resolution_Q), int(days * 86400 / self.settings.timeStep)), order='F').mean(axis=0)
        else:
            self.circuit.LoadShapes.Pmult = Ploadshape
            #self.circuit.LoadShapes.Qmult = Qloadshape

        for lname in self.circuit.Loads.AllNames:

            lname = self.circuit.Loads.Name
            load = self.circuit.Loads

            if '_b' not in lname or '_b1' in lname:
                continue

            load.daily = 'loadshape_b_not_1'

        #self.store_B3.close()

        logging.info(f"Fim da alocação das curvas de carga do grupo B3 (ciclo {cycle})...")

    
    def set_loadshapes(self, cycle):

        # if cycle == 0:
        # Curvas originais interpoladas
        npts = int(self.settings.simulationCycleinDays * 86400 / self.settings.timeStep)
        sinterval = int(self.settings.timeStep)
        ts = pd.date_range(start='2018-01-01', periods=96*self.settings.simulationCycleinDays+1, freq='15T')
        for item in self.circuit.LoadShapes.AllNames:
            if 'media' not in item:
                continue
            self.circuit.LoadShapes.Name = item
            temp = self.circuit.LoadShapes.Pmult
            pmult = np.tile(temp, self.settings.simulationCycleinDays)
            pmult = np.hstack((pmult, pmult[0]))
            qmult = pmult * np.tan(np.arccos(0.85))
            pmult = pd.Series(pmult, index=ts).resample(f"{self.settings.timeStep}s")
            qmult = pd.Series(qmult, index=ts).resample(f"{self.settings.timeStep}s")
            pmult = pmult.interpolate(method='linear')
            qmult = qmult.interpolate(method='linear')
            self.circuit.LoadShapes.Npts = npts
            self.circuit.LoadShapes.sInterval = sinterval
            self.circuit.LoadShapes.Pmult = pmult.values[0:-1]
            self.circuit.LoadShapes.Qmult = qmult.values[0:-1]

        self.set_loadshapes_A(cycle)
        if self.settings.enableSyntheticLoadShapesB1:
            self.set_loadshapes_B1(cycle)
        self.set_loadshapes_B3(cycle)
        self.set_loadshapes_PVs(cycle)

    def reset_env(self):
        """Fresh restart
        """
        self.dssObj.Reset
        return

    def set_env(self, cycle, mode):
        self.set_loadshapes(cycle)
        self.solution_settings(mode)
        return

    def solveOneMomentInTime(self,episode):
        '''
        Soluciona o fluxo de carga para um momento do dia

        :param float hour: hora do dia a ser simulada
        '''

        hour = episode*self.settings.timeStep/3600
        self.text.Command = "Set maxcontroliter=150"
        self.text.Command = "Set maxiterations=300"
        self.text.Command = "Set ControlMode=time"
        self.text.Command = "Reset"
        self.text.Command = f'Set Mode=daily number=1 stepSize={self.settings.timeStep}'
        self.solution.dblHour = hour
#        self.solution.Hour = hour
        self.text.Command = "solve"

    def store_vreg(self):
        """save original vreg before training one episode

        Returns:
            list: original vreg values
        """
        allNames = self.regControls.AllNames
        vreg = []
        for i,name in enumerate(allNames):
            self.regControls.Name = name
            vreg.append(self.regControls.ForwardVreg)
            
        return vreg

    def restore_vreg(self, vreg):
        """restore original vreg after training one episode

        Args:
            vreg (list): original vreg values
        """
        allNames = self.regControls.AllNames
        for i,name in enumerate(allNames):
            self.regControls.Name = name
            self.regControls.ForwardVreg = vreg[i]
            

    def update_controls(self, i, train, training_process, action):

        def solve(training_process):

             if training_process == 'on':

                 self.solution.Solve()

             elif training_process == 'off':

                   self.solution.SolveSnap()
                  # self.solution.SolvePlusControl()

        def update_reg_controls():

            allNames = self.regControls.AllNames

            for i,name in enumerate(allNames):

                self.regControls.Name = name

                # if GND in nodes:
                #     nodes.remove(0)
                # if self.dss.neutralNum in nodes:
                #     nodes.remove(self.dss.neutralNum)

                self.circuit.SetActiveBus(self.cktElement.BusNames[0])
                basekV = self.bus.kVBase*1000
                ptRatio  = self.regControls.PTratio

                tapPu = self.actions[action][i]
                # tapStep = action[0]-4

                vreg  = tapPu*basekV/ptRatio

                # self.regControls.TapNumber = int(tapStep)
                self.regControls.ForwardVreg = round(vreg, 3)

        update_reg_controls()
        solve(training_process)
        # new_state, reward , done, voltages = self.dss_feedback(buses)

        return self.dss_feedback(i, train)
    
    def check_all_voltages(self): #TODO FIX THIS FUNCTION
        # for lname in self.circuit.Loads.AllNames:
        for lname in self.critical_loads:
            self.loads.Name = lname
            
            # # if not '_b1' in lname and not '_a' in lname or not self.circuit.ActiveCktElement.Enabled:
            # if not ('_b1' in lname or '_a' in lname and self.circuit.ActiveCktElement.Enabled):
            #     continue
            
            self.circuit.SetActiveBus(self.cktElement.BusNames[0])
            VmodArray = np.array(self.bus.puVmagAngle[::2])
            
            if 1000*self.bus.kVBase <= self.settings.LVMVlimiar/(3**0.5):
                if min(VmodArray[VmodArray > 0.1]) < self.settings.voltageLV['min']:# the 0.1 is to exclude the neutral node Vmag
                    return True
                elif max(VmodArray) > self.settings.voltageLV['max']:
                    return True
            else:
                if min(VmodArray[VmodArray > 0.1]) < self.settings.voltageMV['min']:
                    return True
                elif max(VmodArray) > self.settings.voltageMV['max']:
                    return True
            
            return False

        
    
    def dss_feedback(self, i, train):
        """Feeback to control's actions in the distribution network"""
        # overvoltage = np.array([1.05, 1.06, 1.07, 1.08, 1.09, 1.10]) # 1,2,3
        # undervoltage = np.array([0, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93]) 
        # noproblem = 0
        
        def calc_reward(i, voltage_reward, all_volt):
            nv = self.settings.nv[i]
            c0 = self.settings.c0[i]

            if self.settings.vs[i] == 'r0':
                return nv*voltage_reward + c0*1/(np.sum(np.power(np.array(all_volt)-1,2))) #r0
            elif self.settings.vs[i] == 'r1':
                return nv*voltage_reward - c0*(np.sum(np.power(np.array(all_volt)-1,2))) #r1
            elif self.settings.vs[i] == 'r2':
                return -(np.sum(np.power(np.array(all_volt)-1,2))) #r2
            elif self.settings.vs[i] == 'r3':
                return -np.sum(np.abs(np.array(state))) #r3
            elif self.settings.vs[i] == 'r4':
                return nv*-np.sum(np.abs(np.array(state))) - c0*(np.sum(np.power(np.array(all_volt)-1,2))) #r4
    
        def check_critical_voltages(): #TODO: use monitors here! Find a way to reduce time!
            state = []
            all_crit_volt = []

            for lname in self.critical_loads:

                        self.loads.Name = lname
                        self.circuit.SetActiveBus(self.cktElement.BusNames[0])
                        VmodArray = np.array(self.bus.puVmagAngle[::2])

                        if 1000*self.bus.kVBase <= self.settings.LVMVlimiar/(3**0.5):
                            if min(VmodArray[VmodArray > 0.1]) < self.settings.voltageLV['min']:# the 0.1 is to exclude the neutral node Vmag
                                state.append(min(VmodArray[VmodArray > 0.1]))
                            elif max(VmodArray) > self.settings.voltageLV['max']:
                                state.append(max(VmodArray))
                            else:
                                state.append(np.average(VmodArray[VmodArray > 0.1]))

                        else:
                            if min(VmodArray[VmodArray > 0.1]) < self.settings.voltageMV['min']:
                                state.append(min(VmodArray[VmodArray > 0.1]))
                                
                            elif max(VmodArray) > self.settings.voltageMV['max']:
                                state.append(max(VmodArray))

                            else:
                                state.append(np.average(VmodArray[VmodArray > 0.1]))
                                
                        all_crit_volt.extend(VmodArray)

            return state, all_crit_volt
        
        
        problems = self.check_all_voltages()
        
        if train:
            state, all_crit_volt, critical_problems = check_critical_voltages()
            state.extend(self.settings.voltage_scalers)
            state = np.reshape(state, [1, len(state)])
            scaler = MinMaxScaler()
            scaled_state = scaler.fit_transform(state.transpose())
            scaled_state = scaled_state[:-2]
            scaled_state = scaled_state.transpose()
    
    
            # if problems != critical_problems: 
            #     logging.warning('Critical customers are not capable to model the network behavior!')
    
            if critical_problems: #! See why problems is different from critical_problems
                reward = calc_reward(i, 0, all_crit_volt)
                noviol = False 
            else:
                reward = calc_reward(i, 1, all_crit_volt)
                noviol = True
     
            return scaled_state, reward, noviol #, viol
        
        else:
            if problems:
                reward = 0
                noviol = False
            else:
                reward = self.settings.nv[i]
                noviol = True 
                
            return None, reward, noviol
        
    def get_kvbases(self):
        
        kVBases = []
        
        for lname in self.loads.AllNames:
                
                    self.loads.Name = lname
                    
                    self.circuit.SetActiveBus(self.circuit.ActiveCktElement.BusNames[0])
        
                    if '_b1' in lname or '_a' in lname and self.circuit.ActiveCktElement.Enabled:
                           
                        kVBases.append(self.bus.kVBase)
                        
        return kVBases                

    def get_cycle_info(self, h5file, kVBases):

        ''' Used to obtain viol, voltages and control actions during online training'''
      
        def is_nrp_under(df, kVBase):
            
            if 1000*kVBase <= self.settings.LVMVlimiar/(3**0.5):
                df2 = (df>self.settings.voltageLV['nrp_under']['min']) & (df<self.settings.voltageLV['nrp_under']['max']) & (df>0.5)
           
            else:
                df2 = (df>self.settings.voltageMV['nrp_under']['min']) & (df<self.settings.voltageMV['nrp_under']['max']) & (df>0.5)
            
            viol = df2[0] | df2[1] | df2[2]
            return np.count_nonzero(viol)

        def is_nrp_over(df, kVBase):
        
            if 1000*kVBase <= self.settings.LVMVlimiar/(3**0.5):
                df2 = (df>self.settings.voltageLV['nrp_over']['min']) & (df<self.settings.voltageLV['nrp_over']['max']) & (df>0.5)
            
                viol = df2[0] | df2[1] | df2[2]
                return np.count_nonzero(viol)
            
            else:
                return 0
        def is_nrc_under(df, kVBase):
            
            if 1000*kVBase <= self.settings.LVMVlimiar/(3**0.5):
                df2 = (df < self.settings.voltageLV['nrc_under']) & (df > 0.5)
          
            else:
                df2 = (df < self.settings.voltageMV['nrc_under']) & (df > 0.5)
            
            viol = df2[0] | df2[1] | df2[2]
            
            return np.count_nonzero(viol)
        
        def is_nrc_over(df, kVBase):
            
            if 1000*kVBase <= self.settings.LVMVlimiar/(3**0.5):
                df2 = (df > self.settings.voltageLV['nrc_over']) & (df>0.5)
          
            else:
                df2 = (df > self.settings.voltageMV['nrc_over']) & (df>0.5)
             
            viol = df2[0] | df2[1] | df2[2]
            return np.count_nonzero(viol)
          #-------------------------------------------------------------------
          
        keys = list(h5file.keys())
        try:
            keys.remove('VIOLATIONS')
        except:
            None 
            
        viol = np.zeros((len(keys), h5file[keys[0]]['VOLTAGES'].shape[0], 4))
            
        for i,cycle in enumerate(keys):
            
            logging.info(f'Viol in cycle {cycle}')
             
            ild = 0 
            for kVBase in kVBases:
                
                VmodArray = pd.DataFrame(np.array(h5file[f'{cycle}']['VOLTAGES'][ild])/(1000*kVBase))
                viol[i][ild][0] = is_nrp_under(VmodArray, kVBase)
                viol[i][ild][1] = is_nrp_over(VmodArray,  kVBase)
                viol[i][ild][2] = is_nrc_under(VmodArray, kVBase)
                viol[i][ild][3] = is_nrc_over(VmodArray,  kVBase)
                
                ild +=1
                        
        return viol

    def putMonitors(self):
        
        iload = self.loads.First
        while iload>0:
            
            lname = self.loads.Name
            
            if '_b1' in lname or '_a' in lname and self.circuit.ActiveCktElement.Enabled:

                self.text.Command = 'New Monitor.'+lname+'m0 element=Load.'+lname+' terminal=1 mode=0 vipolar=yes'
            
            iload = self.loads.Next
        
        self.circuit.RegControls.First
        txname = self.circuit.RegControls.Transformer
        self.text.Command = 'New Monitor.'+txname+'m2 element=Transformer.'+txname+' terminal=2 mode=2' 
            
            
    def extractMonitorData(self):
            
        ldData = []
        tdData = []
        imon= self.circuit.Monitors.First
        while imon > 0:
            element = self.circuit.Monitors.Element
            typeElement, nameElement =  element.split(".")
            if typeElement == "load":
                self.circuit.Loads.Name = nameElement
                phases = self.circuit.Loads.Phases
                self.circuit.SetActiveBus(self.cktElement.BusNames[0])
                kVBase = self.bus.kVBase
                n = self.circuit.Monitors.SampleCount
                d = self.circuit.Monitors.ByteStream
                idata = np.frombuffer(d[0:16], dtype=np.uint8).view(np.int32)
                nrec = idata[2]
                sdata = np.frombuffer(d[272:], dtype=np.uint8).view(np.single) # could be defined as the times the circuit was solved
                y = np.reshape(sdata, (n, nrec+2))
                y = np.delete(y,[0,1],axis=1)
                header = [x.strip(' ') for x in self.circuit.Monitors.Header]
                dy = pd.DataFrame(y, columns=header)
                voltages = dy[[f'V{p+1}' for p in range(phases)]]/(kVBase*1000)
                varray = np.full((self.circuit.Monitors.SampleCount, 3), np.nan)
                varray[:,:phases] = voltages
                ldData.append(varray)

            elif typeElement == "transformer":
                n = self.circuit.Monitors.SampleCount
                d = self.circuit.Monitors.ByteStream
                idata = np.frombuffer(d[0:16], dtype=np.uint8).view(np.int32)
                nrec = idata[2]
                sdata = np.frombuffer(d[272:], dtype=np.uint8).view(np.single) # could be defined as the times the circuit was solved
                y = np.reshape(sdata, (n, nrec+2))
                y = np.delete(y,[0,1],axis=1)
                tdData=y.transpose()[0]

            imon = self.circuit.Monitors.Next
        
        da = np.array(ldData)
        cycle_max_voltages = np.nanmax(np.amax(da,axis=2),axis=0)
        cycle_min_voltages = np.nanmin(np.amin(da,axis=2),axis=0)
        self.circuit.Monitors.ResetAll() #TODO: check if this is needed when using 'Reset' command! 

        return da, cycle_max_voltages, cycle_min_voltages, tdData
