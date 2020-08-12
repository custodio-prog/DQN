from collections import namedtuple

BaseSettings = namedtuple('Settings', 'networkName, neutralNum, Sbase, ZIP_P,ZIP_Q,enableCapacitors,enableCapControls,enableRegControls,timeStep, d0, d1, days,simulationCycleinDays,SourceVoltage,loadsPowerFactor,pvsPowerFactor,enableSyntheticLoadShapesB1,limitResidentialPV,bandVR,OnOffCap,customMeter, hdf_file_cuca_a,hdf_file_cuca_a_keys,hdf_file_b1,hdf_file_b3,pv_shapes_file,ls_folder, volt_var_folder, task_folder, network_folder,outp_folder,network_version,master,simStartDblHour,voltageLV,voltageMV,LVMVlimiar,numQlStates,criticalZones,pvIrradValues,criticalLoads,MaxQlIterByTimeStep,learningRate,discountRate,explorationRate, minExplorationRate, maxExplorationRate, explorationDecayRate,ratioTrainTest, qlagent_check_frequency, episolon_estrategy, find_critical_loads, offline_t, online_t, no_control, vs, all_vs, nv, c0')

class Settings(BaseSettings):
    def fill_days(self):
        settings2 = self._asdict()
        settings = Settings(**settings2)
        return settings

