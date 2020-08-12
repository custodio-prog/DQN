# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 17:41:27 2020

@author: custodio
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pandas as pd 
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib as jl
import matplotlib.patheffects as pe
import matplotlib
import logging 
import h5py
import dask

sns.set(style="darkgrid", palette="colorblind", color_codes=True)

# x_emin = [0.6,0.7,0.8,0.9,1]
# x_edacay = [0.1,0.01,0.001,0.0001]
# x_maxrepet = [50,100,150,200,250,1000]

# y_emin = [11.45,11.45,15.625,16.66,17.7]
# y_emin_t = [10.13,9.33,9.44,8.89,7.91]
# y_edacay = [8.33,11.45,17.7,17.7]
# y_edacay_t = [7.91,9.33,9.71,9.93]
# y_maxrepet = [14.58,8.33,6.25,2.083,1.04,0]
# y_maxrepet_t = [7.91,18.85,22.82,33.2,35.31,52.45]

# fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(9,5))

# ax1.plot(x_emin,y_emin)
# ax1.set(ylabel='voltage transgressions (%)', xlabel='$\epsilon$ min')
# # ax.set_ylim([0.90, 1.1])
# ax1.set_ylim([0, 20])

# ax2.plot(x_edacay,y_edacay)
# ax2.set(ylabel='voltage transgressions (%)', xlabel='$\epsilon$ decay')
# # ax.set_ylim([0.90, 1.1])
# ax2.set_ylim([0, 20])

# ax3.plot(x_maxrepet,y_maxrepet)
# ax3.set(ylabel='voltage transgressions (%)', xlabel='max number of iterations')
# # ax.set_ylim([0.90, 1.1])
# ax3.set_ylim([0, 20])

# ax1.tick_params(direction='in',color='#595959', grid_alpha=0.5)
# ax2.tick_params(direction='in',color='#595959', grid_alpha=0.5)
# ax3.tick_params(direction='in',color='#595959', grid_alpha=0.5)
# ax1.grid('True')
# # ax1.tight_layout()
# ax2.grid('True')
# # ax2.tight_layout()
# ax3.grid('True')
# # ax3.tight_layout()
# plt.tight_layout()


# fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(9,5))

# ax1.plot(x_emin,y_emin_t)
# ax1.set(ylabel='time (s)', xlabel='$\epsilon$ min')
# # ax.set_ylim([0.90, 1.1])
# ax1.set_ylim([0, 60])


# ax2.plot(x_edacay,y_edacay_t)
# ax2.set(ylabel='time (s)', xlabel='$\epsilon$ decay')
# # ax.set_ylim([0.90, 1.1])
# ax2.set_ylim([0, 60])

# ax3.plot(x_maxrepet,y_maxrepet_t)
# ax3.set(ylabel='time (s)', xlabel='max number of iterations')
# # ax.set_ylim([0.90, 1.1])
# ax3.set_ylim([0, 60])

# ax1.tick_params(direction='in',color='#595959', grid_alpha=0.5)
# ax2.tick_params(direction='in',color='#595959', grid_alpha=0.5)
# ax3.tick_params(direction='in',color='#595959', grid_alpha=0.5)
# ax1.grid('True')
# # ax1.tight_layout()
# ax2.grid('True')
# # ax2.tight_layout()
# ax3.grid('True')
# # ax3.tight_layout()
# plt.tight_layout()

def cases():
    
    # x1 = [21,63,126,256]
    x0 = [0,365]
    y0 = [10.8,10.8]
    x1 = [30,90,180,365]
    y1 = [29.63,5.55,5.41,3.33]
    # y2 = [450, 845, 1091, 1842]
    
    # fig, (ax1,ax2) = plt.subplots(1,1, figsize=(10,5))
    fig, (ax1) = plt.subplots(1,1, figsize=(10,5))
    
    ax1.plot(x1,y1, label = 'Control ON')
    ax1.plot(x0,y0, label = 'Control OFF')
    ax1.legend(loc="upper right", ncol=2, 
                   fontsize= 'small', 
                  fancybox = False,
                  # markerscale = 35,
                  frameon = False, 
                  bbox_to_anchor=(1,1),
                  borderpad=0.1)
    ax1.set(ylabel=f'Overvoltage cases - All Customers (%)', xlabel=f'Q-table obtained with nº days (days)')
    # ax1.set_ylim(ylim)
    # ax1.set_xlim(xlim)
    
    # ax2.plot(x1,y2)
    # ax2.set(ylabel=f'nº states', xlabel=f'nº days employed')
    # ax1.set_ylim(ylim)
    # ax1.set_xlim(xlim)
        
    ax1.tick_params(direction='in',color='#595959', grid_alpha=0.5)
    # ax2.tick_params(direction='in',color='#595959', grid_alpha=0.5)
    # ax3.tick_params(direction='in',color='#595959', grid_alpha=0.5)
    ax1.grid('True')
    # ax1.tight_layout()
    # ax2.grid('True')
    # ax2.tight_layout()
    # ax3.grid('True')
    # ax3.tight_layout()
    plt.tight_layout()
    
def histogram(out):
            
    def plot_hist(ax, out_viol, length, color, label):
        
        df = pd.DataFrame(np.sum(out_viol, axis=0))
        viol = df.sum(axis=1)*100/length
        sns.distplot(viol, color = f'{color}', kde=False, ax=ax, rug=True, 
                     rug_kws = {'height': 0.02, 'alpha':0.5, 'color':'gray'},
                     label = f'{label}')
        ax.tick_params(direction='in',color='#595959', grid_alpha=0.5)
        ax.set(xlabel='% of violations by customer')
        ax.set_xlim([0, 40])
        ax.legend(loc="upper right", ncol=2, 
              fontsize= 'small', 
              fancybox = False,
              # markerscale = 35, 
              frameon = False, 
              bbox_to_anchor=(1,1),
              borderpad=0.1)
        
        
    # length = npts*numTestingCycles
    keys = list(out.keys())
    npts = out[keys[0]]['VOLTAGES'].shape[1]
    numTestingCycles = len(keys) - 1 
    out_viol = out['VIOLATIONS']
    num_quarter_cycles = int(numTestingCycles/4)
 
    sns.set(style="darkgrid", palette="colorblind", color_codes=True)
    color = ['r','b', 'g']
    fig, ax = plt.subplots(1,3, figsize=(10,3), sharey = True)
    
    plot_hist(ax[0], out_viol[0:num_quarter_cycles], num_quarter_cycles*npts, 'r', '0-25%')
    plot_hist(ax[1], out_viol[3*num_quarter_cycles:numTestingCycles], num_quarter_cycles*npts, 'g', '75-100%')
    plot_hist(ax[2], out_viol, numTestingCycles*npts, 'b', '0-100%')
    ax[0].set(ylabel='Frequency')
    
    
    plt.tight_layout()
    
def box_plot(out):
    
    def plot_bp(ax, out_viol, length, color, label):
        for i, iout in enumerate(out):
    
            df = pd.DataFrame(np.sum(out_viol, axis=0))
            viol = df.sum(axis=1)*100/length
            sns.boxplot(viol[viol<10], color = f'{color}', ax=ax)
                
            ax.tick_params(direction='in',color='#595959', grid_alpha=0.5)
            ax.set(ylabel=f'{label}')
            ax.set_xlim([0, 40])
            # ax.set_label([f'{label}'])
            # ax.legend(loc="upper right", ncol=2, 
            #             fontsize= 'small', 
            #           fancybox = False,
            #           # markerscale = 35,
            #           frameon = False, 
            #           bbox_to_anchor=(1,1),
            #           borderpad=0.1)
            

     # length = npts*numTestingCycles
    keys = list(out.keys())
    npts = out[keys[0]]['VOLTAGES'].shape[1]
    numTestingCycles = len(keys) - 1 
    out_viol = out['VIOLATIONS']
    num_quarter_cycles = int(numTestingCycles/4)
 
    sns.set(style="darkgrid", palette="colorblind", color_codes=True)
    color = ['r','b', 'g']
    fig, ax = plt.subplots(3,1, figsize=(7,5), dpi=100, sharey = True)
    
    plot_bp(ax[0], out_viol[0:num_quarter_cycles], num_quarter_cycles*npts, 'r', '0-25%')
    plot_bp(ax[1], out_viol[3*num_quarter_cycles:numTestingCycles], (numTestingCycles-3*num_quarter_cycles)*npts, 'g', '75-100%')
    plot_bp(ax[2], out_viol, numTestingCycles*npts, 'b', '0-100%')
    ax[2].set(xlabel='% of violations by customer')
    plt.xlim(0,40)
    plt.tight_layout()
    
def barplot(out):
    
    sns.reset_orig()
    keys = list(out.keys())[:-1]
    my_range=list(range(1,len(keys)+1))
    out_viol = out['VIOLATIONS']
    num_customers = out_viol.shape[1]
    cust_problems = np.transpose(np.count_nonzero(np.sum(out_viol, axis=2),axis=1)*100/num_customers)
    
    f, ax = plt.subplots(figsize=(12, 5))
    # plt.bar(np.arange(len(keys)), height = cust_problems, color="b")
    plt.vlines(x=keys, ymin=0, ymax=cust_problems, color='#007acc', alpha=0.2, linewidth=5)
    plt.plot(keys, cust_problems, "o", markersize=5, color='#007acc', alpha=0.6)
    plt.grid(False)
    ax.set_ylabel("Customers with problems (%)")
    ax.set_xlabel("Day")
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    plt.xticks(keys[::5])
    plt.tight_layout()
 

def plot_voltage_profile(settings, kVBases):
    
    def plotter(ax,df,color,label, kVBases):
        df = np.delete(df,(1,2),axis=2)
        base = np.array([1000*np.array(kVBases)]*df.shape[1]).transpose()
        max_day = np.amax(np.amax(df,axis=2)/base,axis=0)
        min_day = np.amin(np.amin(df,axis=2)/base,axis=0)
        for i,(v, kv) in enumerate(zip(df,kVBases)):
        
                # ax.plot(v/(1000*kv), 'gray', alpha=0.02, lw=1.0,zorder=2)
                # ax.plot(v/(1000*kv), 'gray', lw=0.5, zorder=1, path_effects=[pe.Stroke(linewidth=0.15, foreground='black'), pe.Normal()])
                ax.plot(v/(1000*kv), 'gray', lw=0.5, zorder=1)
                
                # if i==100:
                #     break
                
        ax.plot(min_day, 'black',lw=1.5,zorder=2)
        ax.plot(max_day, 'black',lw=1.5, label= f'{label}', zorder=1)
        
        ax.legend(loc="lower left", 
                        fontsize= 'medium', 
                      fancybox = False,
                      # markerscale = 35,
                      handletextpad=-2.0, 
                      handlelength=0,
                      frameon = False, 
                      bbox_to_anchor=(0.15,-0.05),
                      # borderpad=0.1
                      )
        
        ax.plot([0,288],[0.92,0.92], ls=":", color='r', lw=1.5)
        ax.plot([0,288],[1.05,1.05], ls=":", color='r', lw=1.5)
        # ax.axhspan(0.92, 1.05, facecolor='#42b883', alpha = 0.1)
        # ax.axhspan(1.05, 1.1, facecolor='#ff7e67', alpha = 0.7)
        # ax.axhspan(0.90, 0.92, facecolor='#ff7e67', alpha = 0.7)
        ax.grid(False)
        ax.set_ylim([0.80,1.1])
        ax.set_xlim([0,287])
        ax.tick_params(direction='in',color='#595959', grid_alpha=0.5)
        # ax.set(ylabel='Voltage (pu)')
        ax.set(xlabel='Time (h)')
        a = [0,48,96,144,192,240]
        b = [0,4,8,12,16,20]
        ax.set_xticks(a)
        ax.set_xticklabels(b)
        
        
    # sns.set(style="darkgrid", palette="colorblind", color_codes=True)
    sns.reset_orig()
    fig, ax = plt.subplots(1,3, figsize=(5,2), dpi=100, sharey=True)
    
    num_customers = 0
    days = ['180 days','365 days']
    x = [0,0,1,1]
    y = [0,1,0,1]
    i=0
    for day in [180,365]:
        i+=1   
        out = h5py.File(rf"{settings.outp_folder}\online_info_{day}_r0_nv100_c01.h5", "r")
        key = list(out.keys())
        voltages_begin = np.array(out[key[1]]['VOLTAGES'])
        voltages_end = np.array(out[key[1]]['VOLTAGES'])
        # voltages[130:170] = voltages[130:170]/1.1
        # voltages[130:170] = voltages[130:170]/1.1
        # plotter(ax[x[i],y[i]], voltages, 'b', days[i], kVBases)
        # plotter(ax[i,0], voltages_begin[:2], 'b', days[i-1], kVBases)
        # plotter(ax[i], voltages_end[:5], 'b', days[i-1], kVBases)
        plotter(ax[i], voltages_end, 'b', days[i-1], kVBases)
        
        
    out = h5py.File(rf"{settings.outp_folder}\wc_info_365.h5", "r")
    key = list(out.keys())
    voltages_end = out[key[0]]['VOLTAGES']
    # plotter(ax[0], voltages_end[:5], 'b', 'Without Control', kVBases)
    plotter(ax[0], voltages_end, 'b', 'Without Control', kVBases)
    # ax[2].set(xlabel='Time (h)')
    ax[0].set(ylabel='Voltage (pu)')
    plt.tight_layout()
    fname = rf'C:\Users\custo\OneDrive\Msc\BEPE\Conference Papers\ISGT - Asia\Figures\profile.png'
    plt.subplots_adjust(left=0.097, right=0.990, top=0.960, bottom=0.223, wspace=0.100, hspace=0.309)
    plt.savefig(fname, dpi=1000)
    # plt.subplots_adjust(wspace=0.145)
    # plt.tight_layout()

def voltage_range(out, kVBases):

    ''''''                   
    # sns.reset_orig()
    keys = list(out.keys())[:-1]
    volt_range = {
        'max': [],
        'min': []
        }
    
    for cycle in keys:
        
        volt_cycle = np.max(np.max(out[f'{cycle}']['VOLTAGES'],axis=1),axis=1)
        volt_cycle_pu = np.divide(volt_cycle,kVBases)
        max_volt_cycle_pu = np.max(volt_cycle_pu)
        min_volt_cycle_pu = np.min(volt_cycle_pu)
        volt_range['min'].append(min_volt_cycle_pu)
        volt_range['max'].append(max_volt_cycle_pu)
        
        
    fig, ax = plt.subplots(figsize=(6, 5))
        
    ax.plot(volt_range['min'], color = '#4a47a3', alpha = 0.3, lw=3, label = 'Lowest Voltage')
    ax.plot(volt_range['max'], color = '#ea6227', alpha = 0.3, lw=3, label = 'Highest Voltage')
    ax.plot(volt_range['min'], "o", markersize=7, color='#4a47a3', alpha=0.8)
    ax.plot(volt_range['max'], "o", markersize=7, color='#ea6227', alpha=0.8)
    # ax.grid(False)
    ax.set_ylim([0.91,1.12])
    ax.set_xlim([-1,len(keys)+1])
    ax.tick_params(direction='in',color='#595959', grid_alpha=0.5)
    ax.set(ylabel='Voltage (pu)', xlabel='Day')
    a = np.arange(0,len(keys)+1,5)
    ax.set_xticks(a)
    ax.set_xticklabels(keys[::5])
    ax.fill_between(np.arange(len(keys)), volt_range['min'], volt_range['max'], color = "k" , alpha = 0.2)    
    
    ax.legend(loc="upper right", ncol=2, 
                        fontsize= 'small', 
                      fancybox = False,
                      # markerscale = 35,
                      frameon = False, 
                      bbox_to_anchor=(1,1),
                      borderpad=0.1)
    plt.tight_layout()

        
def all_cases_analysis_bp_per_customer(settings):
      
    def plot_bp(ax, df, color, label):
        # for i, iout in enumerate(out):
    
            sns.boxplot(x='% violations by customer', y= case, data = df, color = f'{color}', ax=ax)
                
            ax.tick_params(direction='in',color='#595959', grid_alpha=0.5)
            ax.set_title(f'{label}')
            ax.set_xlim([0, 102])
            # ax.set_label([f'{label}'])
            # ax.legend(loc="upper right", ncol=2, 
            #             fontsize= 'small', 
            #           fancybox = False,
            #           # markerscale = 35,
            #           frameon = False, 
            #           bbox_to_anchor=(1,1),
            #           borderpad=0.1)
                        
            
    sns.set(style="darkgrid", palette="colorblind", color_codes=True)
    color = ['r','b', 'g']
    fig, ax = plt.subplots(1,3, figsize=(7,5), dpi=100, sharey = True, sharex=True)

    d_025 = []
    d_75100 = []
    d_0100 = []
    case = []
    num_customers = 0
    
     # length = npts*numTestingCycles
    for vs in settings.all_vs: 
        for inv,ic0 in zip(settings.nv,settings.c0):
            
            out = h5py.File(rf"{settings.outp_folder}\online_info_{settings.days}_{vs}_nv{inv}_c0{ic0}.h5", "r")
            keys = list(out.keys())
            npts = out[keys[0]]['VOLTAGES'].shape[1]
            numTestingCycles = len(keys) - 1 
            try:
                out_viol = out['VIOLATIONS2']
            except:
                out_viol = out['VIOLATIONS']
            num_quarter_cycles = int(numTestingCycles/4)
            
            cur_025 = np.sum(out_viol[0:num_quarter_cycles], axis=0)
            cur_75100 = np.sum(out_viol[3*num_quarter_cycles:numTestingCycles], axis=0)
            cur_0100 = np.sum(out_viol, axis=0)
            
            num_customers = cur_025.shape[0]
            case.extend([f'{vs}_{inv}_{ic0}']*num_customers)
            
            d_025.extend(cur_025.sum(axis=1)*100/(num_quarter_cycles*npts))
            d_75100.extend(cur_75100.sum(axis=1)*100/((numTestingCycles-3*num_quarter_cycles)*npts))
            d_0100.extend(cur_0100.sum(axis=1)*100/(numTestingCycles*npts))
            
            
    df_025 = pd.DataFrame({'case': case, '% violations by customer': d_025})
    df_75100 = pd.DataFrame({'case': case, '% violations by customer': d_75100})
    df_0100 = pd.DataFrame({'case': case, '% violations by customer': d_0100})
    # df_025 = pd.DataFrame(dic_025, index = list(dic_025.keys()), coluns = index)
    # df_75100 = pd.DataFrame(dic_75100, columns = list(dic_75100.keys()), index = index)
    # df_0100 = pd.DataFrame(dic_0100, columns = list(dic_0100.keys()), index = index)

            
    # plot_bp(ax[0], out_viol[0:num_quarter_cycles], num_quarter_cycles*npts, 'r', '0-25%')
    # plot_bp(ax[1], out_viol[3*num_quarter_cycles:numTestingCycles], (numTestingCycles-3*num_quarter_cycles)*npts, 'g', '75-100%')
    # plot_bp(ax[2], out_viol, numTestingCycles*npts, 'b', '0-100%')
    
    
    plot_bp(ax[0], df_025, 'r', '0-25%')
    plot_bp(ax[1], df_75100, 'g', '75-100%')
    plot_bp(ax[2], df_0100, 'b', '0-100%')

    plt.xlim(0,102)
    plt.tight_layout()
    
def all_cases_analysis_bp_per_cycle(settings):
    
    def plot_bp(ax, df, color, label):
        # for i, iout in enumerate(out):
    
            sns.boxplot(x='% daily problems', y='case', data = df, color = f'w', ax=ax,
                        whiskerprops = {'color':'#0000FF','ls': '--'}, 
                        medianprops = {'color':'#FF0000'}, 
                        capprops = {'color':'#0000FF'}, 
                        boxprops = {'edgecolor':'#0000FF'}, 
                        flierprops = {'marker':"|", 'markeredgecolor':'#0000FF'}, 
                        )
                
            ax.tick_params(direction='in',color='#595959', grid_alpha=0.5)
            ax.set_title(f'{label}', fontdict={'fontsize':10})
            ax.set_xlim([0, 102])
            ax.set_ylabel('')
            ax.set_xlabel('')
  
  
    # sns.set(style="darkgrid", palette="colorblind", color_codes=True)
    sns.set(style="whitegrid")
    sns.reset_orig()
    fig, ax = plt.subplots(1,3, figsize=(5,4), dpi=100, sharey = True, sharex=True)

    d_025 = []
    d_2575 = []
    d_75100 = []
    d_0100 = []
    case_025 = []
    case_2575 = []
    case_75100 = []
    case_0100 = []
    num_customers = 0
    for vs in settings.all_vs: 
        icase=0
        for inv,ic0 in zip(settings.nv,settings.c0):
            icase+=1
            out = h5py.File(rf"{settings.outp_folder}\online_info_{settings.days}_{vs}_nv{inv}_c0{ic0}.h5", "r")
            keys = list(out.keys())
            npts = out[keys[0]]['VOLTAGES'].shape[1]
            numTestingCycles = len(keys) - 1 
            try:
                out_viol = out['VIOLATIONS2']
            except:
                out_viol = out['VIOLATIONS']
                
            num_quarter_cycles = int(numTestingCycles/4)
            num_customers = out_viol.shape[1]
            cust_problems = np.transpose(np.count_nonzero(np.sum(out_viol, axis=2),axis=1)*100/num_customers)
            
            d_025.extend(cust_problems[0:num_quarter_cycles])
            d_2575.extend(cust_problems[num_quarter_cycles:3*num_quarter_cycles])
            d_75100.extend(cust_problems[3*num_quarter_cycles:])
            d_0100.extend(cust_problems)
           
            case_025.extend([f'${vs[0]}^{vs[1]}$-{icase}']*num_quarter_cycles)
            case_2575.extend([f'${vs[0]}^{vs[1]}$-{icase}']*(cust_problems[num_quarter_cycles:3*num_quarter_cycles].shape[0]))
            case_75100.extend([f'${vs[0]}^{vs[1]}$-{icase}']*(cust_problems[3*num_quarter_cycles:].shape[0]))
            case_0100.extend([f'${vs[0]}^{vs[1]}$-{icase}']*cust_problems.shape[0])
            # case_025.extend([f'{vs}_{inv}_{ic0}']*num_quarter_cycles)
            # case_75100.extend([f'{vs}_{inv}_{ic0}']*(cust_problems[3*num_quarter_cycles:].shape[0]))
            # case_0100.extend([f'{vs}_{inv}_{ic0}']*cust_problems.shape[0])
            
            
    df_025 = pd.DataFrame({'case': case_025, '% daily problems': d_025})
    df_2575 = pd.DataFrame({'case': case_2575, '% daily problems': d_2575})
    df_75100 = pd.DataFrame({'case': case_75100, '% daily problems': d_75100})
    df_0100 = pd.DataFrame({'case': case_0100, '% daily problems': d_0100})
    
    plot_bp(ax[0], df_025, 'r', '0-25% of days')
    plot_bp(ax[1], df_2575, 'r', '25-75% of days')
    plot_bp(ax[2], df_75100, 'g', '75-100% of days')
    # plot_bp(ax[2], df_0100, 'b', '0-100% of days')

    # plt.xlim(0,40)
    # plt.tight_layout()]
    ax[1].set_xlabel('% customer with problems for each day\n of the online training')
    plt.subplots_adjust(left=0.080, right=0.975, top=0.940, bottom=0.120, wspace=0.170)
    # plt.tight_layout()

def all_cases_analysis_bp_per_cycle_reduced(settings):
    
    def plot_bp(ax, df, color, label):
         # for i, iout in enumerate(out):
     
             sns.boxplot(x='% daily problems', y='case', data = df, color = f'w', ax=ax,
                         whiskerprops = {'color':'#0000FF','ls': '--'}, 
                         medianprops = {'color':'#FF0000'}, 
                         capprops = {'color':'#0000FF'}, 
                         boxprops = {'edgecolor':'#0000FF'}, 
                         flierprops = {'marker':"|", 'markeredgecolor':'#0000FF'}, 
                         )
                 
             ax.tick_params(direction='in',color='#595959', grid_alpha=0.5)
             ax.set_title(f'{label}', fontdict={'fontsize':10})
             ax.set_xlim([0, 102])
             ax.set_ylabel('')
             ax.set_xlabel('')
  
  
    # sns.set(style="darkgrid", palette="colorblind", color_codes=True)
    sns.reset_orig()
    fig, ax = plt.subplots(1, figsize=(5,2), dpi=100, sharey = True, sharex=True)
    
    d_025 = []
    d_75100 = []
    d_0100 = []
    case_025 = []
    case_75100 = []
    case_0100 = []
    num_customers = 0
    days = ['30 (25/5)','90 (85/5)','180 (175/5)','365 (255/5)']
    for vs in settings.all_vs: 
        icase=-1
        for day in [90,30,180,365]:
            icase+=1
            out = h5py.File(rf"{settings.outp_folder}\online_info_{day}_r0_nv100_c01.h5", "r")
            keys = list(out.keys())
            npts = out[keys[0]]['VOLTAGES'].shape[1]
            numTestingCycles = len(keys) - 1 
            try:
                out_viol = out['VIOLATIONS2']
            except:
                out_viol = out['VIOLATIONS']
                
            num_quarter_cycles = int(numTestingCycles/4)
            num_customers = out_viol.shape[1]
            cust_problems = np.transpose(np.count_nonzero(np.sum(out_viol, axis=2),axis=1)*100/num_customers)
            
            d_025.extend(cust_problems[0:num_quarter_cycles])
            d_75100.extend(cust_problems[3*num_quarter_cycles:])
            d_0100.extend(cust_problems)
            case_025.extend([f'{days[icase]}']*num_quarter_cycles)
            case_75100.extend([f'{days[icase]}']*(cust_problems[3*num_quarter_cycles:].shape[0]))
            case_0100.extend([f'{days[icase]}']*cust_problems.shape[0])
            # case_025.extend([f'{vs}_{inv}_{ic0}']*num_quarter_cycles)
            # case_75100.extend([f'{vs}_{inv}_{ic0}']*(cust_problems[3*num_quarter_cycles:].shape[0]))
            # case_0100.extend([f'{vs}_{inv}_{ic0}']*cust_problems.shape[0])
            
            
    df_025 = pd.DataFrame({'case': case_025, '% daily problems': d_025})
    df_75100 = pd.DataFrame({'case': case_75100, '% daily problems': d_75100})
    df_0100 = pd.DataFrame({'case': case_0100, '% daily problems': d_0100})
    
    # plot_bp(ax[0], df_025, 'r', '0-25% of days')
    # plot_bp(ax[1], df_75100, 'g', '75-100% of days')
    plot_bp(ax, df_0100, 'b', '')
    
    # plt.xlim(0,40)
    # plt.tight_layout()]
    ax.set_xlabel('% customer with problems for each day\n of the online training')
    ax.set_ylabel('Total Days\n (Offline/Online)')
    plt.subplots_adjust(left=0.264, right=0.952, top=0.940, bottom=0.245, wspace=0.145)
    # plt.tight_layout()
    
    
    
    
if __name__ == "__main__":

    out = []
    # outp_folder = r'C:\Users\custo\OneDrive\Msc\BEPE\rl'
    
    # on_h5file = h5py.File(rf"{settings.outp_folder}\online_info_{settings.days}_{settings.vs}.h5", "r+")
    
    # out.append(pd.read_pickle(rf'{outp_folder}/online_info_180.pickle'))
    # out.append(pd.read_pickle(rf'{outp_folder}/online_info_365.pickle'))
    histogram(out)
    # box_plot(out)
    # plot_voltage_profile(out)
