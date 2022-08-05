from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import json
from datetime import timedelta, datetime
import matplotlib.dates as mdates
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math


def get_env_plotting_data(memory=None, env_ids=None):
    (bg, cgm, t, cho, ins, ma) = memory.get_simu_data()
    data = dict(bg=bg, cgm=cgm, t=t, cho=cho, ins=ins, ma=ma)
    dataCPU = {k: v.detach().cpu().numpy() for k, v in data.items()}
    sub, dfs = [], {}
    col = ['bg', 'cgm', 't', 'meal', 'ins', 'ma']
    for x in env_ids:
        sub.append(np.array([dataCPU['bg'][:,x], dataCPU['cgm'][:,x], dataCPU['t'][:,x],
                             dataCPU['cho'][:,x], dataCPU['ins'][:,x], dataCPU['ma'][:,x]]))
        dfs[x] = pd.DataFrame(np.transpose(sub[x]), columns=col)
    return dfs


def set_time(df):
    df['time'] = np.arange(len(df))
    prev_hour, day, prev_t = 0, 1, 0
    for i, row in df.iterrows():
        hour = int((row['t']/60)%24)
        min = int(row['t']%60)
        if prev_hour > hour:
            day += 1
        if prev_t > row['t']:
            day += 1
        df.at[i, 'time'] = datetime(2022, 1, day, hour, min, 0)
        prev_hour = hour
        prev_t = row['t']
    return df


def render(memory, env_ids):
    dict = get_env_plotting_data(memory=memory, env_ids=env_ids)
    df = dict[0]
    df = set_time(df)

    fig = plt.figure(figsize=(16, 6))
    ax2 = fig.add_subplot(111)
    ax2.set_yscale('log')
    ax2.set_ylim((1e-3, 5))
    divider = make_axes_locatable(ax2)
    ax = divider.append_axes("top", size=3.0, pad=0.02, sharex=ax2)

    s_t = memory.sample_time.to('cpu')
    sensor_name = memory.sensor_params
    pump_name = memory.pump_params
    tot_len = df.shape[0] #int(df.shape[0] / s_t)
    cgm_color = '#000080'
    ins_color = 'mediumseagreen'
    meal_color = '#800000'
    max_ins = max(df['ins'])
    max_cgm = min(max(df['cgm']) + 100, 620)

    # plot glcose and insulin.
    ax.plot(df['time'], df['cgm'], markerfacecolor=cgm_color, linewidth=2.0)
    ax2.bar(df['time'], df['ins'], (1/tot_len), color=ins_color)
    ax.axhline(y=50, color='r', linestyle='--')
    ax.axhline(y=300, color='r', linestyle='--')
    ax.axhspan(70, 180, alpha=0.2, color='limegreen', lw=0)

    x = True
    for t in range(0, len(df)):
        if df.iloc[t]['meal']:
            off_set = (max_cgm - 125) if x else (max_cgm - 75)
            ax.annotate('Carbohydrates: ' + str(df.iloc[t]['meal'])+'g', (df.iloc[t]['time'], off_set), color=meal_color)  #df.iloc[t]['cgm']
            ax.plot((df.iloc[t]['time'], df.iloc[t]['time']), (df.iloc[t]['cgm'], off_set), color=meal_color)
            x = not(x)

    start_time = df['time'].iloc[0]  #
    end_time = df['time'].iloc[-1]
    ax2.set_xlim([start_time, end_time]) # start_time + timedelta(hours=3)] #start_time + timedelta(hours=24)
    ax2.xaxis.set_minor_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M\n'))
    ax2.xaxis.set_major_locator(mdates.DayLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('\n%b %d'))

    ax.set_ylim(5, max_cgm)
    ax.set_ylabel('CGM [mg/dL]', color=cgm_color)
    ax2.set_ylabel('Insulin [U/min]', color=ins_color)
    ax2.set_xlabel('Time (hrs)')
    ax.set_title('Simulation: Glucose Regulation')
    ax.grid()

    cgm_line = mlines.Line2D([], [], color=cgm_color, label='CGM (Sensor: '+sensor_name+')')
    ins_line = mlines.Line2D([], [], color=ins_color, label='Insulin (Pump: '+pump_name+')')
    meal_ann_line = mlines.Line2D([], [], color='k', marker='D', linestyle='None', label='Meal Announcement (20min)')
    ax.legend(handles=[cgm_line, ins_line], loc='upper right')  # meal_ann_line

    # TIR ranges
    # ax.text(df.iloc[1]['time'], 310, 'Severe Hyperglycemia', size=12, color='r')
    # ax.text(df.iloc[1]['time'], 280, 'Hyperglycemia', size=12)
    # ax.text(df.iloc[1]['time'], 100, 'Normoglcemia', size=12, color=cgm_color)
    # ax.text(df.iloc[1]['time'], 54, 'Hypoglycemia', size=12)
    # ax.text(df.iloc[1]['time'], 30, 'Severe Hypoglycemia', size=12, color='r')

    #fig.savefig(experiment.experiment_dir +'/'+ str(tester))
    plt.show()

