import numpy as np
import matplotlib.pyplot as plt
import pylab
import pickle
import os
import pandas as pd
import statsmodels.tsa.api as sta
import math
import statsmodels.tsa.stattools as sts
import datetime
import time
import math
import statistics as st
import statsmodels.stats.stattools as ssts
import seaborn
import itertools

time_diff_ratio_list = []
eta_intervals = [[] for i in range(10)]
eta_avg_list = {}
measures_list = {}

class DataProcess:
    def __init__(self, mfile, tfile):
        measure_file_path = 'F://Github//Causality-Prediction//data//measure_series//inhib//v2'
        steep_inhib_times = pickle.load(
            open('F://Inhibition//VAR_causality//data_files//steep_inhib_times.pickle', 'rb'))


class TLogic:
    def __init__(self, data, mid, inhib_time, measures, lag, r, s, k):
        self.cascade_df = data
        self.mid = mid
        self.measures = measures
        self.lag = lag
        self.r = r
        self.s = s
        self.k = k

        # Storing inhib time as mktuple()
        rt_time = str(inhib_time)
        rt_date = rt_time[:10]
        rt_t = rt_time[11:19]
        record_time = rt_date + ' ' + rt_t
        time_x = datetime.datetime.strptime(record_time, '%Y-%m-%d %H:%M:%S')
        inhib_time = time.mktime(time_x.timetuple())
        self.inhib_time = inhib_time

        self.time_intervals = {0:200, 1: 500, 2: 1000, 3:1800, 4: 2800, 5: 4000, 6:6000, 7:9000, 8:15000, 9:30000}
        self.cascade_df['time_date'] = pd.to_datetime(self.cascade_df['time_date'], format='%Y-%m-%d %H:%M:%S')
        self.cascade_df = self.cascade_df.reset_index(drop=True)
        self.time_intervals_list = {}
        self.time_points = self.cascade_df['time_date'].tolist()


        self.dnIntervals_effect = {}
        self.dnIntervals_sig_effect = {}

        for idx in range(len(self.measures)):
            self.dnIntervals_effect[self.measures[idx]] = []
            self.dnIntervals_sig_effect[self.measures[idx]] = []

    def test_stationarity(self, timeseries, name):

        timeseries = list(timeseries)
        # Determing rolling statistics
        rolmean = pd.rolling_mean(pd.Series(timeseries), window=12)
        rolstd = pd.rolling_std(pd.Series(timeseries), window=12)

        # Plot rolling statistics:
        plt.close()
        orig = plt.plot(timeseries, color='blue',label='Original')
        mean = plt.plot(rolmean, color='red', label='Rolling Mean')
        std = plt.plot(rolstd, color='black', label = 'Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation: ' + str(name))
        dir_save = '../../plots/' + str(mid)
        if not os.path.exists(dir_save):
            os.makedirs(dir_save)
        plt.savefig(dir_save + '/'+ 'Series_' + str(name))

    def dynamic_intervals(self):
        self.cascade_df.index = pd.to_datetime(self.cascade_df['time_date'], format='%Y-%m-%d %H:%M:%S')
        # print(self.cascade_df)

        # self.test_stationarity(self.cascade_df['time_diff'].tolist(), 'time_diff')
        for idx in range(len(self.measures)):
            # self.test_stationarity(self.cascade_df[self.measures[idx]].tolist(), self.measures[idx])

            # check whether the first formula is satisfied
            # This part is to check whether the feature traces satisfy the
            # behaviors laid down by STL semantics.

            # First formula using until operator to select potential causes
            # 2nd formula in until operator to select causes that leads to fall in values.
            for t_series in range(len(self.time_points)):
                # print('Time point: ', t_series)
                if t_series+self.s >= len(self.time_points):
                    break
                mean_interval_cause = np.mean(self.cascade_df[self.measures[idx]][t_series - self.s + self.r:t_series])
                if self.cascade_df[self.measures[idx]][t_series] < mean_interval_cause:
                    for t_points in range(t_series+self.r, t_series+self.s):
                        mean_interval_effect = np.mean(self.cascade_df['time_diff'][t_points - self.s + self.r:t_points])
                        if self.cascade_df['time_diff'][t_points] > self.k*mean_interval_effect:
                            self.dnIntervals_effect[self.measures[idx]].append((t_series, t_points))
                            break

    def potential_causes(self):
        # for idx_measures in range(len(self.measures)):
        #     # causes_increase = self.dnIntervals_cause_increase[self.measures[idx_measures]]
        #     causes_decrease = self.dnIntervals_cause_decrease[self.measures[idx_measures]]
        #     for idx_cause in range(0, len(causes_decrease)):
        #         effect_cond_exp = 0
        #         c_prime = causes_decrease[idx_cause]
        #         effect = 0
        #         effect_cond_exp = self.cascade_df['time_diff'][c_prime+self.lag]
        #         for idx_cond in range(0, c_prime[0]):
        #             effect += self.cascade_df['time_diff'][idx_cond]
        #         effect_cond_exp /= (c_prime[1] - c_prime[0])
        #         effect /= (c_prime[0])
        #         # print(effect, effect_cond_exp)
        #
        #         if effect < effect_cond_exp:
        #             self.dnIntervals_sig_cause_decrease[self.measures[0]].append(causes_decrease[idx_cause])

        self.dnIntervals_sig_effect = self.dnIntervals_effect
        # print(self.dnIntervals_cause_decrease)

    def eta_avg_func(self):
        # testing whether E(e|c and x) > E(e|not c and x) for a cause c on effect e.
            for idx_prima in range(len(self.measures)):
                # print(self.measures[idx_prima])
                t_points_prima = []
                eta_avg = 0

                try:
                    causes_prima = self.dnIntervals_sig_effect[self.measures[idx_prima]]
                except KeyError:
                    continue
                for idx_cause in range(len(causes_prima)):
                    t_points_prima.append(causes_prima[idx_cause][1])

                for idx_others in range(len(self.measures)):
                    t_points_union = []
                    if idx_others == idx_prima:
                        continue
                    try:
                        causes_others = self.dnIntervals_sig_effect[self.measures[idx_others]]
                    except KeyError:
                        continue
                    for idx_cause in range(len(causes_others)):
                        t_points_union.append(causes_others[idx_cause][1])

                    c_inter_x = list(set(t_points_union).intersection(set(t_points_prima)))
                    not_c_inter_x = list(set(t_points_union) - set(c_inter_x))

                    effect_incl = 0
                    cnt = 0
                    for t in c_inter_x:
                        try:
                            effect_incl += self.cascade_df['time_diff'][t]
                            cnt += 1
                        except KeyError:
                            continue
                    if cnt != 0:
                        effect_incl /= cnt

                    cnt = 0
                    effect_excl = 0
                    for t in not_c_inter_x:
                        try:
                            effect_excl += self.cascade_df['time_diff'][t]
                            cnt += 1
                        except KeyError:
                            continue
                    if cnt != 0:
                        effect_excl /= cnt

                    eta = (effect_incl - effect_excl)
                    eta_avg += eta

                eta_avg /= (len(self.measures) - 1)
                eta_avg_list[self.measures[idx_prima]].append(eta_avg)
        # print(eta_avg_list)

if __name__ == '__main__':
    measure_file_path = 'F://Github//Causality-Prediction//data//measure_series//inhib//v2'
    steep_inhib_times = pickle.load(open('F://Inhibition//VAR_causality//data_files//steep_inhib_times.pickle', 'rb'))

    dependent_variables = []
    # print(cascade_int_time)

    # Load the measure files
    cnt_measures = 0
    model_df = pd.DataFrame()
    control_variables = {}
    dependent_variables = {}

    measures = []
    model_order = 10
    f_list_null = []
    p_list_null = []

    f_list_full = []
    p_list_full = []
    measure_time_df = []

    for files in os.listdir(measure_file_path):
        full_path = measure_file_path + '//' + files
        mname = files[:len(files)-7]
        measure_time_df.append(pickle.load(open(full_path, 'rb')))
        measures.append(mname)

    statistic_values = {}
    p_values = {}
    critical_values = {}
    granger_cause_count = {}

    for m in measures:
        eta_avg_list[m] = []

    for L in range(len(measures), len(range(len(measures))) + 1):
        for subset in itertools.combinations(range(len(measures)), L):
            num_measures = len(subset)
            if num_measures == 0:
                continue
            measure_series = []
            cnt_mids = 0
            for mid in measure_time_df[0]:
                cascade_df_feat = pd.DataFrame()
                steep_time = pd.to_datetime(steep_inhib_times[mid]['steep'])
                inhib_time = pd.to_datetime(steep_inhib_times[mid]['decay'])
                # print(inhib_time)
                # Combine all the features in a dataframe for sorting them by time.
                cascade_df = measure_time_df[0][mid]
                cascade_df_feat['time'] = pd.to_datetime(cascade_df['time'])
                for idx in range(num_measures):
                    # if measures[subset[idx]] == 'bw' or measures[subset[idx]] == 'cc':
                    cascade_df_feat[measures[subset[idx]]] = measure_time_df[subset[idx]][mid][measures[subset[idx]]]
                cascade_df_feat = cascade_df_feat.sort('time')
                cascade_df_feat = cascade_df_feat.dropna()
                #cascade_df_feat = cascade_df_feat[cascade_df_feat['time'] < pd.to_datetime(inhib_time)]

                # print(len(cascade_df_feat))
                time_series = list(cascade_df_feat['time'])
                measure_series_all = []
                for idx in range(num_measures):
                    measure_series_all.append(list(cascade_df_feat[measures[subset[idx]]]))

                Y_act = [0 for idx in range(len(time_series))]
                Y_time = []

                for idx in range(1, len(time_series)):
                    rt_time = str(time_series[idx])
                    rt_date = rt_time[:10]
                    rt_t = rt_time[11:19]
                    record_time = rt_date + ' ' + rt_t
                    time_x = datetime.datetime.strptime(record_time, '%Y-%m-%d %H:%M:%S')
                    cur_time = time.mktime(time_x.timetuple())

                    rt_time = str(time_series[idx-1])
                    rt_date = rt_time[:10]
                    rt_t = rt_time[11:19]
                    record_time = rt_date + ' ' + rt_t
                    time_x = datetime.datetime.strptime(record_time, '%Y-%m-%d %H:%M:%S')
                    prev_time = time.mktime(time_x.timetuple())

                    Y_act[idx] = (cur_time - prev_time)/60

                X = [[] for i in range(len(subset))]
                Y = []
                time_new = []
                measures_ratios = [[] for i in range(len(subset))]
                # Remove the time_series rows with values of \delta t=0
                for idx in range(len(Y_act)):
                    # if Y_act[idx] == 0 and idx > 0:
                    #     continue
                    # measures_ratios[idx_sub].append(1)
                    for idx_sub in range(len(subset)):
                        X[idx_sub].append(measure_series_all[idx_sub][idx])
                        if idx > 0:
                            measures_ratios[idx_sub].append(measure_series_all[idx_sub][idx] / measure_series_all[idx_sub][idx-1])
                        else:
                            measures_ratios[idx_sub].append(1)
                        # print(measures_ratios[idx_sub])
                    Y.append(Y_act[idx])
                    time_new.append(time_series[idx])

                cascade_ts_df = pd.DataFrame()
                cascade_ts_df['time_diff'] = Y
                cascade_ts_df['time_date']= time_new
                for idx_sub in range(len(subset)):
                    cascade_ts_df[measures[subset[idx_sub]]] = X[idx_sub]
                    cascade_ts_df[measures[subset[idx_sub]] + '_ratios'] = measures_ratios[idx_sub]
                s = cascade_ts_df.shape
                cascade_ts_df = cascade_ts_df.fillna(0)

                # for idx_sub in range(len(subset)):
                #     measures_causality.append(measures[subset[idx_sub]])
                #     measures_string += (measures[subset[idx_sub]] + ' + ')
                # measures_string = measures_string[:len(measures_string) - 3]

                Y_diff = []
                X_diff = [[] for i in range(len(subset))]
                for idx in range(len(Y)):
                    if np.isnan(Y[idx]):
                        continue
                    Y_diff.append(Y[idx])
                    for idx_sub in range(len(subset)):
                        X_diff[idx_sub].append(X[idx_sub][idx])

                X = np.asarray(X_diff)
                Y = np.asarray(Y_diff)

                cascade_VAR_df = pd.DataFrame()
                cascade_VAR_df['time_diff'] = Y
                for idx_sub in range(len(subset)):
                    # print(measures[subset[idx_sub]])
                    cascade_VAR_df[measures[subset[idx_sub]]] = X[idx_sub, :]

                cnt_mids += 1
                temporal_logic = TLogic(cascade_ts_df, mid, inhib_time, measures, 2, 2, 5, 5)
                temporal_logic.dynamic_intervals()
                # temporal_logic.rules_formulas(50, 150)
                temporal_logic.potential_causes()
                temporal_logic.eta_avg_func()
                print('Mid: ', cnt_mids)

                if cnt_mids > 1500:
                    break

    eta_store = [[] for i in range(5)]
    titles = []
    for idx in range(len(measures)):
        eta_store[idx] = eta_avg_list[measures[idx]]
        titles.append(str(measures[idx]))

    # for idx in range(len(eta_intervals)):
    #     print(len(eta_intervals[idx]))
    # print(eta_intervals[9][:10])
    data_to_plot = eta_store
    # Create the box_plots
    fig = plt.figure(1, figsize=(15, 12))

    # Create an axes instance
    ax = fig.add_subplot(111)

    # Create the boxplot
    bp = ax.boxplot(data_to_plot, patch_artist=True)

    third_quartile = [item.get_ydata()[0] for item in bp['whiskers']]
    quarts = []
    for idx in range(len(third_quartile)):
        if not np.isnan(third_quartile[idx]):
            quarts.append(third_quartile[idx])
    upper_quart = max(quarts)
    lower_quart = min(quarts)

    # print(upper_quart, lower_quart)

    for box in bp['boxes']:
        # change outline color
        box.set(color='#7570b3', linewidth=2)
        # change fill color
        box.set(facecolor='#1b9e77')

    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)

    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=2)

    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=2)
    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)

    ax.set_title('Causal Significance: r=2, s=5, k=5', size=30)
    ax.set_xlabel('Features', size=30)
    # ax.set_ylim([0, 100])
    ax.set_xticklabels(titles, size=30)

    dir_save = '../../plots/causal_significance/10_24/decrease_increase'
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)
    file_save = dir_save + '/' + 'r_2_s_5_k_5' + '.png'
    plt.ylim([lower_quart - 2*math.pow(10, int(math.log10(abs(lower_quart)))), 2*upper_quart + math.pow(10, int(math.log10(upper_quart)))])
    plt.savefig(file_save)
    plt.close()
