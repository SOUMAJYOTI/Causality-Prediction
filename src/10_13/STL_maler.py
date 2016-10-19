# Use some existing time series data to test PCTLc logic by Samantha Kleinberg
# Fix the window time here ( |r - s| is fixed here).

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


class DataProcess:
    def __init__(selfself, mfile, tfile):
        measure_file_path = 'F://Github//Causality-Prediction//data//measure_series//inhib//v2'
        steep_inhib_times = pickle.load(
            open('F://Inhibition//VAR_causality//data_files//steep_inhib_times.pickle', 'rb'))


class TLogic:
    def __init__(self, data, measures):
        self.cascade_df = data
        self.measures = measures
        self.cascade_df['time_date']= pd.to_datetime(self.cascade_df['time_date'], format='%Y-%m-%d %H:%M:%S')
        self.cascade_df = self.cascade_df.reset_index(drop=True)
        self.dnIntervals_cause_increase = {}
        self.dnIntervals_cause_decrease = {}
        self.dnIntervals_effect = {}
        self.dnIntervals_sig_cause_increase = {}
        self.dnIntervals_sig_cause_decrease = {}
        for idx in range(len(self.measures)):
            self.dnIntervals_effect[self.measures[idx]] = []
            self.dnIntervals_cause_increase[self.measures[idx]] = []
            self.dnIntervals_cause_decrease[self.measures[idx]] = []
            self.dnIntervals_sig_cause_increase[self.measures[idx]] = []
            self.dnIntervals_sig_cause_decrease[self.measures[idx]] = []

    def dynamic_intervals(self, r, s, lag):
        mean_series = np.mean(self.cascade_df[self.measures[0]]) /2
        self.cascade_df.index = pd.to_datetime(self.cascade_df.index, format='%Y-%m-%d %H:%M:%S')
        # print(len(self.cascade_df))
        startPoint = 0
        endPoint = 0

        for idx_m in range(len(self.measures)):
            time_points = self.cascade_df['time_date'].tolist()

            # check whether the first formula is satisfied
            # This part is to check whether the feature traces satisfy the
            # behaviors laid down by STL semantics.

            for t_series in range(len(time_points)):
                # print('Time point: ', t_series)
                if t_series+s >= len(time_points):
                    break
                for t_points in range(t_series+r, t_series+s):
                    # print('Start_time: ', t_points)
                    if t_series-r < 0:
                        break
                    mean_interval = np.mean(self.cascade_df[self.measures[0]][t_series-r:t_series-r+s])

                    idx_cur = t_series
                    for idx_cur in range(t_series, t_points+1):
                        # print(self.cascade_df[self.measures[0]][idx_cur], mean_series)
                        if self.cascade_df[self.measures[0]][idx_cur] <= mean_interval:
                            break
                    if idx_cur == t_points:

                        #the second formula pertains to the effect - not considering it right now !!!
                        # time_diff_cur = self.cascade_df['time_diff'][idx_cur]
                        # time_diff_prev = self.cascade_df['time_diff'][idx_cur-1]
                        #
                        # # print('Time_diff ratio: ', time_diff_cur/time_diff_prev)
                        # if time_diff_prev !=0 and (time_diff_cur!=0) and (time_diff_cur/time_diff_prev) > 2:
                        #     time_diff_ratio_list.append(time_diff_cur/time_diff_prev)

                        # the second formula pertains to the cause only !!!

                        # print(self.cascade_df[self.measures[0]][idx_cur+lag], mean_interval)
                        if self.cascade_df[self.measures[0]][idx_cur+lag] <= 2*mean_interval:
                            self.dnIntervals_cause_decrease[self.measures[0]].append((t_series, t_points))
                            # print('Decrease cause: ', (t_series, t_points))
                        elif self.cascade_df[self.measures[0]][idx_cur+lag] >= 2*mean_interval:
                            self.dnIntervals_cause_increase[self.measures[0]].append((t_series, t_points))
                            # print('Increase cause: ', (t_series, t_points))


    def potential_causes(self, lag):
        for idx_measures in range(len(self.measures)):
            causes_increase = self.dnIntervals_cause_increase[self.measures[idx_measures]]
            causes_decrease = self.dnIntervals_cause_decrease[self.measures[idx_measures]]
            for idx_cause in range(0, len(causes_decrease)):
                effect_cond_exp = 0
                c_prime = causes_decrease[idx_cause]
                effect = 0
                for idx_cond in range(c_prime[0], c_prime[1]+1):
                    effect_cond_exp += self.cascade_df['time_diff'][idx_cond+lag]
                for idx_cond in range(0, c_prime[0]):
                    effect += self.cascade_df['time_diff'][idx_cond]
                effect_cond_exp /= (c_prime[1] - c_prime[0])
                effect /= (c_prime[0])
                # print(effect, effect_cond_exp)

                if effect < effect_cond_exp:
                    self.dnIntervals_sig_cause_decrease[self.measures[0]].append(causes_decrease[idx_cause])

    def eta_avg(self, lag):
        for idx_measures in range(len(self.measures)):
            causes_increase = self.dnIntervals_sig_cause_increase[self.measures[idx_measures]]
            causes_decrease = self.dnIntervals_sig_cause_decrease[self.measures[idx_measures]]
            for idx_cause in range(1, len(causes_decrease)):
                for idx_prev in range(0, idx_cause):
                    cause_prev = causes_decrease[idx_prev]
                    effect_sum_excl = self.cascade_df['time_diff'][cause_prev[1]+lag]
                    for idx_cond in range(cause_prev[0], cause_prev[1]+1):
                        effect_cond_exp += self.cascade_df['time_diff'][idx_cond]
                c_prime = causes_decrease[idx_cause]
                mean_sum_excl = effect_sum_excl / idx_cause
                effect_sum_incl = effect_sum_excl + self.cascade_df['time_diff'][c_prime[1]+lag]
                mean_sum_incl = effect_sum_incl /(idx_cause+1)

                eta_avg = (mean_sum_incl - mean_sum_excl) / (idx_cause+1)
                print(eta_avg)

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
                if True: #cnt_mids == 100:
                    temporal_logic = TLogic(cascade_ts_df, measures)
                    temporal_logic.dynamic_intervals(5, 15, 3)
                    # temporal_logic.rules_formulas(50, 150)
                    temporal_logic.potential_causes(5)
                    # temporal_logic.eta_avg(5)
                print('Mid: ', cnt_mids)
                if cnt_mids > 10:
                    break

    # print(len(time_diff_ratio_list))
    # n, bins, patches = plt.hist(time_diff_ratio_list, 30, facecolor='g')
    # plt.xlabel('Time diff')
    # plt.ylabel('Frequency')
    # # plt.title('Histogram of User activity Times')
    # plt.grid(True)
    # plt.show()
    # # plt.savefig('Cascade_figures/trash/user_activity_histogram.png')

