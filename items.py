import copy

import pandas as pd
import spark
from plotly.subplots import make_subplots
import plotly.graph_objs as go

from settings import *


class Data():

    def __init__(self, df, df_bit, df_phys, df_temp, df_temp_phys, fig, default_states):
        self.df = df
        self.df_bit = df_bit
        self.df_phys = df_phys
        self.df_temp = df_temp
        self.df_temp_phys = df_temp_phys
        self.fig = fig
        self.default_states = default_states

    def drop_conseq_duplicates(self, df):
        '''
        keep one signal combination from consecutive duplicated signal values
        Input:
          - df: pandas dataframe
        Output:
          - df_seq: pandas dataframe without consecutive duplicates
          - frequent_values: list of the most frequent signal values
        '''
        frequent_values = {}
        for col in df.columns:
            frequent_values[col] = df[col].value_counts().idxmax()
        df_bool = df.shift(-1).values != df.values
        df_seq = df.loc[df_bool.any(axis=1)]
        return df_seq, frequent_values

    def extract_sequences(self, df, df_index):
        '''
        Extract relevant sequences between two default states
        Input:
          - df: pandas dataframe without cosequtive duplicates (output from the function drop_conseq_duplicates)
          - df_index: pandas dataframe containing the index of the default states
        Output:
          - sequences: pattern list
        '''
        sequences = {}
        sequences_nbr = list(df_index.index.values)
        df = df.reset_index()
        df_index = df_index.reset_index()  # not used?
        ind = 0
        ser = 0
        for i in sequences_nbr:
            indx_row = df.timestamp[df.timestamp == i].index[0]
            sequences[ind] = df[ser:indx_row]
            ser = indx_row
            ind = i
        return sequences

# WHERE ARE WE CALLING THIS FUNCTION?
    def seq_frequency(self, df):
        # Compute the frequency of different row combination in a pandas dataframe
        df_freq = df[Labels.rel_labels]
        df_freq = df_freq.groupby(df_freq.columns.tolist()).size().reset_index().rename(columns={0: 'frequency'})
        df_freq['%frequency'].sum(axis=0, skipna=True)['incidence']
        return df_freq

    def import_data(self, labels):
        df_new = self.df.filter(self.df.label.isin(labels)).toPandas()
        df_new = df_new[['timestamp', 'value', 'label', 'filename']].sort_values(by=['timestamp'])
        df_new = pd.pivot_table(df_new, values='value', index=['timestamp', 'filename'], columns='label').reset_index()
        if labels is Labels.phys_labels:
            df_new['tg_M_gear_driv'] = df_new['tg_M_gear_driv'] * 1000
        if labels is Labels.rel_labels:
            df_new['msdl_M_state_agr'] = ((df_new['msdl_M_state_agr'] - 770) * 255).div(6401 - 770).round(2)
        return df_new

    def extract_patterns(self, df_bit):
        pattern_list = {}
        for file_name in self.file_names:
            df_temp = df_bit.loc[df_bit.filename == file_name]
            df_seq, frequent_values = self.drop_conseq_duplicates(df_temp)
            df_indexed = self.df[:0]
            for _, ii in self.default_states.iterrows():
                df_indexed = df_indexed.append(df_seq.loc[df_seq[Labels.rel_labels].isin(ii.tolist()).all(axis=1)])
                df_indexed = df_indexed.append(
                    df_seq.loc[(df_seq.msdl_M_state_submdu == 255) & (df_seq.msdl_M_state_submdu2 == 255)])
            df_indexed = df_indexed[~df_indexed.index.duplicated(keep='first')].sort_index()
            pattern_list[file_name] = self.extract_sequences(df_seq, df_indexed)
        return pattern_list

    def plot_filenames(self, filenames_to_plot, messung=None):
        for filename in filenames_to_plot.tolist():
            self.fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
            self.df_temp = self.df_bit[self.df_bit['filename'] == filename].drop(['filename'], axis=1)
            self.df_temp_phys = self.df_phys[self.df_phys['filename'] == filename].drop(['filename'], axis=1)
            if messung is True:
                self.df_temp = self.df_temp.iloc[Limits.LOWER_LIMIT:Limits.UPPER_LIMIT_MESSUNG]
                self.df_temp_phys = self.df_temp_phys.iloc[Limits.LOWER_LIMIT:Limits.UPPER_LIMIT_MESSUNG]
                self.plot_results(self.df_temp, messung=True)

            else:
                self.plot_results(self.df_temp)
                self.plot_results(self.df_temp_phys, phys=True)  # should fig be returned or it's updated in function?

                self.fig.show(renderer="databricks")

    def plot_results(self, df_temp, phys=None, messung=None):
        if phys:
            for i in self.df_temp_phys.columns[1:]:
                if messung:
                    self.fig.add_trace(go.Scatter(x=self.df_temp_phys.timestamp, y=self.df_temp_phys[i], name=i), row=2, col=1)
                else:
                    self.fig.add_trace(
                        go.Scatter(x=self.df_temp_phys[:Limits.UPPER_LIMIT].timestamp, y=self.df_temp_phys[i][0:Limits.UPPER_LIMIT], name=i),
                        row=2, col=1)
            self.fig.update_layout(hovermode="x unified")
        else:
            for i in df_temp.columns:
                if messung:
                    self.fig.add_trace(go.Scatter(x=df_temp.index, y=df_temp[i], name=i), row=1, col=1)
                else:
                    self.fig.add_trace(go.Scatter(x=df_temp.index[:Limits.UPPER_LIMIT], y=df_temp[i][0:Limits.UPPER_LIMIT], name=i), row=1, col=1)
            self.fig.update_layout(hovermode="x unified", yaxis5=dict(type='category', categoryorder='category ascending'),
                              title=self.filename)
        incr = 0  # not used
        for k in self.pattern_list[self.filename]:
            if len(self.pattern_list[self.filename][k]) < 3:
                continue
            x0 = self.pattern_list[self.filename][k].timestamp.min()
            x1 = self.pattern_list[self.filename][k].timestamp.max()
            if phys and x0 > Limits.UPPER_LIMIT / 100:
                break
            if messung and x0 > Limits.UPPER_LIMIT_MESSUNG / 100:
                break
            elif messung and x1 < Limits.LOWER_LIMIT / 100:
                continue
            self.fig.add_vrect(x0=x0, x1=x1, annotation_text="pattern", annotation_position="top", fillcolor="red",
                          opacity=0.1, line_width=1)

    def check_and_save_patterns(self, pattern_list):
        pattern_list_temp = copy.deepcopy(pattern_list)
        patterns_all = pd.DataFrame(columns=Labels.rel_labels)

        for filename in pattern_list_temp:
            shift = 99
            shift_label = ''
            for seq in pattern_list_temp[filename]:
                shift = 99
                shift_label = ''
                for t, _ in pattern_list_temp[filename][seq].iterrows():
                    gear_st = self.df_phys[(self.df_phys['filename'] == filename) & (
                            self.df_phys['timestamp'] == pattern_list_temp[filename][seq]['timestamp'].loc[t])][
                        'tgd_M_gear_driv'].values[0]
                    if gear_st != shift:
                        shift = gear_st
                        shift_label = shift_label + str(int(gear_st / 1000))
                    patterns_all = patterns_all.append(
                        pattern_list_temp[filename][seq].drop(['timestamp', 'filename'], axis=1))
                    pattern_list_temp[filename][seq] = pattern_list_temp[filename][seq].drop(['timestamp', 'filename'],
                                                                                             axis=1)
                    patterns_all = patterns_all.append(dict(zip(patterns_all.columns, [0, 0, 0, 0])), ignore_index=True)

        result = {}
        result_fin = {}
        incr = 1
        tolerance = 2
        for filename in pattern_list:
            for aa, value in pattern_list[filename].items():
                if (value[1:].drop(['timestamp', 'filename'], axis=1).values.tolist() not in result.values()):
                    if len(value[1:].values) < 2: continue
                    result[incr] = value[1:].drop(['timestamp', 'filename'], axis=1).values.tolist()
                    result_fin[incr] = value[1:].drop(['timestamp', 'filename'], axis=1).values.tolist()
                    shift = 99
                    shift_label = ''
                    for t in range(len(result[incr]) + tolerance):
                        if t < len(result[incr]):
                            gear_st = self.df_phys[
                                (self.df_phys['filename'] == filename) & (
                                            self.df_phys['timestamp'] == value['timestamp'].iloc[t])][
                                'tgd_M_gear_driv'].values[0]
                        else:
                            index = t - len(result[incr]) + 1
                            gear_st = self.df_phys[(self.df_phys['filename'] == filename) & (
                                    self.df_phys['timestamp'] == round(value['timestamp'].iloc[t - index] + 0.01 * index,
                                                                  2))]['tgd_M_gear_driv'].values[0]
                        if gear_st != shift:
                            shift = gear_st
                            shift_label = shift_label + str(int(gear_st / 1000))
                            # if len(shift_label)>5: assert(False)
                    for t in range(len(result[incr])):
                        result_fin[incr][t].append(shift_label)
                    if len(shift_label) == 2: incr += 1

        overwrite = False

        patterns_distinct = pd.DataFrame(columns=Labels.rel_labels + ['shift', 'delta_shift', 'labels'])
        for key in result_fin:
            # patterns_distinct.loc[len(patterns_distinct)] = result[key]
            # patterns_distinct = patterns_distinct.append( result[key], ignore_index=True)
            seq = pd.DataFrame(result_fin[key], columns=Labels.rel_labels + ['shift'])
            seq = seq.assign(labels=[int(key)] * len(seq))
            seq['delta_shift'] = seq['shift'].astype('int32') % 10 - seq['shift'].astype('int32') // 10
            patterns_distinct = patterns_distinct.append(seq)
        patterns_distinct = patterns_distinct[patterns_distinct['shift'].str.len() == 2]
        patterns_distinct['comment'] = 'extracted and automatically checked on the simulation data'
        # split sequences with a zero-row
        # patterns_distinct = patterns_distinct.append (dict(zip(patterns_all.columns, [0,0,0,0])), ignore_index=True)
        # patterns_distinct = patterns_distinct.drop('label',axis = 1).reset_index( drop = True)
        if overwrite:
            spark.createDataFrame(patterns_distinct).write.mode("overwrite").saveAsTable("Patterns_simple_maneuvers")