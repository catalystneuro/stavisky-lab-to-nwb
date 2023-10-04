import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import functools

import nwbwidgets.utils.timeseries as ts
from ndx_events import LabeledEvents
from ipywidgets import widgets, Output, FloatProgress, Layout
from ipydatagrid import DataGrid
from typing import Optional, Union
from pynwb.base import DynamicTable, TimeSeries, ProcessingModule
from pynwb.ecephys import ElectricalSeries
from pynwb.epoch import TimeIntervals
from nwbwidgets.utils.widgets import interactive_output, clean_axes
from nwbwidgets.controllers import (
    GroupAndSortController,
    StartAndDurationController,
    RangeController,
    make_trial_event_controller,
)
from nwbwidgets.timeseries import _prep_timeseries


class MultiTableWidget(widgets.VBox):
    def __init__(self, names, tables):
        assert len(names) == len(tables)
        
        super().__init__()
        self.names = names
        self.tables = tables
        
        if len(names) == 0:
            self.children = [widgets.HTML("No sessions found")]
            return
        if len(names) == 1:
            self.children = [DataGrid(tables[0])]
            return
        
        self.session_controller = widgets.Dropdown(
            options=names,
            value=names[0],
            description='Session:',
            disabled=False,
        )
        
        self.children = [self.session_controller, DataGrid(tables[0])]
        
        def on_dropdown_update(change):
            idx = names.index(change.new)
            self.children[1] = DataGrid(tables[idx])
        
        self.session_controller.observe(on_dropdown_update, "value")
        
    def update(self, session: str):
        idx = self.names.index(session)
        return DataGrid(self.tables[idx])


class DecodingErrorWidget(widgets.Tab):
    def __init__(
        self,
        trials: Union[TimeIntervals, list[TimeIntervals]],
    ):
        super().__init__()
        from .wer import mean_wer, mean_per, sentences_to_phonemes
        if isinstance(trials, TimeIntervals):
            trials = [trials]
        
        agg_name = []
        agg_wer = []
        agg_per = []
        agg_table = []
        for trial_table in trials:
            name = trial_table.get_ancestor("NWBFile").session_id
            wer, (sent_wer, sent_w) = mean_wer(
                trial_table.decoded_sentence[()].tolist(),
                trial_table.sentence_cue[()].tolist(), 
            )
            per, (sent_per, sent_p) = mean_per(
                trial_table.decoded_sentence[()].tolist(),
                trial_table.sentence_cue[()].tolist(), 
            )
            trnscrbd_decoded = sentences_to_phonemes(trial_table.decoded_sentence[()].tolist())
            trnscrbd_true = sentences_to_phonemes(trial_table.sentence_cue[()].tolist())
            data_dict = dict(
                true_sentence=trial_table.sentence_cue[()].tolist(),
                decoded_sentence=trial_table.decoded_sentence[()].tolist(),
                true_phonemes=trnscrbd_true,
                decoded_phonemes=trnscrbd_decoded,
                word_distance=sent_wer,
                word_count=sent_w,
                sentence_wer=[a/b for a, b in zip(sent_wer, sent_w)],
                phoneme_distance=sent_per,
                phoneme_count=sent_p,
                sentence_per=[a/b for a, b in zip(sent_per, sent_p)],
            )
            agg_name.append(name)
            agg_wer.append(wer)
            agg_per.append(per)
            agg_table.append(pd.DataFrame(data_dict))
        
        out = Output()
        with out:
            width = 1.0
            base_x = np.arange(len(agg_name)) * width * 3
            plt.bar(base_x - width/2, agg_wer, width=width, label='WER')
            plt.bar(base_x + width/2, agg_per, width=width, label='PER')
            plt.legend()
            plt.xlabel('Session')
            plt.xticks(ticks=base_x, labels=agg_name, rotation=0)
            plt.show()
        
        self.children = [out, MultiTableWidget(agg_name, agg_table)]
        self.titles = ("Overview", "Session Results")


def show_aligned_traces(
    data,
    start=-0.5,
    end=2.0,
    group_inds=None,
    labels=None,
    show_legend=True,
    ax_single=None,
    ax_mean=None,
    fontsize=12,
    overlay=True,
    figsize=(8, 6),
    gap_scale=10.,
):
    if not len(data):
        return ax_single, ax_mean

    data = np.stack(data, axis=1)
    tt = np.linspace(start, end, data.shape[0])
    gap = np.median(np.nanstd(data, axis=0)) * gap_scale

    legend_kwargs_single = dict()
    fig = plt.gcf()
    if (ax_single is None) or (ax_mean is None):
        fig.set_size_inches(*figsize)
        ax_single = fig.add_subplot(121)
        ax_mean = fig.add_subplot(122)
        if hasattr(fig, "canvas"):
            fig.canvas.header_visible = False
        else:
            legend_kwargs.update(bbox_to_anchor=(1.01, 1))

    if group_inds is not None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        ugroup_inds = np.unique(group_inds)
        handles_single = []
        handles_mean = []

        this_iter = enumerate(ugroup_inds)
        offset = 0
        for i, ui in this_iter:
            color = colors[ugroup_inds[i] % len(colors)]
            if overlay:
                lineoffset = np.zeros((np.sum(group_inds == ui),))
            else:
                lineoffset = np.arange(np.sum(group_inds == ui)) + offset
            line_collection_single = ax_single.plot(
                tt,
                data[:, group_inds == ui] + lineoffset[None, :] * gap,
                color=color,
            )
            handles_single.append(line_collection_single[0])
            offset = lineoffset[-1] + 1

            line_collection_mean = ax_mean.plot(
                tt,
                data[:, group_inds == ui] + (0 if overlay else i * gap),
                color=color,
            )
            handles_mean.append(line_collection_mean)
        if show_legend:
            ax_single.legend(
                handles=handles_single[::-1],
                labels=list(labels[ugroup_inds][::-1]),
                loc="upper left",
                bbox_to_anchor=(1.01, 1),
                **legend_kwargs,
            )
            ax_mean.legend(
                handles=handles_mean[::-1],
                labels=list(labels[ugroup_inds][::-1]),
                loc="upper left",
                bbox_to_anchor=(1.01, 1),
                **legend_kwargs,
            )
    else:
        if overlay:
            lineoffset = np.zeros((data.shape[1],))
        else:
            lineoffset = np.arange(data.shape[1])
        ax_single.plot(
            tt,
            data + lineoffset[None, :] * gap,
            color='black',
            linewidth=0.5,
        )
        
        ax_mean.plot(
            tt,
            data.mean(axis=1),
            color='black',
        )
    
    return ax_single, ax_mean

        
class AlignedAveragedTimeSeriesWidget(widgets.VBox):
    def __init__(
        self,
        time_series: TimeSeries,
        trials: TimeIntervals = None,
        processing_module: ProcessingModule = None,
        trace_index=0,
        trace_controller=None,
        trace_controller_kwargs=None,
        sem=True,
        gap_scale=10.,
    ):
        self.time_series = time_series
        self.time_series_data = time_series.data[()]
        self.time_series_timestamps = None
        self.sem = sem
        self.gap_scale = gap_scale
        if time_series.rate is None:
            self.time_series_timestamps = time_series.timestamps[()]
        super().__init__()

        if trials is None:
            self.trials = self.get_trials()
            if self.trials is None:
                self.children = [widgets.HTML("No trials present")]
                return
        else:
            self.trials = trials
        
        if processing_module is not None:
            self.processing_module = processing_module
            ts_keys = [
                key for key in processing_module.data_interfaces.keys() 
                if isinstance(processing_module.data_interfaces[key], TimeSeries)
            ]
            self.interface_controller = widgets.Dropdown(
                options=ts_keys,
                value=ts_keys[0],
                description="interface:",
            )
        else:
            self.processing_module = None
            self.interface_controller = None

        if trace_controller is None:
            ntraces = self.time_series.data.shape[1]
            input_trace_controller_kwargs = dict(
                options=[x for x in range(ntraces)],
                value=trace_index,
                description="trace:",
                layout=Layout(width="200px"),
            )
            if trace_controller_kwargs is not None:
                input_trace_controller_kwargs.update(trace_controller_kwargs)
            self.trace_controller = widgets.Dropdown(**input_trace_controller_kwargs)
        else:
            self.trace_controller = trace_controller

        self.trial_event_controller = make_trial_event_controller(
            self.trials, layout=Layout(width="200px"), multiple=True
        )
        self.start_ft = widgets.FloatText(
            -0.5,
            step=0.1,
            description="start (s):",
            layout=Layout(width="200px"),
            tooltip="Start time for calculation before or after (negative or positive) the reference point (aligned to)",
        )

        self.end_ft = widgets.FloatText(
            1.0,
            step=0.1,
            description="end (s):",
            layout=Layout(width="200px"),
            tooltip="End time for calculation before or after (negative or positive) the reference point (aligned to).",
        )

        self.gas = self.make_group_and_sort(window=False, control_order=False)

        self.overlay_cb = widgets.Checkbox(description="overlay traces")

        self.controls = dict(
            index=self.trace_controller,
            end=self.end_ft,
            start=self.start_ft,
            start_labels=self.trial_event_controller,
            gas=self.gas,
            overlay=self.overlay_cb,
        )
        if self.interface_controller is not None:
            self.controls.update(interface=self.interface_controller)

        out_fig = interactive_output(self.update, self.controls)

        self.children = [
            widgets.HBox(
                [
                    widgets.VBox(
                        [
                            self.gas,
                            self.overlay_cb,
                        ]
                    ),
                    widgets.VBox(
                        ([self.interface_controller] if self.interface_controller is not None else []) + 
                        [
                            self.trace_controller,
                            self.trial_event_controller,
                            self.start_ft,
                            self.end_ft,
                        ]
                    ),
                ]
            ),
            out_fig,
        ]

    def get_trials(self):
        return self.time_series.get_ancestor("NWBFile").trials

    def make_group_and_sort(self, window=None, control_order=False):
        return GroupAndSortController(self.trials, window=window, control_order=control_order)

    def update(
        self,
        index: int,
        start_labels: tuple = ("start_time",),
        start: float = 0.0,
        end: float = 1.0,
        order=None,
        group_inds=None,
        labels=None,
        interface=None,
        figsize=(12, 7),
        overlay=True,
        align_line_color=(0.7, 0.7, 0.7),
    ):
        if interface is not None:
            self.time_series = self.processing_module.data_interfaces[interface]
        fig, axs = plt.subplots(2, len(start_labels), figsize=figsize, sharex=True)
        clean_axes(axs.ravel())

        ax1_ylims = []
        for i_s, start_label in enumerate(start_labels):
            if len(start_labels) > 1:
                ax0 = axs[0, i_s]
                ax1 = axs[1, i_s]
            else:
                ax0 = axs[0]
                ax1 = axs[1]

            data = ts.align_by_time_intervals(
                self.time_series,
                self.trials,
                start_label,
                start,
                end,
                index,
            )

            if i_s == len(start_labels) - 1:
                show_legend = True
            else:
                show_legend = False

            show_aligned_traces(
                data,
                start,
                end,
                group_inds,
                labels,
                show_legend=show_legend,
                ax_single=ax0,
                ax_mean=ax1,
                fontsize=12,
                overlay=overlay,
                gap_scale=self.gap_scale,
            )

            ax0.set_title(f"{start_label}")
            ax0.set_xticks([])
            ax0.set_xlabel("")
            ax0.set_yticks([])
            if not overlay:
                ax0.set_ylabel("trials")

            ax1.set_xlim([start, end])
            ax1.set_ylabel("mean", fontsize=12)
            ax1.set_xlabel("time (s)", fontsize=12)
            ax1.axvline(color=align_line_color)
            ax1_ylims.append(ax1.get_ylim())

        if len(start_labels) > 1:
            # Adjust bottom axes y axis
            min_y = np.min(np.array(ax1_ylims)[:, 0])
            max_y = np.max(np.array(ax1_ylims)[:, 1])
            for i_b, ax_btm in enumerate(axs[1, :]):
                ax_btm.set_ylim(min_y, max_y)
                if i_b > 0:
                    ax_btm.set_ylabel("")
                    # After adjusting ylims we can avoid showing tick labels
                    ax_btm.set_yticklabels([])

        fig.suptitle(f"Trace {index}", fontsize=15)
        fig.subplots_adjust(wspace=0.3)
        return fig


PHONEMES = ["\" \"", "<s>", "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", 
"D", "DH", "EH", "ER", "EY", "F", "G", "HH", "IH", 
"IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", 
"P", "R", "S", "SH", "T", "TH", "UH", "UW", "V", 
"W", "Y", "Z", "ZH"]

strikethrough = lambda x: ''.join([(c + '\u0336') for c in x])

def plot_overlaid_traces(
    time_series: TimeSeries,
    events=None,
    time_window=None,
    order=None,
    figsize=(8, 6),
    group_inds=None,
    labels=None,
    cmap=None,
    show_legend=True,
    dynamic_table_region_name=None,
    window=None,
    transform=functools.partial(scipy.special.softmax, axis=1),
    text_diff=True,
):
    if events is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        figsize = (figsize[0], figsize[1] * 1.1)
        fig, (ax_ev, ax) = plt.subplots(2, 1, figsize=figsize, height_ratios=[1, 10], sharex=True)
        ax_ev.set_ylim(-0.5, 0.5)
        ax_ev.spines['top'].set_visible(False)
        ax_ev.spines['right'].set_visible(False)
        ax_ev.spines['bottom'].set_position('center')
        ax_ev.spines['left'].set_visible(False)
        ax_ev.set_yticks([])
    
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(*time_window)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("probs")
    
    if events is not None:
        if hasattr(events, "labels"):
            event_labels = events.labels
        else:
            event_labels = events.data__labels
    
    if text_diff:
        trials = events.get_ancestor("NWBFile").trials
        if trials is None:
            text_diff = False
    
    if cmap is None:
        cmap = plt.cm.rainbow

    if order is None:
        if len(time_series.data.shape) > 1:
            order = np.arange(time_series.data.shape[1])
        else:
            order = [0]

    if group_inds is not None:
        row_ids = getattr(time_series, dynamic_table_region_name).data[:]
        channel_inds = [np.argmax(row_ids == x) for x in order]
    elif window is not None:
        order = order[window[0] : window[1]]
        channel_inds = order
    else:
        channel_inds = order

    if len(channel_inds):
        mini_data, tt, offsets = _prep_timeseries(time_series, time_window, channel_inds)
    else:
        mini_data = None
        tt = time_window

    if mini_data is None or len(mini_data) == 0:
        ax.plot(tt, np.ones_like(tt) * np.nan, color="k")
        return

    mini_data -= offsets
    mini_data = transform(mini_data)

    for i in range(mini_data.shape[1]):
        ax.plot(tt, mini_data[:, i], c=cmap(i / mini_data.shape[1]))
        
    max_prob_idx = np.argmax(mini_data, axis=1)
    i = 0
    while i < len(max_prob_idx):
        n = 1
        while ((i + n) < len(max_prob_idx)) and (max_prob_idx[i] == max_prob_idx[i + n]):
            n += 1
        text_idx = i + np.argmax(mini_data[i:i+n, max_prob_idx[i]])
        max_phoneme = PHONEMES[order[max_prob_idx[text_idx]]]
        ax.text(tt[text_idx], mini_data[text_idx, max_prob_idx[text_idx]] + 0.02, max_phoneme, ha="center")
        i += n
    
    if events is not None:
        if time_window is None:
            event_ind_start = 0
            event_ind_stop = events.data.shape[0]
        else:
            event_ind_start = ts.timeseries_time_to_ind(events, time_window[0])
            event_ind_stop = ts.timeseries_time_to_ind(events, time_window[1])
    if (event_ind_stop - event_ind_start) > 0:
        for i in range(event_ind_start, event_ind_stop):
            ax_ev.scatter(events.timestamps[i], 0, marker="|", color="black")
            event_label = event_labels[events.data[i].astype(int)]
            if text_diff:
                past_trial_idx = np.searchsorted(trials.stop_time[()], events.timestamps[i-1])
                trial_idx = np.searchsorted(trials.stop_time[()], events.timestamps[i])
                if past_trial_idx == trial_idx:
                    if i > 0:
                        past_labels = event_labels[events.data[i-1].astype(int)].split()
                        for pl in past_labels:
                            if pl in event_label:
                                event_label = event_label.partition(pl)[2].strip()
                final_sentence = trials.decoded_sentence[trial_idx]
                event_label = " ".join([
                    (el if el in final_sentence else strikethrough(el))
                    for el in event_label.split()
                ])
            ax_ev.text(events.timestamps[i], 0.5 * ((i % 2) - 0.5), event_label, ha='center', va='center')
    
    return fig


class DecodingOutputWidget(widgets.HBox):
    def __init__(
        self,
        time_series: TimeSeries,
        events: LabeledEvents = None,
        dynamic_table_region_name=None,
        foreign_time_window_controller: StartAndDurationController = None,
        foreign_group_and_sort_controller: GroupAndSortController = None,
        mpl_plotter=plot_overlaid_traces,
    ):
        """

        Parameters
        ----------
        time_series: TimeSeries
        dynamic_table_region_name: str, optional
        foreign_time_window_controller: StartAndDurationController, optional
        foreign_group_and_sort_controller: GroupAndSortController, optional
        mpl_plotter: function
            Choose function to use when creating figures
        kwargs
        """

        if dynamic_table_region_name is not None and foreign_group_and_sort_controller is not None:
            raise TypeError(
                "You cannot supply both `dynamic_table_region_name` and `foreign_group_and_sort_controller`."
            )

        super().__init__()
        self.time_series = time_series
        self.events = events

        if foreign_time_window_controller is not None:
            self.time_window_controller = foreign_time_window_controller
        else:
            self.tmin = ts.get_timeseries_mint(time_series)
            self.tmax = ts.get_timeseries_maxt(time_series)
            self.time_window_controller = StartAndDurationController(tmin=self.tmin, tmax=self.tmax)

        self.controls = dict(
            time_series=widgets.fixed(self.time_series),
            events=widgets.fixed(self.events),
            time_window=self.time_window_controller,
            dynamic_table_region_name=widgets.fixed(dynamic_table_region_name),
        )
        if foreign_group_and_sort_controller is None:
            if dynamic_table_region_name is not None:
                dynamic_table_region = getattr(time_series, dynamic_table_region_name)
                table = dynamic_table_region.table
                referenced_rows = np.array(dynamic_table_region.data)
                self.gas = GroupAndSortController(
                    dynamic_table=table,
                    keep_rows=referenced_rows,
                )
                self.controls.update(gas=self.gas)
            else:
                self.gas = None
                range_controller_max = min(41, self.time_series.data.shape[1])
                self.range_controller = RangeController(
                    0,
                    self.time_series.data.shape[1],
                    start_value=(0, range_controller_max),
                    dtype="int",
                    description="traces",
                    orientation="vertical",
                )
                self.controls.update(window=self.range_controller)
        else:
            self.gas = foreign_group_and_sort_controller
            self.controls.update(gas=self.gas)

        # Sets up interactive output controller
        out_fig = interactive_output(mpl_plotter, self.controls)

        if foreign_time_window_controller:
            right_panel = out_fig
        else:
            right_panel = widgets.VBox(
                children=[
                    self.time_window_controller,
                    out_fig,
                ],
                layout=widgets.Layout(width="100%"),
            )

        if foreign_group_and_sort_controller or self.gas is None:
            if self.range_controller is None:
                self.children = [right_panel]
            else:
                self.children = [self.range_controller, right_panel]
        else:
            self.children = [self.gas, right_panel]


def phoneme_timeintervals(
    rnn_logits: TimeSeries,
    min_consecutive=4,
):
    ti = TimeIntervals(name="phoneme_alignment")
    ti.add_column(name="phoneme", description="phoneme")
    ti.add_column(name="start_idx", description="start_idx")
    timestamps = rnn_logits.timestamps
    if timestamps is None:
        timestamps = np.arange(len(rnn_logits)) * rnn_logits.rate + rnn_logits.starting_time
    preds = rnn_logits.data[()].argmax(axis=1)
    i = 0
    while (i + min_consecutive) < len(preds): # iterative approach, too slow?
        if preds[i] == 0 or preds[i] == 1:
            n = 1
        elif np.all(np.isin(preds[i:(i+min_consecutive)], np.array([1, preds[i]]))):
            n = min_consecutive + 1
            while ((i + n) < len(preds)) and (preds[i + n] == preds[i]):
                n += 1
            ti.add_row(
                start_time=timestamps[i],
                stop_time=timestamps[min(i+n, len(timestamps)-1)],
                start_idx=i,
                phoneme=PHONEMES[preds[i].astype(int)],
            )
        else:
            n = 1
        i += n
    return ti