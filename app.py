import atexit
import gc
import json
import logging
import multiprocessing
import os
import sys
import threading
import time
import tkinter as tk
import tkinter.filedialog
import tkinter.messagebox
from queue import Queue
from tkinter import ttk
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

from CHClab import CV, IVscan, STM_bj


class Queue_Item:

    def __init__(self, func, *args, **kwargs) -> None:
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        self.func(*self.args, **self.kwargs)


class Main:

    def __init__(self) -> None:
        self.window = root
        self.window.title('STM histogram')
        self.window.protocol("WM_DELETE_WINDOW", sys.exit)
        self.window.resizable(False, False)
        frame = tk.Frame(self.window)
        frame.grid(row=0, column=0, sticky='nw')
        tk.Label(frame, text='Experiment: ').pack(side='left')
        self.option = tk.StringVar(self.window, value='Select')
        tk.OptionMenu(frame, self.option, *['STM bj', 'IV scan', 'CV'], command=self.new_tab).pack(side='left')
        self.CPU_threads = tk.IntVar(self.window, value=multiprocessing.cpu_count())
        self.always_on_top = tk.BooleanVar(self.window, value=False)
        tk.Label(frame, text='CPU threads: ').pack(side='left')
        tk.Entry(frame, textvariable=self.CPU_threads, width=10, justify='center').pack(side='left')
        tk.Checkbutton(frame, variable=self.always_on_top, text="Always on top", command=self.on_top).pack(side='left')
        self.tabcontrol = ttk.Notebook(self.window)
        self.tabcontrol.grid(row=1, columnspan=2, sticky='nw')
        tk.Button(self.window, text='Log', command=handler.show).grid(row=0, column=1, sticky='ne')
        tk.Button(self.window, text='❌', fg='red', command=self.close_tab).grid(row=0, column=2, padx=10, sticky='ne')
        self.window.bind("<Control-t>", lambda *args: self.new_tab(self.option.get()))
        self.window.bind("<Control-w>", self.close_tab)
        self.rename_text = tk.StringVar(self.tabcontrol)
        self.rename_entry = tk.Entry(self.tabcontrol, textvariable=self.rename_text, bg='grey')
        self.tabcontrol.bind("<Button-3>", self.tab_rename_start)
        self.window.bind("<F2>", self.tab_rename_start)
        self.rename_entry.bind("<Return>", self.tab_rename_stop)
        self.rename_entry.bind("<Escape>", self.tab_rename_cancel)

    def new_tab(self, name: str):
        match name:
            case 'STM bj':
                tab = ttk.Frame(self.tabcontrol)
                self.tabcontrol.add(tab, text=name)
                self.tabcontrol.select(tab)
                tab.gui_object = STM_bj_GUI(tab)
            case 'IV scan':
                tab = ttk.Frame(self.tabcontrol)
                self.tabcontrol.add(tab, text=name)
                self.tabcontrol.select(tab)
                tab.gui_object = IVscan_GUI(tab)
            case 'CV':
                tab = ttk.Frame(self.tabcontrol)
                self.tabcontrol.add(tab, text=name)
                self.tabcontrol.select(tab)
                tab.gui_object = CV_GUI(tab)

    def tab_rename_start(self, *args):
        try:
            self.tabcontrol.index('current')
        except:
            return
        self.rename_entry.place(width=100, height=20, x=0, y=0, anchor="nw")
        self.rename_text.set(self.tabcontrol.tab(self.tabcontrol.index('current'))['text'])
        self.rename_entry.focus_set()
        self.rename_entry.grab_set()

    def tab_rename_stop(self, *args):
        self.tabcontrol.tab(self.tabcontrol.index('current'), text=self.rename_entry.get())
        self.rename_entry.place_forget()
        self.rename_entry.grab_release()

    def tab_rename_cancel(self, *args):
        self.rename_entry.delete(0, "end")
        self.rename_entry.place_forget()
        self.rename_entry.grab_release()

    def close_tab(self, *args):
        try:
            tab = self.tabcontrol.nametowidget(self.tabcontrol.select())
            tab.gui_object.cleanup('all')
            tab.destroy()
            del tab.gui_object
            self.window.update_idletasks()
            gc.collect()
        except:
            return

    def on_top(self):
        self.window.attributes('-topmost', self.always_on_top.get())
        self.window.update_idletasks()


def _set_directory(var: tk.StringVar, value: str):
    if value == '""' or value == '': return
    else: var.set(value)


class STM_bj_GUI:

    def __init__(self, root: tk.Frame) -> None:
        self.window = root
        self.export_prompt = STM_bj_export_prompt(self)
        # config frame
        self.frame_config = tk.Frame(self.window)
        self.frame_config.pack(side='top', anchor='w')
        # row 0
        self.directory_path = tk.StringVar(self.window)
        tk.Label(self.frame_config, text='Path: ').grid(row=0, column=0)
        tk.Entry(self.frame_config, textvariable=self.directory_path, width=80).grid(row=0, column=1, columnspan=5)
        tk.Button(self.frame_config, text="Files", bg='#ffe9a2', command=lambda: _set_directory(self.directory_path, json.dumps(tkinter.filedialog.askopenfilenames(), ensure_ascii=False))).grid(row=0, column=6)
        tk.Button(self.frame_config, text="Folder", bg='#ffe9a2', command=lambda: _set_directory(self.directory_path, tkinter.filedialog.askdirectory())).grid(row=0, column=7, padx=5)
        # row 1
        self.extract_length = tk.IntVar(self.window, value=2000)
        self.upper = tk.DoubleVar(self.window, value=3.2)
        self.lower = tk.DoubleVar(self.window, value=1e-6)
        self.is_raw = tk.StringVar(self.window, value='raw')
        self.directory_recursive = tk.BooleanVar(self.window, value=False)
        tk.Label(self.frame_config, text='Length: ').grid(row=1, column=0)
        tk.Entry(self.frame_config, textvariable=self.extract_length, justify='center').grid(row=1, column=1)
        tk.Label(self.frame_config, text='G upper: ').grid(row=1, column=2)
        tk.Entry(self.frame_config, textvariable=self.upper, justify='center').grid(row=1, column=3)
        tk.Label(self.frame_config, text='G lower: ').grid(row=1, column=4)
        tk.Entry(self.frame_config, textvariable=self.lower, justify='center').grid(row=1, column=5)
        tk.OptionMenu(self.frame_config, self.is_raw, *['raw', 'cut']).grid(row=1, column=6)
        tk.Checkbutton(self.frame_config, variable=self.directory_recursive, text="Recursive").grid(row=1, column=7, sticky='w')
        # row 2
        self.zero_point = tk.DoubleVar(self.window, value=0.5)
        self.points_per_nm = tk.DoubleVar(self.window, value=800)
        self.direction = tk.StringVar(self.window, value='pull')
        tk.Label(self.frame_config, text='X=0@G= ').grid(row=2, column=0)
        tk.Entry(self.frame_config, textvariable=self.zero_point, justify='center').grid(row=2, column=1)
        tk.Label(self.frame_config, text='Points/nm: ').grid(row=2, column=2)
        tk.Entry(self.frame_config, textvariable=self.points_per_nm, justify='center').grid(row=2, column=3)
        tk.Label(self.frame_config, text='Direction: ').grid(row=2, column=4)
        tk.OptionMenu(self.frame_config, self.direction, *['pull', 'crash', 'both']).grid(row=2, column=5)
        # row 3
        self.G_min = tk.DoubleVar(self.window, value=0.00001)
        self.G_max = tk.DoubleVar(self.window, value=3.16)
        self.G_bins = tk.IntVar(self.window, value=550)
        self.G_scale = tk.StringVar(self.window, value='log')
        tk.Label(self.frame_config, text='G min: ').grid(row=3, column=0)
        tk.Entry(self.frame_config, textvariable=self.G_min, justify='center').grid(row=3, column=1)
        tk.Label(self.frame_config, text='G max: ').grid(row=3, column=2)
        tk.Entry(self.frame_config, textvariable=self.G_max, justify='center').grid(row=3, column=3)
        tk.Label(self.frame_config, text='G #bins: ').grid(row=3, column=4)
        tk.Entry(self.frame_config, textvariable=self.G_bins, justify='center').grid(row=3, column=5)
        tk.Label(self.frame_config, text='G scale: ').grid(row=3, column=6)
        tk.OptionMenu(self.frame_config, self.G_scale, *['log', 'linear']).grid(row=3, column=7)
        # row 4
        self.X_min = tk.DoubleVar(self.window, value=-0.4)
        self.X_max = tk.DoubleVar(self.window, value=0.6)
        self.X_bins = tk.IntVar(self.window, value=1000)
        self.X_scale = tk.StringVar(self.window, value='linear')
        tk.Label(self.frame_config, text='X min: ').grid(row=4, column=0)
        tk.Entry(self.frame_config, textvariable=self.X_min, justify='center').grid(row=4, column=1)
        tk.Label(self.frame_config, text='X max: ').grid(row=4, column=2)
        tk.Entry(self.frame_config, textvariable=self.X_max, justify='center').grid(row=4, column=3)
        tk.Label(self.frame_config, text='X #bins: ').grid(row=4, column=4)
        tk.Entry(self.frame_config, textvariable=self.X_bins, justify='center').grid(row=4, column=5)
        tk.Label(self.frame_config, text='X scale: ').grid(row=4, column=6)
        tk.OptionMenu(self.frame_config, self.X_scale, *['log', 'linear']).grid(row=4, column=7)
        # row 5
        self.t_min = tk.DoubleVar(self.window, value=0)
        self.t_max = tk.DoubleVar(self.window, value=3600)
        self.t_bin_size = tk.IntVar(self.window, value=30)
        self.t_scale = tk.StringVar(self.window, value='linear')
        tk.Label(self.frame_config, text='t min: ').grid(row=5, column=0)
        tk.Entry(self.frame_config, textvariable=self.t_min, justify='center').grid(row=5, column=1)
        tk.Label(self.frame_config, text='t max: ').grid(row=5, column=2)
        tk.Entry(self.frame_config, textvariable=self.t_max, justify='center').grid(row=5, column=3)
        tk.Label(self.frame_config, text='t bin size: ').grid(row=5, column=4)
        tk.Entry(self.frame_config, textvariable=self.t_bin_size, justify='center').grid(row=5, column=5)
        tk.Label(self.frame_config, text='t scale: ').grid(row=5, column=6)
        tk.OptionMenu(self.frame_config, self.t_scale, *['log', 'linear']).grid(row=5, column=7)
        # row 6
        tk.Label(self.frame_config, text='Colorbar: ').grid(row=6, column=0)
        self.colorbar_conf = tk.Text(self.frame_config, height=3, wrap='none', undo=True, maxundo=-1)
        self.colorbar_conf.bind('<<Modified>>', self.colorbar_apply)
        self.colorbar_conf.grid(row=6, column=1, columnspan=5, sticky='w')
        self.colorbar_conf.insert('0.0', '{"red":  [[0,1,1],[0.05,0,0],[0.1,0,0],[0.15,1,1],[0.3,1,1],[1,1,1]],\n "green":[[0,1,1],[0.05,0,0],[0.1,1,1],[0.15,1,1],[0.3,0,0],[1,0,0]],\n "blue": [[0,1,1],[0.05,1,1],[0.1,0,0],[0.15,0,0],[0.3,0,0],[1,1,1]]}')
        self.run_button = tk.Button(self.frame_config, text='Run', bg='lime', command=self.run)
        self.run_button.grid(row=6, column=6, padx=10)
        self.is_run = False
        try:
            import yaml
            tk.Button(self.frame_config, text='Import', command=self.import_setting).grid(row=6, column=7)
        except ImportError:
            logger.warning('Module PyYAML was not found. Import/export settings can not be used.')
        tk.Button(self.frame_config, text='Export', command=self.export_prompt.show).grid(row=6, column=8)
        # is_plot frame
        self.frame_is_plot = tk.Frame(self.window)
        self.frame_is_plot.pack(side='top', anchor='w')
        self.plot_hist_G = tk.BooleanVar(self.window, value=True)
        self.plot_hist_GS = tk.BooleanVar(self.window, value=True)
        self.plot_hist_Gt = tk.BooleanVar(self.window, value=False)
        self.plot_2DCH = tk.BooleanVar(self.window, value=False)
        tk.Label(self.frame_is_plot, text='Plot: ').pack(side='left')
        tk.Checkbutton(self.frame_is_plot, text='Histogram G', variable=self.plot_hist_G).pack(side='left')
        tk.Checkbutton(self.frame_is_plot, text='Histogram GS', variable=self.plot_hist_GS).pack(side='left')
        tk.Checkbutton(self.frame_is_plot, text='Histogram Gt', variable=self.plot_hist_Gt).pack(side='left')
        tk.Checkbutton(self.frame_is_plot, text='Cross-correlation', variable=self.plot_2DCH).pack(side='left')
        # figure frame
        self.frame_figure = tk.Frame(self.window)
        self.frame_figure.pack(side='top', anchor='w')
        self.frame_figure.columnconfigure([0, 5, 10, 15], weight=1)
        self.frame_figure.rowconfigure([0], weight=1)
        # status frame
        self.frame_status = tk.Frame(self.window)
        self.frame_status.pack(side='bottom', anchor='w')
        tk.Label(self.frame_status, text='#Traces: ').pack(side='left')
        self.status_traces = tk.Label(self.frame_status, text=0)
        self.status_traces.pack(side='left')
        tk.Label(self.frame_status, text='File: ', padx=20).pack(side='left')
        self.status_last_file = tk.Label(self.frame_status, text='Waiting')
        self.status_last_file.pack(side='left')
        self.queue = Queue()
        self.updatetk()
        if ini_config:
            if 'IVscan' in ini_config:
                self.import_setting(ini_config['STM-bj'])

    def colorbar_apply(self, *args):
        try:
            colorbar_conf = self.colorbar_conf.get('0.0', 'end')
            if colorbar_conf != "\n":
                cmap = json.loads(colorbar_conf)
                if self.run_config['plot_hist_GS']:
                    self.hist_GS.set_cmap(cmap=cmap)
                    self.canvas_GS.draw_idle()
                if self.run_config['plot_hist_Gt']:
                    self.hist_Gt.set_cmap(cmap=cmap)
                    self.canvas_Gt.draw_idle()
        except Exception as E:
            return
        finally:
            self.colorbar_conf.edit_modified(False)

    def run(self):
        match self.is_run:
            case False:
                self.cleanup('partial')
                self.run_config = {
                    "length": self.extract_length.get(),
                    "upper": self.upper.get(),
                    "lower": self.lower.get(),
                    "method": self.direction.get(),
                    "zero_point": self.zero_point.get(),
                    "x_conversion": self.points_per_nm.get(),
                    'G_scale': self.G_scale.get(),
                    'X_scale': self.X_scale.get(),
                    't_scale': self.t_scale.get(),
                    'recursive': self.directory_recursive.get(),
                    'data_type': self.is_raw.get(),
                    'plot_hist_G': self.plot_hist_G.get(),
                    'plot_hist_GS': self.plot_hist_GS.get(),
                    'plot_hist_Gt': self.plot_hist_Gt.get(),
                    'plot_2DCH': self.plot_2DCH.get()
                }
                path = self.directory_path.get()
                if not os.path.isdir(path):
                    try:
                        path = json.loads(path)
                    except Exception as E:
                        tkinter.messagebox.showerror('Error', 'Invalid directory')
                        return
                for item in self.frame_figure.winfo_children():
                    item.destroy()
                gc.collect()
                self.G = np.empty((0, self.run_config['length']))
                self.X = np.empty((0, self.run_config['length']))
                # hist G
                if self.run_config['plot_hist_G']:
                    self.hist_G = STM_bj.Hist_G([self.G_min.get(), self.G_max.get()], self.G_bins.get(), self.G_scale.get())
                    self.canvas_G = FigureCanvasTkAgg(self.hist_G.fig, self.frame_figure)
                    self.canvas_G.get_tk_widget().grid(row=0, column=0, columnspan=5, pady=10)
                    self.navtool_G = NavigationToolbar2Tk(self.canvas_G, self.frame_figure, pack_toolbar=False)
                    self.navtool_G.grid(row=1, column=0, columnspan=4, sticky='w')
                    self.autoscale_G = tk.BooleanVar(self.window, value=True)
                    tk.Checkbutton(self.frame_figure, variable=self.autoscale_G, text="Autoscale").grid(row=2, column=0, sticky='w')
                    self.canvas_G.draw_idle()
                # hist GS
                if self.run_config['plot_hist_GS']:
                    self.hist_GS = STM_bj.Hist_GS([self.X_min.get(), self.X_max.get()], [self.G_min.get(), self.G_max.get()], self.X_bins.get(), self.G_bins.get(), self.X_scale.get(), self.G_scale.get(), self.zero_point.get(), self.points_per_nm.get())
                    self.canvas_GS = FigureCanvasTkAgg(self.hist_GS.fig, self.frame_figure)
                    self.canvas_GS.get_tk_widget().grid(row=0, column=5, columnspan=5, pady=10)
                    self.navtool_GS = NavigationToolbar2Tk(self.canvas_GS, self.frame_figure, pack_toolbar=False)
                    self.navtool_GS.grid(row=1, column=5, columnspan=4, sticky='w')
                    self.canvas_GS.draw_idle()
                # hist Gt
                if (self.is_raw.get() == 'raw') & self.run_config['plot_hist_Gt']:
                    self.hist_Gt = STM_bj.Hist_Gt([self.t_min.get(), self.t_max.get()], [self.G_min.get(), self.G_max.get()], self.t_bin_size.get(), self.G_bins.get(), self.t_scale.get(), self.G_scale.get())
                    self.canvas_Gt = FigureCanvasTkAgg(self.hist_Gt.fig, self.frame_figure)
                    self.canvas_Gt.get_tk_widget().grid(row=0, column=10, columnspan=5, pady=10)
                    self.navtool_Gt = NavigationToolbar2Tk(self.canvas_Gt, self.frame_figure, pack_toolbar=False)
                    self.navtool_Gt.grid(row=1, column=10, columnspan=4, sticky='w')
                    self.canvas_Gt.draw_idle()
                # hist 2DCH
                if self.run_config['plot_2DCH']:
                    self.hist_2DCH = STM_bj.Hist_Correlation([self.G_min.get(), self.G_max.get()], self.G_bins.get(), self.G_scale.get())
                    self.canvas_2DCH = FigureCanvasTkAgg(self.hist_2DCH.fig, self.frame_figure)
                    self.canvas_2DCH.get_tk_widget().grid(row=0, column=15, columnspan=5, pady=10)
                    self.navtool_2DCH = NavigationToolbar2Tk(self.canvas_2DCH, self.frame_figure, pack_toolbar=False)
                    self.navtool_2DCH.grid(row=1, column=15, columnspan=4, sticky='w')
                    self.canvas_2DCH.draw_idle()
                # Trace GS
                if self.run_config['plot_hist_GS']:

                    def show_trace_GS(action: Literal['next', 'prev', 'show', 'clear']):
                        trace = self.current_trace_GS.get()
                        if action == 'show':
                            if self.plot_trace_GS.get():
                                if -self.G.shape[0] <= trace < self.G.shape[0]:
                                    if self.current_lines_GS: self.current_lines_GS.set_data(self.X[trace], self.G[trace])
                                    else: self.current_lines_GS = self.hist_GS.ax.plot(self.X[trace], self.G[trace], color='k')[0]
                                    self.canvas_GS.draw_idle()
                        elif action == 'next':
                            if -self.G.shape[0] <= trace + 1 < self.G.shape[0]:
                                self.current_trace_GS.set(trace + 1)
                        elif action == 'prev':
                            if -self.G.shape[0] <= trace - 1 < self.G.shape[0]:
                                self.current_trace_GS.set(trace - 1)
                        elif action == 'clear':
                            if self.current_lines_GS:
                                self.current_lines_GS.remove()
                                self.current_lines_GS = None
                                self.canvas_GS.draw_idle()

                    self.plot_trace_GS = tk.BooleanVar(self.window, value=False)
                    self.current_trace_GS = tk.IntVar(self.window, value=-1)
                    self.current_lines_GS = None
                    frame_trace_GS = tk.Frame(self.frame_figure)
                    frame_trace_GS.grid(row=2, column=5, sticky='w')
                    tk.Checkbutton(frame_trace_GS, text="Trace: ", variable=self.plot_trace_GS).pack(side='left', anchor='w')
                    tk.Button(frame_trace_GS, text='<', command=lambda: show_trace_GS('prev')).pack(side='left', anchor='w')
                    current_trace_GS_entry = tk.Entry(frame_trace_GS, textvariable=self.current_trace_GS, justify='center', width=8)
                    current_trace_GS_entry.pack(side='left', anchor='w')
                    tk.Button(frame_trace_GS, text='>', command=lambda: show_trace_GS('next')).pack(side='left', anchor='w')
                    current_trace_GS_entry.bind("<Down>", lambda *args: show_trace_GS('prev'))
                    current_trace_GS_entry.bind("<Up>", lambda *args: show_trace_GS('next'))
                    self.plot_trace_GS.trace_add('write', lambda *args: show_trace_GS('show' if self.plot_trace_GS.get() else 'clear'))
                    self.current_trace_GS.trace_add('write', lambda *args: show_trace_GS('show'))
                self.colorbar_apply()
                self.status_traces.config(text=0)
                self.window.update_idletasks()
                self._lock = threading.Lock()
                self._lock.acquire()
                threading.Thread(target=self.add_data, args=(path, )).start()
                if isinstance(path, list): return
                try:
                    from watchdog.observers import Observer
                    self.observer = Observer()
                    self.observer.schedule(self.FileHandler(self), path=path, recursive=self.run_config['recursive'])
                    self.observer.start()
                    logger.info(f'Start observer: {path}')
                    atexit.register(self.observer.stop)
                    self.run_button.config(text='Stop', bg='red')
                    self.is_run = True
                except ImportError:
                    logger.warning('Module watchdog was not found. Data can not be updated in realtime.')
            case True:
                self.run_button.config(text='Run', bg='lime')
                self.is_run = False
                threading.Thread(target=self.observer.stop).start()
                logger.info(f'Stop observer')
                gc.collect()

    try:
        from watchdog.events import FileSystemEventHandler

        class FileHandler(FileSystemEventHandler):

            def __init__(self, GUI) -> None:
                self.GUI = GUI

            def on_created(self, event):
                from watchdog.events import FileCreatedEvent
                if isinstance(event, FileCreatedEvent):
                    if (event.src_path.endswith('.txt')):
                        try:
                            with self.GUI._lock:
                                if os.path.getsize(event.src_path) == 0:
                                    time.sleep(0.5)
                                    if os.path.getsize(event.src_path) == 0: raise RuntimeWarning('Empty file')
                            self.GUI.add_data(event.src_path)
                        except Exception as E:
                            logger.warning(f'Add data failed: {event.src_path}: {type(E).__name__}: {E.args}')
    except ImportError:
        pass

    def add_data(self, path: str | list):
        if isinstance(path, str):
            self.queue.put(Queue_Item(self.status_last_file.config, text=path, bg='yellow'))
            if os.path.isdir(path):
                if not os.listdir(path):  # empty directory
                    self.queue.put(Queue_Item(self.status_last_file.config, bg='lime'))
                    self._lock.release()
                    return
        else:
            self.queue.put(Queue_Item(self.status_last_file.config, text=f"{len(path)} files" if len(path) > 1 else path[0], bg='yellow'))
        try:
            logger.debug(f'Add data: {path}')
            match self.run_config['data_type']:
                case 'raw':
                    df = STM_bj.load_data_with_metadata(path, **self.run_config, max_workers=GUI.CPU_threads.get())
                    df['extracted'] = df['data'].apply(lambda g: STM_bj.extract_data(g, **self.run_config))
                    G = np.concatenate(df['extracted'].values)
                    time = np.repeat(df['time'].values, df['extracted'].apply(lambda g: g.shape[0]).values)
                    if not hasattr(self, 'time_init'):
                        self.time_init = time.min()
                        if self._lock.locked(): self._lock.release()
                case 'cut':
                    df = STM_bj.load_data_with_metadata(path, **self.run_config, max_workers=GUI.CPU_threads.get())['data']
                    length = df.apply(lambda x: x.shape[-1])
                    max_length = max(*length, self.run_config['length'])
                    df[length < max_length] = df[length < max_length].apply(lambda x: np.pad(x, (0, max_length - x.shape[-1]), 'constant', constant_values=0))
                    G = np.stack(df)
                    self.G = np.empty((0, max_length))
                    self.X = np.empty((0, max_length))
        except Exception as E:
            logger.warning(f'Failed to extract files: {path}: {type(E).__name__}: {E.args}')
            self.queue.put(Queue_Item(self.status_last_file.config, bg='red'))
            return
        if G.size > 0:
            X = STM_bj.get_displacement(G, self.run_config['zero_point'], self.run_config['x_conversion'])
            self.G = np.vstack([self.G, G])
            self.X = np.vstack([self.X, X])
            if self.run_config['plot_hist_G']:
                self.hist_G.add_data(G, set_ylim=self.autoscale_G.get())
                self.queue.put(Queue_Item(self.canvas_G.draw_idle))
            if self.run_config['plot_hist_GS']:
                self.hist_GS.add_data(G, X)
                if self.current_trace_GS.get() < 0: self.queue.put(Queue_Item(self.current_trace_GS.set, self.current_trace_GS.get()))
                self.queue.put(Queue_Item(self.canvas_GS.draw_idle))
            if self.run_config['plot_hist_Gt'] and hasattr(self, 'time_init'):
                self.hist_Gt.add_data(G, time - self.time_init)
                self.queue.put(Queue_Item(self.canvas_Gt.draw_idle))
            if self.run_config['plot_2DCH']:
                self.hist_2DCH.add_data(G)
                self.queue.put(Queue_Item(self.canvas_2DCH.draw_idle))
            self.queue.put(Queue_Item(self.status_traces.config, text=self.G.shape[0]))
        self.queue.put(Queue_Item(self.status_last_file.config, bg='lime'))

    def import_setting(self, data: dict = None):
        if not data:
            path = tkinter.filedialog.askopenfilename(filetypes=[('YAML', '*.yaml'), ('All Files', '*.*')])
            if not path: return
            import yaml
            with open(path, mode='r', encoding='utf-8') as f:
                data = yaml.load(f.read(), yaml.SafeLoader)['STM-bj']
        settings = {
            'Data type': self.is_raw,
            'Recursive': self.directory_recursive,
            'Length': self.extract_length,
            'G upper': self.upper,
            'G lower': self.lower,
            'X=0@G=': self.zero_point,
            'Points/nm': self.points_per_nm,
            'Direction': self.direction,
            'G min': self.G_min,
            'G max': self.G_max,
            'G #bins': self.G_bins,
            'G scale': self.G_scale,
            'X min': self.X_min,
            'X max': self.X_max,
            'X #bins': self.X_bins,
            'X scale': self.X_scale,
            't min': self.t_min,
            't max': self.t_max,
            't bin size': self.t_bin_size,
            't scale': self.t_scale,
            'hist_G': self.plot_hist_G,
            'hist_GS': self.plot_hist_GS,
            'hist_Gt': self.plot_hist_Gt,
            'hist_2DCH': self.plot_2DCH
        }
        not_valid = list()
        for setting, attribute in settings.items():
            try:
                if setting in data: attribute.set(data[setting])
            except Exception as E:
                not_valid.append(setting)
        try:
            if 'Colorbar' in data:
                self.colorbar_conf.delete('1.0', 'end')
                self.colorbar_conf.insert('0.0', data['Colorbar'])
        except Exception as E:
            not_valid.append(setting)
        if len(not_valid):
            tkinter.messagebox.showwarning('Warning', f'Invalid values:\n{", ".join(not_valid)}')

    def updatetk(self):
        while not self.queue.empty():
            try:
                item: Queue_Item = self.queue.get()
                item.run()
            except Exception as E:
                logger.warning(f'{type(E).__name__}: {E.args}')
        self.window.after(100, self.updatetk)

    def cleanup(self, catagory: Literal['partial', 'all']):
        if hasattr(self, 'time_init'): del self.time_init
        if hasattr(self, 'G'): del self.G
        if hasattr(self, 'X'): del self.X
        if hasattr(self, 'hist_G'):
            self.hist_G.fig.clear()
            del self.hist_G
        if hasattr(self, 'hist_GS'):
            self.hist_GS.fig.clear()
            del self.hist_GS
        if hasattr(self, 'hist_Gt'):
            self.hist_Gt.fig.clear()
            del self.hist_Gt
        if hasattr(self, 'hist_2DCH'):
            self.hist_2DCH.fig.clear()
            del self.hist_2DCH
        if catagory == 'partial':
            gc.collect()
            return
        else:
            if hasattr(self, 'observer'):
                self.observer.stop()
                del self.observer
            self.export_prompt.window.destroy()
            self.window.destroy()


class STM_bj_export_prompt:

    def __init__(self, root: STM_bj_GUI, **kwargs) -> None:
        self.window = tk.Toplevel()
        self.hide()
        self.window.protocol("WM_DELETE_WINDOW", self.hide)
        self.window.title('Export')
        self.window.resizable(False, False)
        self.root = root
        self.kwargs = kwargs
        # tab
        self.tabcontrol = ttk.Notebook(self.window)
        self.tabcontrol.pack(side='top')
        tab_raw = ttk.Frame(self.tabcontrol)
        tab_1D = ttk.Frame(self.tabcontrol)
        tab_2D = ttk.Frame(self.tabcontrol)
        tab_settings = ttk.Frame(self.tabcontrol)
        self.tabcontrol.add(tab_raw, text='Raw data')
        self.tabcontrol.add(tab_1D, text='1D histogram')
        self.tabcontrol.add(tab_2D, text='2D histogram')
        try:
            import yaml
            self.tabcontrol.add(tab_settings, text='Settings')
        except ImportError:
            pass
        # raw
        self.check_raw_X = tk.BooleanVar(self.window, value=True)  #disabled
        self.check_raw_G = tk.BooleanVar(self.window, value=True)
        self.check_raw_logG = tk.BooleanVar(self.window, value=True)
        tk.Checkbutton(tab_raw, variable=self.check_raw_X, text='X', state='disabled').grid(row=0, column=0)
        tk.Checkbutton(tab_raw, variable=self.check_raw_G, text='G').grid(row=0, column=1)
        tk.Checkbutton(tab_raw, variable=self.check_raw_logG, text='logG').grid(row=0, column=2)
        # 1D
        self.check_1D_G = tk.BooleanVar(self.window, value=True)  #disabled
        self.option_1D_count = tk.StringVar(self.window, value='Count')
        tk.Checkbutton(tab_1D, variable=self.check_1D_G, text='G', state='disabled').grid(row=0, column=0)
        tk.OptionMenu(tab_1D, self.option_1D_count, *['Count', 'Count/trace']).grid(row=0, column=1)
        # 2D
        self.check_2D_axis = tk.BooleanVar(self.window, value=False)
        self.option_2D_count = tk.StringVar(self.window, value='Count')
        tk.Checkbutton(tab_2D, variable=self.check_2D_axis, text='Axis').grid(row=0, column=0)
        tk.OptionMenu(tab_2D, self.option_2D_count, *['Count', 'Count/trace']).grid(row=0, column=1)
        # button
        tk.Button(self.window, text='Export', command=self.run).pack(side='top')

    def show(self):
        self.window.deiconify()
        self.window.grab_set()

    def hide(self):
        self.window.withdraw()
        self.window.grab_release()

    def run(self):
        tabname = GUI.tabcontrol.tab(GUI.tabcontrol.index('current'), 'text')
        match self.tabcontrol.index('current'):
            case 0:
                path = tkinter.filedialog.asksaveasfilename(confirmoverwrite=True, initialfile=f'{tabname}.csv', defaultextension='.csv', filetypes=[('Comma delimited', '*.csv'), ('All Files', '*.*')])
                if not path: return
                A = self.root.X.ravel()
                if self.check_raw_G.get(): A = np.vstack([A, self.root.G.ravel()])
                if self.check_raw_logG.get(): A = np.vstack([A, np.log10(np.abs(self.root.G)).ravel()])
                np.savetxt(path, A.T, delimiter=",")
            case 1:
                path = tkinter.filedialog.asksaveasfilename(confirmoverwrite=True, initialfile=f'{tabname}.csv', defaultextension='.csv', filetypes=[('Comma delimited', '*.csv'), ('All Files', '*.*')])
                if not path: return
                G = np.log10(np.abs(self.root.hist_G.x)) if self.root.run_config['G_scale'] == 'log' else self.root.hist_G.x
                count = self.root.hist_G.height_per_trace if self.option_1D_count.get() == 'Count/trace' else self.root.hist_G.height
                np.savetxt(path, np.vstack([G, count]).T, delimiter=',')
            case 2:
                path = tkinter.filedialog.asksaveasfilename(confirmoverwrite=True, initialfile=f'{tabname}.csv', defaultextension='.csv', filetypes=[('Comma delimited', '*.csv'), ('All Files', '*.*')])
                if not path: return
                count = self.root.hist_GS.height_per_trace.T if self.option_2D_count.get() == 'Count/trace' else self.root.hist_GS.height.T
                if self.check_2D_axis.get():
                    df = pd.DataFrame(count)
                    df.columns = np.log10(np.abs(self.root.hist_GS.x)) if self.root.run_config['X_scale'] == 'log' else self.root.hist_GS.x
                    df.index = np.log10(np.abs(self.root.hist_GS.y)) if self.root.run_config['G_scale'] == 'log' else self.root.hist_GS.y
                    df.to_csv(path, sep=',')
                else:
                    np.savetxt(path, count, delimiter=",")
            case 3:
                import yaml
                path = tkinter.filedialog.asksaveasfilename(confirmoverwrite=True, initialfile='config.yaml', defaultextension='.yaml', filetypes=[('YAML', '*.yaml'), ('All Files', '*.*')])
                if not path: return
                data = {
                    'Data type': self.root.is_raw.get(),
                    'Recursive': self.root.directory_recursive.get(),
                    'Length': self.root.extract_length.get(),
                    'G upper': self.root.upper.get(),
                    'G lower': self.root.lower.get(),
                    'X=0@G=': self.root.zero_point.get(),
                    'Points/nm': self.root.points_per_nm.get(),
                    'Direction': self.root.direction.get(),
                    'G min': self.root.G_min.get(),
                    'G max': self.root.G_max.get(),
                    'G #bins': self.root.G_bins.get(),
                    'G scale': self.root.G_scale.get(),
                    'X min': self.root.X_min.get(),
                    'X max': self.root.X_max.get(),
                    'X #bins': self.root.X_bins.get(),
                    'X scale': self.root.X_scale.get(),
                    't min': self.root.t_min.get(),
                    't max': self.root.t_max.get(),
                    't bin size': self.root.t_bin_size.get(),
                    't scale': self.root.t_scale.get(),
                    'hist_G': self.root.plot_hist_G.get(),
                    'hist_GS': self.root.plot_hist_GS.get(),
                    'hist_Gt': self.root.plot_hist_Gt.get(),
                    'hist_2DCH': self.root.plot_2DCH.get(),
                    'Colorbar': self.root.colorbar_conf.get('0.0', 'end')
                }
                if os.path.exists(path):
                    with open(path, mode='r', encoding='utf-8') as f:
                        old_data = yaml.load(f.read(), yaml.SafeLoader)
                        if old_data: old_data.update({'STM-bj': data})
                        else: old_data = {'STM-bj': data}
                    with open(path, mode='w', encoding='utf-8') as f:
                        data = yaml.dump(old_data, f, yaml.SafeDumper)
                else:
                    with open(path, mode='w', encoding='utf-8') as f:
                        data = yaml.dump({'STM-bj': data}, f, yaml.SafeDumper)
        self.hide()


class IVscan_GUI:

    def __init__(self, iframe: tk.Frame) -> None:
        self.window = iframe
        self.export_prompt = IVscan_export_prompt(self)
        # config frame
        self.frame_config = tk.Frame(self.window)
        self.frame_config.pack(side='top', anchor='w')
        # row 0
        self.directory_path = tk.StringVar()
        tk.Label(self.frame_config, text='Path: ').grid(row=0, column=0)
        tk.Entry(self.frame_config, textvariable=self.directory_path, width=85).grid(row=0, column=1, columnspan=5)
        tk.Button(self.frame_config, text="Files", bg='#ffe9a2', command=lambda: _set_directory(self.directory_path, json.dumps(tkinter.filedialog.askopenfilenames(), ensure_ascii=False))).grid(row=0, column=6)
        tk.Button(self.frame_config, text="Folder", bg='#ffe9a2', command=lambda: _set_directory(self.directory_path, tkinter.filedialog.askdirectory())).grid(row=0, column=7, padx=5)
        # row 1
        self.mode = tk.StringVar(self.window, value='Ebias')
        self.Ebias = tk.DoubleVar(self.window, value=0.05)
        self.I_unit = tk.DoubleVar(self.window, value=1e-6)
        self.V_unit = tk.DoubleVar(self.window, value=1)
        self.is_raw = tk.StringVar(self.window, value='raw')
        self.directory_recursive = tk.BooleanVar(self.window, value=False)
        tk.Label(self.frame_config, text='Mode: ').grid(row=1, column=0)
        tk.OptionMenu(self.frame_config, self.mode, *['Ebias', 'Ewk']).grid(row=1, column=1)
        tk.Label(self.frame_config, text='Ebias: ').grid(row=1, column=2)
        tk.Entry(self.frame_config, textvariable=self.Ebias, justify='center').grid(row=1, column=3)
        tk.Label(self.frame_config, text='Units (I, V): ').grid(row=1, column=4)
        frame_units = tk.Frame(self.frame_config)
        frame_units.grid(row=1, column=5)
        tk.Entry(frame_units, textvariable=self.I_unit, justify='center', width=10).pack(side='left')
        tk.Entry(frame_units, textvariable=self.V_unit, justify='center', width=10).pack(side='left')
        tk.OptionMenu(self.frame_config, self.is_raw, *['raw', 'cut']).grid(row=1, column=6)
        tk.Checkbutton(self.frame_config, variable=self.directory_recursive, text="Recursive").grid(row=1, column=7, sticky='w')
        #row 2
        self.num_segment = tk.IntVar(self.window, value=4)  # number of segments in one cycle
        self.points_per_file = tk.IntVar(self.window, value=1000)
        self.sampling_rate = tk.IntVar(self.window, value=40000)
        tk.Label(self.frame_config, text='Points/File: ').grid(row=2, column=0)
        tk.Entry(self.frame_config, textvariable=self.points_per_file, justify='center').grid(row=2, column=1)
        tk.Label(self.frame_config, text='#Segments: ').grid(row=2, column=2)
        tk.Entry(self.frame_config, textvariable=self.num_segment, justify='center').grid(row=2, column=3)
        tk.Label(self.frame_config, text='Sampling\nrate: ').grid(row=2, column=4)
        tk.Entry(self.frame_config, textvariable=self.sampling_rate, justify='center').grid(row=2, column=5)
        # row 3
        self.V_upper = tk.DoubleVar(self.window, value=1.45)
        self.V_lower = tk.DoubleVar(self.window, value=-1.45)
        self.length = tk.IntVar(self.window, value=1200)
        self.tolerance = tk.IntVar(self.window, value=0)
        self.offset0 = tk.IntVar(self.window, value=1200)
        self.offset1 = tk.IntVar(self.window, value=1200)
        self.extract_method = tk.StringVar(self.window, value='height')
        tk.Label(self.frame_config, text='V upper/\nlower: ').grid(row=3, column=0)
        frame_Vlimit = tk.Frame(self.frame_config)
        frame_Vlimit.grid(row=3, column=1)
        tk.Entry(frame_Vlimit, textvariable=self.V_upper, justify='center', width=10).pack(side='left')
        tk.Entry(frame_Vlimit, textvariable=self.V_lower, justify='center', width=10).pack(side='left')
        tk.Label(self.frame_config, text='Length: ').grid(row=3, column=2)
        frame_length = tk.Frame(self.frame_config)
        frame_length.grid(row=3, column=3)
        tk.Entry(frame_length, textvariable=self.length, justify='center', width=8).pack(side='left')
        tk.Label(frame_length, text='±').pack(side='left')
        tk.Entry(frame_length, textvariable=self.tolerance, justify='center', width=8).pack(side='left')
        tk.Label(self.frame_config, text='Offset: ').grid(row=3, column=4)
        frame_offset = tk.Frame(self.frame_config)
        frame_offset.grid(row=3, column=5)
        tk.Entry(frame_offset, textvariable=self.offset0, justify='center', width=10).pack(side='left')
        tk.Entry(frame_offset, textvariable=self.offset1, justify='center', width=10).pack(side='left')
        tk.Label(self.frame_config, text='Method: ').grid(row=3, column=6)
        tk.OptionMenu(self.frame_config, self.extract_method, *['height', 'gradient']).grid(row=3, column=7)
        # row 4
        self.I_limit = tk.DoubleVar(self.window, value=1e-5)
        self.is_noise_remove = tk.BooleanVar(self.window, value=True)
        self.V0 = tk.DoubleVar(self.window, value=0)
        self.dV = tk.DoubleVar(self.window, value=0.1)
        self.is_zeroing = tk.BooleanVar(self.window, value=True)
        self.zeroing_center = tk.DoubleVar(self.window, value=0)
        self.direction = tk.StringVar(self.window, value='both')
        tk.Label(self.frame_config, text='I limit: ').grid(row=4, column=0)
        tk.Entry(self.frame_config, textvariable=self.I_limit, justify='center').grid(row=4, column=1)
        tk.Checkbutton(self.frame_config, variable=self.is_noise_remove, text='I min@V=').grid(row=4, column=2)
        frame_noise = tk.Frame(self.frame_config)
        frame_noise.grid(row=4, column=3)
        tk.Entry(frame_noise, textvariable=self.V0, justify='center', width=8).pack(side='left')
        tk.Label(frame_noise, text='±').pack(side='left')
        tk.Entry(frame_noise, textvariable=self.dV, justify='center', width=8).pack(side='left')
        tk.Checkbutton(self.frame_config, variable=self.is_zeroing, text='Zeroing').grid(row=4, column=4, sticky='w')
        tk.Entry(self.frame_config, textvariable=self.zeroing_center, justify='center').grid(row=4, column=5)
        tk.Label(self.frame_config, text='Direction: ').grid(row=4, column=6)
        tk.OptionMenu(self.frame_config, self.direction, *['both', '-→+', '+→-']).grid(row=4, column=7)
        # row 5
        self.V_min = tk.DoubleVar(self.window, value=-1.5)
        self.V_max = tk.DoubleVar(self.window, value=1.5)
        self.V_bins = tk.IntVar(self.window, value=300)
        self.V_scale = tk.StringVar(self.window, value='linear')
        tk.Label(self.frame_config, text='V min: ').grid(row=5, column=0)
        tk.Entry(self.frame_config, textvariable=self.V_min, justify='center').grid(row=5, column=1)
        tk.Label(self.frame_config, text='V max: ').grid(row=5, column=2)
        tk.Entry(self.frame_config, textvariable=self.V_max, justify='center').grid(row=5, column=3)
        tk.Label(self.frame_config, text='V #bins: ').grid(row=5, column=4)
        tk.Entry(self.frame_config, textvariable=self.V_bins, justify='center').grid(row=5, column=5)
        tk.Label(self.frame_config, text='V scale: ').grid(row=5, column=6)
        tk.OptionMenu(self.frame_config, self.V_scale, *['log', 'linear']).grid(row=5, column=7)
        # row 6
        self.G_min = tk.DoubleVar(self.window, value=1e-5)
        self.G_max = tk.DoubleVar(self.window, value=1e-1)
        self.G_bins = tk.IntVar(self.window, value=400)
        self.G_scale = tk.StringVar(self.window, value='log')
        tk.Label(self.frame_config, text='G min: ').grid(row=6, column=0)
        tk.Entry(self.frame_config, textvariable=self.G_min, justify='center').grid(row=6, column=1)
        tk.Label(self.frame_config, text='G max: ').grid(row=6, column=2)
        tk.Entry(self.frame_config, textvariable=self.G_max, justify='center').grid(row=6, column=3)
        tk.Label(self.frame_config, text='G #bins: ').grid(row=6, column=4)
        tk.Entry(self.frame_config, textvariable=self.G_bins, justify='center').grid(row=6, column=5)
        tk.Label(self.frame_config, text='G scale: ').grid(row=6, column=6)
        tk.OptionMenu(self.frame_config, self.G_scale, *['log', 'linear']).grid(row=6, column=7)
        # row 7
        self.I_min = tk.DoubleVar(self.window, value=1e-11)
        self.I_max = tk.DoubleVar(self.window, value=1e-5)
        self.I_bins = tk.IntVar(self.window, value=600)
        self.I_scale = tk.StringVar(self.window, value='log')
        tk.Label(self.frame_config, text='I min: ').grid(row=7, column=0)
        tk.Entry(self.frame_config, textvariable=self.I_min, justify='center').grid(row=7, column=1)
        tk.Label(self.frame_config, text='I max: ').grid(row=7, column=2)
        tk.Entry(self.frame_config, textvariable=self.I_max, justify='center').grid(row=7, column=3)
        tk.Label(self.frame_config, text='I #bins: ').grid(row=7, column=4)
        tk.Entry(self.frame_config, textvariable=self.I_bins, justify='center').grid(row=7, column=5)
        tk.Label(self.frame_config, text='I scale: ').grid(row=7, column=6)
        tk.OptionMenu(self.frame_config, self.I_scale, *['log', 'linear']).grid(row=7, column=7)
        # row 8
        self.t_min = tk.DoubleVar(self.window, value=0)
        self.t_max = tk.DoubleVar(self.window, value=0.18)
        self.t_bins = tk.IntVar(self.window, value=1800)
        self.t_scale = tk.StringVar(self.window, value='linear')
        tk.Label(self.frame_config, text='t min: ').grid(row=8, column=0)
        tk.Entry(self.frame_config, textvariable=self.t_min, justify='center').grid(row=8, column=1)
        tk.Label(self.frame_config, text='t max: ').grid(row=8, column=2)
        tk.Entry(self.frame_config, textvariable=self.t_max, justify='center').grid(row=8, column=3)
        tk.Label(self.frame_config, text='t #bins: ').grid(row=8, column=4)
        tk.Entry(self.frame_config, textvariable=self.t_bins, justify='center').grid(row=8, column=5)
        tk.Label(self.frame_config, text='t scale: ').grid(row=8, column=6)
        tk.OptionMenu(self.frame_config, self.t_scale, *['log', 'linear']).grid(row=8, column=7)
        # row 9
        tk.Label(self.frame_config, text='Colorbar: ').grid(row=9, column=0)
        self.colorbar_conf = tk.Text(self.frame_config, height=3, wrap='none', undo=True, maxundo=-1)
        self.colorbar_conf.bind('<<Modified>>', self.colorbar_apply)
        self.colorbar_conf.grid(row=9, column=1, columnspan=5, sticky='w')
        self.colorbar_conf.insert('0.0', '{"red":  [[0,1,1],[0.05,0,0],[0.1,0,0],[0.15,1,1],[0.3,1,1],[1,1,1]],\n "green":[[0,1,1],[0.05,0,0],[0.1,1,1],[0.15,1,1],[0.3,0,0],[1,0,0]],\n "blue": [[0,1,1],[0.05,1,1],[0.1,0,0],[0.15,0,0],[0.3,0,0],[1,1,1]]}')
        self.run_button = tk.Button(self.frame_config, text='Run', bg='lime', command=self.run)
        self.run_button.grid(row=9, column=6, padx=10)
        self.is_run = False
        try:
            import yaml
            tk.Button(self.frame_config, text='Import', command=self.import_setting).grid(row=9, column=7)
        except ImportError:
            logger.warning('Module PyYAML was not found. Import/export settings can not be used.')
        tk.Button(self.frame_config, text='Export', command=self.export_prompt.show).grid(row=9, column=8)
        # is_plot frame
        self.frame_is_plot = tk.Frame(self.window)
        self.frame_is_plot.pack(side='top', anchor='w')
        self.plot_hist_GV = tk.BooleanVar(self.window, value=True)
        self.plot_hist_IV = tk.BooleanVar(self.window, value=True)
        self.plot_hist_IVt = tk.BooleanVar(self.window, value=False)
        self.plot_hist_GVt = tk.BooleanVar(self.window, value=False)
        tk.Label(self.frame_is_plot, text='Plot: ').pack(side='left')
        tk.Checkbutton(self.frame_is_plot, text='Histogram IV', variable=self.plot_hist_IV).pack(side='left')
        tk.Checkbutton(self.frame_is_plot, text='Histogram GV', variable=self.plot_hist_GV).pack(side='left')
        tk.Checkbutton(self.frame_is_plot, text='Histogram IVt', variable=self.plot_hist_IVt).pack(side='left')
        tk.Checkbutton(self.frame_is_plot, text='Histogram GVt', variable=self.plot_hist_GVt).pack(side='left')
        # figure frame
        self.frame_figure = tk.Frame(self.window)
        self.frame_figure.pack(side='top', anchor='w')
        self.frame_figure.columnconfigure([0, 5, 10, 15], weight=1)
        self.frame_figure.rowconfigure([0], weight=1)
        # status frame
        self.frame_status = tk.Frame(self.window)
        self.frame_status.pack(side='bottom', anchor='w')
        tk.Label(self.frame_status, text='#Cycles: ').pack(side='left', anchor='w')
        self.status_cycles = tk.Label(self.frame_status, text=0)
        self.status_cycles.pack(side='left', anchor='w')
        tk.Label(self.frame_status, text='#Segments: ', padx=20).pack(side='left', anchor='w')
        self.status_traces = tk.Label(self.frame_status, text=0)
        self.status_traces.pack(side='left', anchor='w')
        tk.Label(self.frame_status, text='File: ', padx=20).pack(side='left')
        self.status_last_file = tk.Label(self.frame_status, text='Waiting', anchor='w')
        self.status_last_file.pack(side='left', anchor='w')
        self.queue = Queue()
        self.updatetk()
        if ini_config:
            if 'IVscan' in ini_config:
                self.import_setting(ini_config['IVscan'])

    def colorbar_apply(self, *args):
        try:
            colorbar_conf = self.colorbar_conf.get('0.0', 'end')
            if colorbar_conf != "\n":
                cmap = json.loads(colorbar_conf)
                if self.run_config['plot_hist_GV']:
                    self.hist_GV.set_cmap(cmap=cmap)
                    self.canvas_GV.draw_idle()
                if self.run_config['plot_hist_IV']:
                    self.hist_IV.set_cmap(cmap=cmap)
                    self.canvas_IV.draw_idle()
                if self.run_config['plot_hist_IVt']:
                    self.hist_IVt.set_cmap(cmap=cmap)
                    self.canvas_IVt.draw_idle()
                if self.run_config['plot_hist_GVt']:
                    self.hist_GVt.set_cmap(cmap=cmap)
                    self.canvas_GVt.draw_idle()
        except Exception as E:
            return
        finally:
            self.colorbar_conf.edit_modified(False)

    def run(self):
        match self.is_run:
            case False:
                self.cleanup('partial')
                full_length = self.num_segment.get() * self.length.get() + self.offset0.get() + self.offset1.get()
                self.run_config = {
                    "mode": self.mode.get(),
                    "Ebias": self.Ebias.get(),
                    "units": (self.I_unit.get(), self.V_unit.get()),
                    'recursive': self.directory_recursive.get(),
                    'data_type': self.is_raw.get(),
                    'num_segment': self.num_segment.get(),
                    'num_files': full_length // self.points_per_file.get() + 3,
                    "V_upper": self.V_upper.get(),
                    "V_lower": self.V_lower.get(),
                    "length_segment": self.length.get(),
                    "tolerance": self.tolerance.get(),
                    "offset": (self.offset0.get(), self.offset1.get()),
                    'is_noise_remove': self.is_noise_remove.get(),
                    'V0': self.V0.get(),
                    'dV': self.dV.get(),
                    "I_limit": self.I_limit.get(),
                    'is_zeroing': self.is_zeroing.get(),
                    'zeroing_center': self.zeroing_center.get(),
                    'direction': self.direction.get(),
                    'extract_method': self.extract_method.get(),
                    'V_scale': self.V_scale.get(),
                    'G_scale': self.G_scale.get(),
                    'I_scale': self.I_scale.get(),
                    'plot_hist_GV': self.plot_hist_GV.get(),
                    'plot_hist_IV': self.plot_hist_IV.get(),
                    'plot_hist_IVt': self.plot_hist_IVt.get(),
                    'plot_hist_GVt': self.plot_hist_GVt.get()
                }
                path = self.directory_path.get()
                if not os.path.isdir(path):
                    try:
                        path = json.loads(path)
                    except Exception as E:
                        tkinter.messagebox.showerror('Error', 'Invalid directory')
                        return
                for item in self.frame_figure.winfo_children():
                    item.destroy()
                gc.collect()
                self.I = np.empty((0, self.length.get()))
                self.V = np.empty((0, self.length.get()))
                self.G = np.empty((0, self.length.get()))
                self.I_full = np.empty((0, full_length))
                self.V_full = np.empty((0, full_length))
                self.G_full = np.empty((0, full_length))
                # hist IV
                if self.run_config['plot_hist_IV']:
                    self.hist_IV = IVscan.Hist_IV([self.V_min.get(), self.V_max.get()], [self.I_min.get(), self.I_max.get()], self.V_bins.get(), self.I_bins.get(), self.V_scale.get(), self.I_scale.get(), 'wk' if self.mode.get() == 'Ewk' else 'bias')
                    self.canvas_IV = FigureCanvasTkAgg(self.hist_IV.fig, self.frame_figure)
                    self.canvas_IV.get_tk_widget().grid(row=0, column=0, columnspan=5, pady=10)
                    self.navtool_IV = NavigationToolbar2Tk(self.canvas_IV, self.frame_figure, pack_toolbar=False)
                    self.navtool_IV.grid(row=1, column=0, columnspan=4, sticky='w')
                    self.canvas_IV.draw_idle()
                #hist GV
                if self.run_config['plot_hist_GV']:
                    self.hist_GV = IVscan.Hist_GV([self.V_min.get(), self.V_max.get()], [self.G_min.get(), self.G_max.get()], self.V_bins.get(), self.G_bins.get(), self.V_scale.get(), self.G_scale.get(), 'wk' if self.mode.get() == 'Ewk' else 'bias')
                    self.canvas_GV = FigureCanvasTkAgg(self.hist_GV.fig, self.frame_figure)
                    self.canvas_GV.get_tk_widget().grid(row=0, column=5, columnspan=5, pady=10)
                    self.navtool_GV = NavigationToolbar2Tk(self.canvas_GV, self.frame_figure, pack_toolbar=False)
                    self.navtool_GV.grid(row=1, column=5, columnspan=4, sticky='w')
                    self.canvas_GV.draw_idle()
                # hist IVt
                if self.run_config['plot_hist_IVt']:
                    self.hist_IVt = IVscan.Hist_IVt([self.t_min.get(), self.t_max.get()], [self.I_min.get(), self.I_max.get()], [self.V_min.get(), self.V_max.get()], self.t_bins.get(), self.I_bins.get(), self.t_scale.get(), self.I_scale.get(), self.V_scale.get(), self.sampling_rate.get(),
                                                    'wk' if self.mode.get() == 'Ewk' else 'bias')
                    self.canvas_IVt = FigureCanvasTkAgg(self.hist_IVt.fig, self.frame_figure)
                    self.canvas_IVt.get_tk_widget().grid(row=0, column=10, columnspan=5, pady=10)
                    self.navtool_IVt = NavigationToolbar2Tk(self.canvas_IVt, self.frame_figure, pack_toolbar=False)
                    self.navtool_IVt.grid(row=1, column=10, columnspan=4, sticky='w')
                    self.canvas_IVt.draw_idle()
                # hist GVt
                if self.run_config['plot_hist_GVt']:
                    self.hist_GVt = IVscan.Hist_GVt([self.t_min.get(), self.t_max.get()], [self.G_min.get(), self.G_max.get()], [self.V_min.get(), self.V_max.get()], self.t_bins.get(), self.G_bins.get(), self.t_scale.get(), self.G_scale.get(), self.V_scale.get(), self.sampling_rate.get(),
                                                    'wk' if self.mode.get() == 'Ewk' else 'bias')
                    self.canvas_GVt = FigureCanvasTkAgg(self.hist_GVt.fig, self.frame_figure)
                    self.canvas_GVt.get_tk_widget().grid(row=0, column=15, columnspan=5, pady=10)
                    self.navtool_GVt = NavigationToolbar2Tk(self.canvas_GVt, self.frame_figure, pack_toolbar=False)
                    self.navtool_GVt.grid(row=1, column=15, columnspan=4, sticky='w')
                    self.canvas_GVt.draw_idle()
                # # Trace IV/GV
                if self.run_config['plot_hist_IV'] or self.run_config['plot_hist_GV']:

                    def show_trace_IV_GV(action: Literal['next', 'prev', 'show', 'clear']):
                        trace = self.current_trace_IV_GV.get()
                        if action == 'show':
                            if self.plot_trace_IV_GV.get():
                                if -self.V.shape[0] <= trace < self.V.shape[0]:
                                    if self.run_config['plot_hist_IV']:
                                        if self.current_lines_IV: self.current_lines_IV.set_data(self.V[trace], np.abs(self.I[trace]))
                                        else: self.current_lines_IV = self.hist_IV.ax.plot(self.V[trace], np.abs(self.I[trace]), color='k')[0]
                                        self.canvas_IV.draw_idle()
                                    if self.run_config['plot_hist_GV']:
                                        if self.current_lines_GV: self.current_lines_GV.set_data(self.V[trace], np.abs(self.G[trace]))
                                        else: self.current_lines_GV = self.hist_GV.ax.plot(self.V[trace], np.abs(self.G[trace]), color='k')[0]
                                        self.canvas_GV.draw_idle()
                        elif action == 'next':
                            if -self.V.shape[0] <= trace + 1 < self.V.shape[0]:
                                self.current_trace_IV_GV.set(trace + 1)
                        elif action == 'prev':
                            if -self.V.shape[0] <= trace - 1 < self.V.shape[0]:
                                self.current_trace_IV_GV.set(trace - 1)
                        elif action == 'clear':
                            if self.current_lines_IV:
                                self.current_lines_IV.remove()
                                self.current_lines_IV = None
                                self.canvas_IV.draw_idle()
                            if self.current_lines_GV:
                                self.current_lines_GV.remove()
                                self.current_lines_GV = None
                                self.canvas_GV.draw_idle()

                    self.plot_trace_IV_GV = tk.BooleanVar(self.window, value=False)
                    self.current_trace_IV_GV = tk.IntVar(self.window, value=-1)
                    self.current_lines_IV = None
                    self.current_lines_GV = None
                    frame_trace_IV = tk.Frame(self.frame_figure)
                    frame_trace_IV.grid(row=2, column=0, sticky='w') if self.run_config['plot_hist_IV'] else frame_trace_IV.grid(row=2, column=5, sticky='w')
                    tk.Checkbutton(frame_trace_IV, text="Segment: ", variable=self.plot_trace_IV_GV).pack(side='left', anchor='w')
                    tk.Button(frame_trace_IV, text='<', command=lambda: show_trace_IV_GV('prev')).pack(side='left', anchor='w')
                    current_trace_IV_entry = tk.Entry(frame_trace_IV, textvariable=self.current_trace_IV_GV, justify='center', width=8)
                    current_trace_IV_entry.pack(side='left', anchor='w')
                    tk.Button(frame_trace_IV, text='>', command=lambda: show_trace_IV_GV('next')).pack(side='left', anchor='w')
                    current_trace_IV_entry.bind("<Down>", lambda *args: show_trace_IV_GV('prev'))
                    current_trace_IV_entry.bind("<Up>", lambda *args: show_trace_IV_GV('next'))
                    self.plot_trace_IV_GV.trace_add('write', lambda *args: show_trace_IV_GV('show' if self.plot_trace_IV_GV.get() else 'clear'))
                    self.current_trace_IV_GV.trace_add('write', lambda *args: show_trace_IV_GV('show'))
                # Trace IVt/GVt
                if self.run_config['plot_hist_IVt'] or self.run_config['plot_hist_GVt']:

                    def show_trace_It(action: Literal['next', 'prev', 'show', 'clear']):
                        trace = self.current_trace_It_Gt.get()
                        if action == 'show':
                            if self.plot_trace_It_Gt.get():
                                if -self.V_full.shape[0] <= trace < self.V_full.shape[0]:
                                    if self.run_config['plot_hist_IVt']:
                                        if self.current_lines_It: self.current_lines_It.set_data(self.time_array, np.abs(self.I_full[trace]))
                                        else: self.current_lines_It = self.hist_IVt.ax.plot(self.time_array, np.abs(self.I_full[trace]), color='k')[0]
                                        self.canvas_IVt.draw_idle()
                                    if self.run_config['plot_hist_GVt']:
                                        if self.current_lines_Gt: self.current_lines_Gt.set_data(self.time_array, np.abs(self.G_full[trace]))
                                        else: self.current_lines_Gt = self.hist_GVt.ax.plot(self.time_array, np.abs(self.G_full[trace]), color='k')[0]
                                        self.canvas_GVt.draw_idle()
                        elif action == 'next':
                            if -self.V_full.shape[0] <= trace + 1 < self.V_full.shape[0]:
                                self.current_trace_It_Gt.set(trace + 1)
                        elif action == 'prev':
                            if -self.V_full.shape[0] <= trace - 1 < self.V_full.shape[0]:
                                self.current_trace_It_Gt.set(trace - 1)
                        elif action == 'clear':
                            if self.current_lines_It:
                                self.current_lines_It.remove()
                                self.current_lines_It = None
                                self.canvas_IVt.draw_idle()
                            if self.current_lines_Gt:
                                self.current_lines_Gt.remove()
                                self.current_lines_Gt = None
                                self.canvas_GVt.draw_idle()

                    self.plot_trace_It_Gt = tk.BooleanVar(self.window, value=False)
                    self.current_trace_It_Gt = tk.IntVar(self.window, value=-1)
                    self.current_lines_It = None
                    self.current_lines_Gt = None
                    if self.run_config['plot_hist_IVt']: self.time_array = np.arange(full_length) / self.hist_IVt.x_conversion
                    else: self.time_array = np.arange(full_length) / self.hist_GVt.x_conversion
                    frame_trace_It = tk.Frame(self.frame_figure)
                    frame_trace_It.grid(row=2, column=10, sticky='w') if self.run_config['plot_hist_IVt'] else frame_trace_It.grid(row=2, column=15, sticky='w')
                    tk.Checkbutton(frame_trace_It, text="Fullcycle: ", variable=self.plot_trace_It_Gt).pack(side='left', anchor='w')
                    tk.Button(frame_trace_It, text='<', command=lambda: show_trace_It('prev')).pack(side='left', anchor='w')
                    current_trace_It_entry = tk.Entry(frame_trace_It, textvariable=self.current_trace_It_Gt, justify='center', width=8)
                    current_trace_It_entry.pack(side='left', anchor='w')
                    tk.Button(frame_trace_It, text='>', command=lambda: show_trace_It('next')).pack(side='left', anchor='w')
                    current_trace_It_entry.bind("<Down>", lambda *args: show_trace_It('prev'))
                    current_trace_It_entry.bind("<Up>", lambda *args: show_trace_It('next'))
                    self.plot_trace_It_Gt.trace_add('write', lambda *args: show_trace_It('show' if self.plot_trace_It_Gt.get() else 'clear'))
                    self.current_trace_It_Gt.trace_add('write', lambda *args: show_trace_It('show'))
                self.colorbar_apply()
                self.status_cycles.config(text=0)
                self.status_traces.config(text=0)
                self.window.update_idletasks()
                if self.run_config['data_type'] == 'raw': self.pending = list()
                self._lock = threading.RLock()
                threading.Thread(target=self.add_data, args=(path, )).start()
                if isinstance(path, list): return
                try:
                    from watchdog.observers import Observer
                    self.observer = Observer()
                    self.observer.schedule(self.FileHandler(self), path=path, recursive=self.run_config['recursive'])
                    self.observer.start()
                    logger.info(f'Start observer: {path}')
                    atexit.register(self.observer.stop)
                    self.run_button.config(text='Stop', bg='red')
                    self.is_run = True
                except ImportError:
                    logger.warning('Module watchdog was not found. Data can not be updated in realtime.')
            case True:
                self.run_button.config(text='Run', bg='lime')
                self.is_run = False
                threading.Thread(target=self.observer.stop).start()
                logger.info(f'Stop observer')
                gc.collect()

    try:
        from watchdog.events import FileSystemEventHandler

        class FileHandler(FileSystemEventHandler):

            def __init__(self, GUI) -> None:
                self.GUI = GUI

            def on_created(self, event):
                from watchdog.events import FileCreatedEvent
                if isinstance(event, FileCreatedEvent):
                    if (event.src_path.endswith('.txt')):
                        try:
                            if os.path.getsize(event.src_path) == 0: time.sleep(0.1)
                            self.GUI.add_data(event.src_path)
                        except Exception as E:
                            logger.warning(f'Add data failed: {event.src_path}: {type(E).__name__}: {E.args}')
    except ImportError:
        pass

    def add_data(self, path: str | list):
        if isinstance(path, str):
            self.queue.put(Queue_Item(self.status_last_file.config, text=path, bg='yellow'))
            if os.path.isdir(path):
                if not os.listdir(path):  # empty directory
                    self.queue.put(Queue_Item(self.status_last_file.config, bg='lime'))
                    return
        elif isinstance(path, list):
            self.queue.put(Queue_Item(self.status_last_file.config, text=f"{len(path)} files" if len(path) > 1 else path[0], bg='yellow'))
        try:
            logger.debug(f'Add data: {path}')
            match self.run_config['data_type']:
                case 'raw':
                    with self._lock:
                        self.pending.append(IVscan.load_data(path, max_workers=GUI.CPU_threads.get())[::-1])
                        if len(self.pending) > self.run_config['num_files']: del self.pending[:-self.run_config['num_files']]
                        IV_raw = np.concatenate(self.pending, axis=1)
                    I_full, V_full = IVscan.extract_data(IV_raw,
                                                         upper=self.run_config['V_upper'],
                                                         lower=self.run_config['V_lower'],
                                                         length_segment=self.run_config['length_segment'],
                                                         num_segment=self.run_config['num_segment'],
                                                         offset=self.run_config['offset'],
                                                         units=self.run_config['units'],
                                                         mode=self.run_config['extract_method'],
                                                         tolerance=self.run_config['tolerance'])
                    if I_full.size == 0:
                        self.queue.put(Queue_Item(self.status_last_file.config, bg='lime'))
                        return
                    else:
                        ind = np.where(V_full[-1, -1] == self.pending[-1][1])[0]
                        if ind.size: self.pending = [self.pending[-1][:, ind[-1]:]]
                        else: self.pending = []
                        if self.run_config['plot_hist_GV'] or self.run_config['plot_hist_IV']:
                            I, V = IVscan.extract_data(IV_raw,
                                                       upper=self.run_config['V_upper'],
                                                       lower=self.run_config['V_lower'],
                                                       length_segment=self.run_config['length_segment'],
                                                       num_segment=1,
                                                       offset=[0, 0],
                                                       units=self.run_config['units'],
                                                       mode=self.run_config['extract_method'],
                                                       tolerance=self.run_config['tolerance'])
                case 'cut':
                    df = IVscan.load_data_with_metadata(path, **self.run_config, max_workers=GUI.CPU_threads.get())['data']
                    length = df.apply(lambda x: x.shape[-1])
                    max_length = max(*length, self.run_config['length_segment'])
                    df[length < max_length] = df[length < max_length].apply(lambda x: np.pad(x, ((0, 0), (0, max_length - x.shape[-1])), 'constant', constant_values=0))
                    V, I = np.stack(df).swapaxes(0, 1)
                    self.I = np.empty((0, max_length))
                    self.V = np.empty((0, max_length))
                    self.G = np.empty((0, max_length))
        except Exception as E:
            logger.warning(f'Failed to extract files: {path}: {type(E).__name__}: {E.args}')
            self.queue.put(Queue_Item(self.status_last_file.config, bg='red'))
            return
        else:
            if self.run_config['plot_hist_GV'] or self.run_config['plot_hist_IV']:
                I, V = IVscan.noise_remove(I, V, I_limit=self.run_config['I_limit'])
                if self.run_config['is_noise_remove']: I, V = IVscan.noise_remove(I, V, V0=self.run_config['V0'], dV=self.run_config['dV'])
                if self.run_config['is_zeroing']: I, V = IVscan.zeroing(I, V, self.run_config['zeroing_center'])
                if I.size == 0:
                    self.queue.put(Queue_Item(self.status_last_file.config, bg='lime'))
                    return
                if self.run_config['direction'] == '-→+': I, V = IVscan.split_scan_direction(I, V)[0]
                elif self.run_config['direction'] == '+→-': I, V = IVscan.split_scan_direction(I, V)[1]
                G = IVscan.conductance(I, V if self.run_config['mode'] == 'Ebias' else self.run_config['Ebias'])
                self.I = np.vstack([self.I, I])
                self.V = np.vstack([self.V, V])
                self.G = np.vstack([self.G, G])
                if self.run_config['plot_hist_IV']:
                    self.hist_IV.add_data(I, V)
                    self.queue.put(Queue_Item(self.canvas_IV.draw_idle))
                if self.run_config['plot_hist_GV']:
                    self.hist_GV.add_data(V=V, G=G)
                    self.queue.put(Queue_Item(self.canvas_GV.draw_idle))
                if self.current_trace_IV_GV.get() < 0: self.queue.put(Queue_Item(self.current_trace_IV_GV.set, self.current_trace_IV_GV.get()))
                self.queue.put(Queue_Item(self.status_traces.config, text=self.I.shape[0]))
            if self.run_config['plot_hist_IVt'] or self.run_config['plot_hist_GVt']:
                G_full = IVscan.conductance(I_full, V_full if self.run_config['mode'] == 'Ebias' else self.run_config['Ebias'])
                self.I_full = np.vstack([self.I_full, I_full])
                self.V_full = np.vstack([self.V_full, V_full])
                self.G_full = np.vstack([self.G_full, G_full])
                if self.run_config['plot_hist_IVt']:
                    self.hist_IVt.add_data(I_full, V_full, self.time_array)
                    self.queue.put(Queue_Item(self.canvas_IVt.draw_idle))
                if self.run_config['plot_hist_GVt']:
                    self.hist_GVt.add_data(V=V_full, G=G_full, t=self.time_array)
                    self.queue.put(Queue_Item(self.canvas_GVt.draw_idle))
                if self.current_trace_It_Gt.get() < 0: self.queue.put(Queue_Item(self.current_trace_It_Gt.set, self.current_trace_It_Gt.get()))
                self.queue.put(Queue_Item(self.status_cycles.config, text=self.I_full.shape[0]))
            self.queue.put(Queue_Item(self.status_last_file.config, bg='lime'))

    def import_setting(self, data: dict = None):
        if not data:
            path = tkinter.filedialog.askopenfilename(filetypes=[('YAML', '*.yaml'), ('All Files', '*.*')])
            if not path: return
            import yaml
            with open(path, mode='r', encoding='utf-8') as f:
                data = yaml.load(f.read(), yaml.SafeLoader)['IVscan']
        settings = {
            'Data type': self.is_raw,
            'Recursive': self.directory_recursive,
            'Mode': self.mode,
            'Ebias': self.Ebias,
            'I unit': self.I_unit,
            'V unit': self.V_unit,
            '#Segments': self.num_segment,
            'points_per_file': self.points_per_file,
            'Sampling rate': self.sampling_rate,
            'V upper': self.V_upper,
            'V lower': self.V_lower,
            'Length': self.length,
            'Tolerance': self.tolerance,
            'Offset0': self.offset0,
            'Offset1': self.offset1,
            'I limit': self.I_limit,
            'Noise remove': self.is_noise_remove,
            'V0': self.V0,
            'dV': self.dV,
            'Zeroing': self.is_zeroing,
            'Zeroing center': self.zeroing_center,
            'Direction': self.direction,
            'Extract_method': self.extract_method,
            'V min': self.V_min,
            'V max': self.V_max,
            'V #bins': self.V_bins,
            'V scale': self.V_scale,
            'G min': self.G_min,
            'G max': self.G_max,
            'G #bins': self.G_bins,
            'G scale': self.G_scale,
            'I max': self.I_min,
            'I min': self.I_max,
            'I #bins': self.I_bins,
            'I scale': self.I_scale,
            't max': self.t_min,
            't min': self.t_max,
            't #bins': self.t_bins,
            't scale': self.t_scale,
            'hist_GV': self.plot_hist_GV,
            'hist_IV': self.plot_hist_IV,
            'hist_IVt': self.plot_hist_IVt,
            'hist_GVt': self.plot_hist_GVt
        }
        not_valid = list()
        for setting, attribute in settings.items():
            try:
                if setting in data: attribute.set(data[setting])
            except Exception as E:
                not_valid.append(setting)
        try:
            if 'Colorbar' in data:
                self.colorbar_conf.delete('1.0', 'end')
                self.colorbar_conf.insert('0.0', data['Colorbar'])
        except Exception as E:
            not_valid.append(setting)
        if len(not_valid):
            tkinter.messagebox.showwarning('Warning', f'Invalid values:\n{", ".join(not_valid)}')

    def updatetk(self):
        while not self.queue.empty():
            try:
                item: Queue_Item = self.queue.get()
                item.run()
            except Exception as E:
                logger.warning(f'{type(E).__name__}: {E.args}')
        self.window.after(100, self.updatetk)

    def cleanup(self, catagory: Literal['partial', 'all']):
        if hasattr(self, 'I'): del self.I
        if hasattr(self, 'V'): del self.V
        if hasattr(self, 'G'): del self.G
        if hasattr(self, 'I_full'): del self.I_full
        if hasattr(self, 'V_full'): del self.V_full
        if hasattr(self, 'G_full'): del self.G_full
        if hasattr(self, 'pending'): del self.pending
        if hasattr(self, 'time_array'): del self.time_array
        if hasattr(self, 'hist_IV'):
            self.hist_IV.fig.clear()
            del self.hist_IV
        if hasattr(self, 'hist_GV'):
            self.hist_GV.fig.clear()
            del self.hist_GV
        if hasattr(self, 'hist_IVt'):
            self.hist_IVt.fig.clear()
            del self.hist_IVt
        if hasattr(self, 'hist_GVt'):
            self.hist_GVt.fig.clear()
            del self.hist_GVt
        if catagory == 'partial':
            gc.collect()
            return
        else:
            if hasattr(self, 'observer'):
                self.observer.stop()
                del self.observer
            self.export_prompt.window.destroy()
            self.window.destroy()


class IVscan_export_prompt:

    def __init__(self, root: IVscan_GUI, **kwargs) -> None:
        self.window = tk.Toplevel()
        self.hide()
        self.window.protocol("WM_DELETE_WINDOW", self.hide)
        self.window.title('Export')
        self.window.resizable(False, False)
        self.root = root
        self.kwargs = kwargs
        # tab
        self.tabcontrol = ttk.Notebook(self.window)
        self.tabcontrol.pack(side='top')
        tab_raw = ttk.Frame(self.tabcontrol)
        tab_GV = ttk.Frame(self.tabcontrol)
        tab_IV = ttk.Frame(self.tabcontrol)
        tab_settings = ttk.Frame(self.tabcontrol)
        self.tabcontrol.add(tab_raw, text='Raw data')
        self.tabcontrol.add(tab_GV, text='GV histogram')
        self.tabcontrol.add(tab_IV, text='IV histogram')
        try:
            import yaml
            self.tabcontrol.add(tab_settings, text='Settings')
        except ImportError:
            pass
        # raw
        self.check_raw_V = tk.BooleanVar(self.window, value=True)  #disabled
        self.check_raw_G = tk.BooleanVar(self.window, value=False)
        self.check_raw_logG = tk.BooleanVar(self.window, value=True)
        self.check_raw_I = tk.BooleanVar(self.window, value=False)
        self.check_raw_absI = tk.BooleanVar(self.window, value=False)
        self.check_raw_logI = tk.BooleanVar(self.window, value=True)
        tk.Checkbutton(tab_raw, variable=self.check_raw_V, text='V', state='disabled').grid(row=0, column=1)
        tk.Checkbutton(tab_raw, variable=self.check_raw_G, text='G').grid(row=0, column=2)
        tk.Checkbutton(tab_raw, variable=self.check_raw_logG, text='logG').grid(row=0, column=3)
        tk.Checkbutton(tab_raw, variable=self.check_raw_I, text='I').grid(row=1, column=1)
        tk.Checkbutton(tab_raw, variable=self.check_raw_absI, text='| I |').grid(row=1, column=2)
        tk.Checkbutton(tab_raw, variable=self.check_raw_logI, text='logI').grid(row=1, column=3)
        # GV
        self.check_GV_axis = tk.BooleanVar(self.window, value=False)
        self.option_GV_count = tk.StringVar(self.window, value='Count')
        tk.Checkbutton(tab_GV, variable=self.check_GV_axis, text='Axis').grid(row=0, column=0)
        tk.OptionMenu(tab_GV, self.option_GV_count, *['Count', 'Count/trace']).grid(row=0, column=1)
        # IV
        self.check_IV_axis = tk.BooleanVar(self.window, value=False)
        self.option_IV_count = tk.StringVar(self.window, value='Count')
        tk.Checkbutton(tab_IV, variable=self.check_IV_axis, text='Axis').grid(row=0, column=0)
        tk.OptionMenu(tab_IV, self.option_IV_count, *['Count', 'Count/trace']).grid(row=0, column=1)
        # button
        tk.Button(self.window, text='Export', command=self.run).pack(side='top')

    def show(self):
        self.window.deiconify()
        self.window.grab_set()

    def hide(self):
        self.window.withdraw()
        self.window.grab_release()

    def run(self):
        tabname = GUI.tabcontrol.tab(GUI.tabcontrol.index('current'), 'text')
        match self.tabcontrol.index('current'):
            case 0:
                path = tkinter.filedialog.asksaveasfilename(confirmoverwrite=True, initialfile=f'{tabname}.csv', defaultextension='.csv', filetypes=[('Comma delimited', '*.csv'), ('All Files', '*.*')])
                if not path: return
                A = self.root.V.ravel()
                G = IVscan.conductance(self.root.I, self.root.V).ravel()
                if self.check_raw_G.get(): A = np.vstack([A, G])
                if self.check_raw_logG.get(): A = np.vstack([A, np.log10(np.abs(G))])
                if self.check_raw_I.get(): A = np.vstack([A, self.root.I.ravel()])
                if self.check_raw_absI.get(): A = np.vstack([A, np.abs(self.root.I.ravel())])
                if self.check_raw_logI.get(): A = np.vstack([A, np.log10(np.abs(self.root.I.ravel()))])
                np.savetxt(path, A.T, delimiter=",")
            case 1:
                path = tkinter.filedialog.asksaveasfilename(confirmoverwrite=True, initialfile=f'{tabname}.csv', defaultextension='.csv', filetypes=[('Comma delimited', '*.csv'), ('All Files', '*.*')])
                if not path: return
                count = self.root.hist_GV.height_per_trace.T if self.option_GV_count.get() == 'Count/trace' else self.root.hist_GV.height.T
                if self.check_GV_axis.get():
                    df = pd.DataFrame(count)
                    df.columns = np.log10(np.abs(self.root.hist_GV.x)) if self.root.run_config['V_scale'] == 'log' else self.root.hist_GV.x
                    df.index = np.log10(np.abs(self.root.hist_GV.y)) if self.root.run_config['G_scale'] == 'log' else self.root.hist_GV.y
                    df.to_csv(path, sep=',')
                else:
                    np.savetxt(path, count, delimiter=",")
            case 2:
                path = tkinter.filedialog.asksaveasfilename(confirmoverwrite=True, initialfile=f'{tabname}.csv', defaultextension='.csv', filetypes=[('Comma delimited', '*.csv'), ('All Files', '*.*')])
                if not path: return
                count = self.root.hist_IV.height_per_trace.T if self.option_IV_count.get() == 'Count/trace' else self.root.hist_IV.height.T
                if self.check_IV_axis.get():
                    df = pd.DataFrame(count)
                    df.columns = np.log10(np.abs(self.root.hist_IV.x)) if self.root.run_config['V_scale'] == 'log' else self.root.hist_IV.x
                    df.index = np.log10(np.abs(self.root.hist_IV.y)) if self.root.run_config['I_scale'] == 'log' else self.root.hist_IV.y
                    df.to_csv(path, sep=',')
                else:
                    np.savetxt(path, count, delimiter=",")
            case 3:
                import yaml
                path = tkinter.filedialog.asksaveasfilename(confirmoverwrite=True, initialfile='config.yaml', defaultextension='.yaml', filetypes=[('YAML', '*.yaml'), ('All Files', '*.*')])
                if not path: return
                data = {
                    'Data type': self.root.is_raw.get(),
                    'Recursive': self.root.directory_recursive.get(),
                    'Mode': self.root.mode.get(),
                    'Ebias': self.root.Ebias.get(),
                    'I unit': self.root.I_unit.get(),
                    'V unit': self.root.V_unit.get(),
                    '#Segments': self.root.num_segment.get(),
                    'points_per_file': self.root.points_per_file.get(),
                    'Sampling rate': self.root.sampling_rate.get(),
                    'V upper': self.root.V_upper.get(),
                    'V lower': self.root.V_lower.get(),
                    'Length': self.root.length.get(),
                    'Tolerance': self.root.tolerance.get(),
                    'Offset0': self.root.offset0.get(),
                    'Offset1': self.root.offset1.get(),
                    'I limit': self.root.I_limit.get(),
                    'Noise remove': self.root.is_noise_remove.get(),
                    'V0': self.root.V0.get(),
                    'dV': self.root.dV.get(),
                    'Zeroing': self.root.is_zeroing.get(),
                    'Zeroing center': self.root.zeroing_center.get(),
                    'Direction': self.root.direction.get(),
                    'Extract_method': self.root.extract_method.get(),
                    'V min': self.root.V_min.get(),
                    'V max': self.root.V_max.get(),
                    'V #bins': self.root.V_bins.get(),
                    'V scale': self.root.V_scale.get(),
                    'G min': self.root.G_min.get(),
                    'G max': self.root.G_max.get(),
                    'G #bins': self.root.G_bins.get(),
                    'G scale': self.root.G_scale.get(),
                    'I max': self.root.I_min.get(),
                    'I min': self.root.I_max.get(),
                    'I #bins': self.root.I_bins.get(),
                    'I scale': self.root.I_scale.get(),
                    't max': self.root.t_min.get(),
                    't min': self.root.t_max.get(),
                    't #bins': self.root.t_bins.get(),
                    't scale': self.root.t_scale.get(),
                    'hist_GV': self.root.plot_hist_GV.get(),
                    'hist_IV': self.root.plot_hist_IV.get(),
                    'hist_IVt': self.root.plot_hist_IVt.get(),
                    'hist_GVt': self.root.plot_hist_GVt.get(),
                    'Colorbar': self.root.colorbar_conf.get('0.0', 'end')
                }
                if os.path.exists(path):
                    with open(path, mode='r', encoding='utf-8') as f:
                        old_data = yaml.load(f.read(), yaml.SafeLoader)
                        if old_data: old_data.update({'IVscan': data})
                        else: old_data = {'IVscan': data}
                    with open(path, mode='w', encoding='utf-8') as f:
                        data = yaml.dump(old_data, f, yaml.SafeDumper)
                else:
                    with open(path, mode='w', encoding='utf-8') as f:
                        data = yaml.dump({'IVscan': data}, f, yaml.SafeDumper)
        self.hide()


class CV_GUI:

    def __init__(self, root: tk.Frame) -> None:
        self.window = root
        # config frame
        self.frame_config = tk.Frame(self.window)
        self.frame_config.pack(side='top', anchor='w')
        # row 0
        self.directory_path = tk.StringVar(self.window)
        tk.Label(self.frame_config, text='Path: ').grid(row=0, column=0)
        tk.Entry(self.frame_config, textvariable=self.directory_path, width=40).grid(row=0, column=1, columnspan=4)
        tk.Button(self.frame_config, text="File", bg='#ffe9a2', command=lambda: _set_directory(self.directory_path, tkinter.filedialog.askopenfilename())).grid(row=0, column=6, padx=10)
        # row 1
        self.I_unit = tk.DoubleVar(self.window, value=-1e-6)
        self.V_unit = tk.DoubleVar(self.window, value=1)
        tk.Label(self.frame_config, text='Units (I, V): ').grid(row=1, column=0)
        frame_units = tk.Frame(self.frame_config)
        frame_units.grid(row=1, column=1)
        tk.Entry(frame_units, textvariable=self.I_unit, justify='center', width=10).pack(side='left')
        tk.Entry(frame_units, textvariable=self.V_unit, justify='center', width=10).pack(side='left')
        self.run_button = tk.Button(self.frame_config, text='Run', bg='lime', command=self.run)
        self.run_button.grid(row=1, column=6)
        # figure frame
        self.frame_figure = tk.Frame(self.window)
        self.frame_figure.pack(side='top', anchor='w')
        self.frame_figure.columnconfigure([0, 5, 10, 15], weight=1)
        self.frame_figure.rowconfigure([0], weight=1)
        self.queue = Queue()
        self.updatetk()

    def run(self):
        self.cleanup('partial')
        path = self.directory_path.get()
        for item in self.frame_figure.winfo_children():
            item.destroy()
        gc.collect()
        self.plot_CV = CV.PlotCV()
        self.canvas_CV = FigureCanvasTkAgg(self.plot_CV.fig, self.frame_figure)
        self.canvas_CV.get_tk_widget().grid(row=0, column=0, columnspan=5, pady=10)
        self.navtool_CV = NavigationToolbar2Tk(self.canvas_CV, self.frame_figure, pack_toolbar=False)
        self.navtool_CV.grid(row=1, column=0, columnspan=4, sticky='w')
        self.canvas_CV.draw_idle()
        self.window.update_idletasks()
        self.V, self.I = np.loadtxt(path, unpack=True) * [[self.V_unit.get()], [self.I_unit.get()]]
        self.plot_CV.add_data(self.V, self.I)
        self.queue.put(Queue_Item(self.canvas_CV.draw_idle))

    def updatetk(self):
        while not self.queue.empty():
            try:
                item: Queue_Item = self.queue.get()
                item.run()
            except Exception as E:
                logger.warning(f'{type(E).__name__}: {E.args}')
        self.window.after(100, self.updatetk)

    def cleanup(self, catagory: Literal['partial', 'all']):
        if hasattr(self, 'I'): del self.I
        if hasattr(self, 'V'): del self.V
        if hasattr(self, 'plot_CV'):
            self.plot_CV.fig.clear()
            del self.plot_CV
        if catagory == 'partial':
            gc.collect()
            return
        else:
            self.window.destroy()

class Logging_GUI(logging.Handler):

    def __init__(self):
        logging.Handler.__init__(self)
        self.window = tk.Toplevel()
        self.hide()
        self.window.protocol("WM_DELETE_WINDOW", self.hide)
        self.window.title('Log')
        self.window.resizable(True, True)
        self.logtext = tk.Text(self.window, height=30, width=120)
        self.logtext.pack(side='top', fill='both', expand=True)
        self.logtext.config(state='disabled')
        buttomframe = tk.Frame(self.window)
        buttomframe.pack(side='bottom')
        tk.OptionMenu(buttomframe, loglevel, *['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], command=self.set_level).pack(side='left')
        tk.Button(buttomframe, text='Save', command=self.save).pack(side='left')
        tk.Button(buttomframe, text='Clear', command=self.clear).pack(side='left')
        self.queue = Queue()
        self.updatetk()

    def emit(self, record):
        self.queue.put(Queue_Item(self.logtext.config, state='normal'))
        self.queue.put(Queue_Item(self.logtext.insert, tk.END, self.format(record) + '\n'))
        self.queue.put(Queue_Item(self.logtext.see, tk.END))
        self.queue.put(Queue_Item(self.logtext.config, state='disabled'))

    def set_level(self, level):
        logger.setLevel(level)

    def save(self):
        import datetime
        path = tkinter.filedialog.asksaveasfilename(confirmoverwrite=True, initialfile=f'{datetime.datetime.now().strftime("%Y%m%dT%H%M%S")}.log', defaultextension='.log', filetypes=[('Text Files', '*.log'), ('All Files', '*.*')])
        if path:
            text = self.logtext.get('0.0', 'end').strip()
            with open(path, mode='w', encoding='utf-8') as f:
                f.write(text)

    def clear(self):
        self.logtext.config(state='normal')
        self.logtext.delete('1.0', 'end')
        self.logtext.config(state='disabled')

    def updatetk(self):
        while not self.queue.empty():
            try:
                item: Queue_Item = self.queue.get()
                item.run()
            except Exception as E:
                logger.warning(f'{type(E).__name__}: {E.args}')
        self.window.after(100, self.updatetk)

    def show(self):
        self.window.deiconify()

    def hide(self):
        self.window.withdraw()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # PyInstaller
    import matplotlib
    matplotlib.use('TkAgg')
    plt.ioff()
    root = tk.Tk()
    import argparse
    parser = argparse.ArgumentParser(description='Run GUI')
    parser.add_argument('--level', type=str, default='WARNING')
    loglevel = tk.StringVar(root, value=parser.parse_args().level.upper())
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    handler = Logging_GUI()
    handler.setFormatter(formatter)
    logger = logging.getLogger('App')
    logger.addHandler(handler)
    logger.setLevel(loglevel.get())
    try:
        import yaml
        ini_config = os.path.join(os.getcwd(), 'config.yaml')
        if os.path.exists(ini_config):
            with open(ini_config, mode='r', encoding='utf-8') as f:
                ini_config = yaml.load(f.read(), yaml.SafeLoader)
        else:
            ini_config = None
    except ImportError:
        logger.warning('Module PyYAML was not found. Import/export settings can not be used.')
    GUI = Main()
    tk.mainloop()
