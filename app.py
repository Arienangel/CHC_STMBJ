import atexit
import json
import multiprocessing
import os
import sys
import time
import tkinter as tk
import tkinter.constants
import tkinter.filedialog
import tkinter.messagebox

import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.colors import LinearSegmentedColormap
from watchdog.events import FileCreatedEvent, FileSystemEventHandler
from watchdog.observers import Observer

import STM_bj
import I_Ebias


class Main:

    def __init__(self) -> None:
        self.window = tk.Tk()
        self.window.title('Select experiment')
        self.window.protocol("WM_DELETE_WINDOW", sys.exit)
        self.frame = tk.Frame(highlightthickness=2, highlightbackground='black')
        self.frame.pack(side='top', anchor='nw', padx=5, pady=5)
        tk.Label(self.frame, text='Experiment: ').pack(side='left')

        global CPU_threads
        CPU_threads = tk.IntVar(value=multiprocessing.cpu_count())
        self.experiment = tk.IntVar(value=0)
        tk.Radiobutton(self.frame, text='STM-bj', variable=self.experiment, value=1, command=self.start).pack(side='left')
        tk.Radiobutton(self.frame, text='I Ebias', variable=self.experiment, value=2, command=self.start).pack(side='left')
        tk.Radiobutton(self.frame, text='I Ewk', variable=self.experiment, value=3, command=self.start).pack(side='left')
        tk.Label(self.frame, text='CPU threads: ').pack(side='left', padx=10)
        tk.Entry(self.frame, textvariable=CPU_threads, width=10, justify='center').pack(side='left')
        self.iframe = tk.Frame()
        self.iframe.pack(side='top', anchor='nw')
        tk.mainloop()

    def start(self):
        for item in self.iframe.winfo_children():
            item.destroy()
        match self.experiment.get():
            case 1:
                self.window.title('STM-bj')
                STM_bj_GUI(self.iframe)
            case 2:
                self.window.title('I Ebias')
                I_Ebias_GUI(self.iframe)
            case 3:
                self.window.title('I Ewk')
                #I_Ewk_GUI(self.iframe)


class STM_bj_GUI(FileSystemEventHandler):

    def __init__(self, iframe: tk.Frame) -> None:
        self.window = iframe
        # config frame
        self.frame_conf = tk.Frame(self.window)
        self.frame_conf.pack(side='top', anchor='w')
        # row 0
        self.directory_path = tk.StringVar()
        self.directory_recursive = tk.BooleanVar(value=False)
        tk.Label(self.frame_conf, text='Path: ').grid(row=0, column=0)
        tk.Entry(self.frame_conf, textvariable=self.directory_path, width=80).grid(row=0, column=1, columnspan=5)
        tk.Button(self.frame_conf, text="Folder", command=lambda: self.directory_path.set(tkinter.filedialog.askdirectory())).grid(row=0, column=6, padx=5)
        tk.Checkbutton(self.frame_conf, variable=self.directory_recursive, text="Recursive").grid(row=0, column=7, padx=5, sticky='w')
        # row 1
        self.extract_length = tk.IntVar(value=2000)
        self.upper = tk.DoubleVar(value=1e-6)
        self.lower = tk.DoubleVar(value=3.2)
        tk.Label(self.frame_conf, text='Length: ').grid(row=1, column=0)
        tk.Entry(self.frame_conf, textvariable=self.extract_length, justify='center').grid(row=1, column=1)
        tk.Label(self.frame_conf, text='Upper: ').grid(row=1, column=2)
        tk.Entry(self.frame_conf, textvariable=self.lower, justify='center').grid(row=1, column=3)
        tk.Label(self.frame_conf, text='Lower: ').grid(row=1, column=4)
        tk.Entry(self.frame_conf, textvariable=self.upper, justify='center').grid(row=1, column=5)
        tk.Button(self.frame_conf, text="Files", command=lambda: self.directory_path.set(json.dumps(tkinter.filedialog.askopenfilenames(), ensure_ascii=False))).grid(row=1, column=6)
        # row 2
        self.zero_point = tk.DoubleVar(value=0.5)
        self.points_per_nm = tk.DoubleVar(value=800)
        self.direction = tk.StringVar(value='pull')
        tk.Label(self.frame_conf, text='X=0@G= ').grid(row=2, column=0)
        tk.Entry(self.frame_conf, textvariable=self.zero_point, justify='center').grid(row=2, column=1)
        tk.Label(self.frame_conf, text='Points/nm: ').grid(row=2, column=2)
        tk.Entry(self.frame_conf, textvariable=self.points_per_nm, justify='center').grid(row=2, column=3)
        tk.Label(self.frame_conf, text='Direction: ').grid(row=2, column=4)
        tk.OptionMenu(self.frame_conf, self.direction, *['pull', 'crash', 'both']).grid(row=2, column=5)
        # row 3
        self.G_min = tk.DoubleVar(value=0.00001)
        self.G_max = tk.DoubleVar(value=3.16)
        self.G_bins = tk.IntVar(value=550)
        self.G_scale = tk.StringVar(value='log')
        tk.Label(self.frame_conf, text='G min: ').grid(row=3, column=0)
        tk.Entry(self.frame_conf, textvariable=self.G_min, justify='center').grid(row=3, column=1)
        tk.Label(self.frame_conf, text='G max: ').grid(row=3, column=2)
        tk.Entry(self.frame_conf, textvariable=self.G_max, justify='center').grid(row=3, column=3)
        tk.Label(self.frame_conf, text='G #bins: ').grid(row=3, column=4)
        tk.Entry(self.frame_conf, textvariable=self.G_bins, justify='center').grid(row=3, column=5)
        tk.Label(self.frame_conf, text='G scale: ').grid(row=3, column=6)
        tk.OptionMenu(self.frame_conf, self.G_scale, *['log', 'linear']).grid(row=3, column=7)
        # row 4
        self.X_min = tk.DoubleVar(value=-0.4)
        self.X_max = tk.DoubleVar(value=0.6)
        self.X_bins = tk.IntVar(value=1000)
        self.X_scale = tk.StringVar(value='linear')
        tk.Label(self.frame_conf, text='X min: ').grid(row=4, column=0)
        tk.Entry(self.frame_conf, textvariable=self.X_min, justify='center').grid(row=4, column=1)
        tk.Label(self.frame_conf, text='X max: ').grid(row=4, column=2)
        tk.Entry(self.frame_conf, textvariable=self.X_max, justify='center').grid(row=4, column=3)
        tk.Label(self.frame_conf, text='X #bins: ').grid(row=4, column=4)
        tk.Entry(self.frame_conf, textvariable=self.X_bins, justify='center').grid(row=4, column=5)
        tk.Label(self.frame_conf, text='X scale: ').grid(row=4, column=6)
        tk.OptionMenu(self.frame_conf, self.X_scale, *['log', 'linear']).grid(row=4, column=7)
        # row 5
        tk.Label(self.frame_conf, text='Colorbar: ').grid(row=5, column=0)
        self.colorbar_conf = tk.Text(self.frame_conf, height=3, wrap='none')
        self.colorbar_conf.grid(row=5, column=1, columnspan=5, sticky='w')
        self.colorbar_conf.insert(
            '0.0', '{"red":   [[0, 1, 1], [0.05, 0, 0], [0.1, 0, 0], [0.15, 1, 1], [0.3, 1, 1], [1, 1, 1]],\n "green": [[0, 1, 1], [0.05, 0, 0], [0.1, 1, 1], [0.15, 1, 1], [0.3, 0, 0], [1, 0, 0]],\n "blue":  [[0, 1, 1], [0.05, 1, 1], [0.1, 0, 0], [0.15, 0, 0], [0.3, 0, 0], [1, 1, 1]]}')
        self.run_button = tk.Button(self.frame_conf, text='Run', bg='lime', command=self.run)
        self.run_button.grid(row=5, column=6, padx=10)
        self.is_run = False
        # figure frame
        self.frame_fig = tk.Frame(self.window)
        self.frame_fig.pack(side='top', anchor='w')
        # status frame
        self.frame_stat = tk.Frame(self.window)
        self.frame_stat.pack(side='bottom', anchor='w')
        tk.Label(self.frame_stat, text='#Traces: ').pack(side='left')
        self.stat_traces = tk.Label(self.frame_stat, text=0)
        self.stat_traces.pack(side='left')
        tk.Label(self.frame_stat, text='File: ', padx=20).pack(side='left')
        self.stat_lastfile = tk.Label(self.frame_stat, text='Waiting')
        self.stat_lastfile.pack(side='left')

    def run(self):
        if not self.is_run:
            if os.path.isdir(self.directory_path.get()):
                path = self.directory_path.get()
            else:
                try:
                    path = json.loads(self.directory_path.get())
                except:
                    tkinter.messagebox.showerror('Error', 'Invalid directory')
                    return
            plt.close()
            for item in self.frame_fig.winfo_children():
                item.destroy()
            self.extract_data = np.empty((0, self.extract_length.get()))
            self.hist_G = STM_bj.Hist_G([self.G_min.get(), self.G_max.get()], self.G_bins.get(), self.G_scale.get())
            self.hist_GS = STM_bj.Hist_GS([self.X_min.get(), self.X_max.get()], [self.G_min.get(), self.G_max.get()], self.X_bins.get(), self.G_bins.get(), self.X_scale.get(), self.G_scale.get(), self.zero_point.get(), self.points_per_nm.get())
            try:
                colorbar_conf = self.colorbar_conf.get('0.0', 'end')
                if colorbar_conf != "\n":
                    self.hist_GS.plot.set_cmap(cmap=LinearSegmentedColormap('Cmap', segmentdata=json.loads(colorbar_conf), N=256))
            except:
                tkinter.messagebox.showwarning('Warning', 'Invalid colorbar setting')
            self.canvas_G = FigureCanvasTkAgg(self.hist_G.fig, self.frame_fig)
            self.canvas_G.get_tk_widget().grid(row=0, column=0, columnspan=4, pady=10)
            self.navtool_G = NavigationToolbar2Tk(self.canvas_G, self.frame_fig, pack_toolbar=False)
            self.navtool_G.grid(row=1, column=0, columnspan=4, sticky=tkinter.constants.W)
            self.canvas_GS = FigureCanvasTkAgg(self.hist_GS.fig, self.frame_fig)
            self.canvas_GS.get_tk_widget().grid(row=0, column=5, columnspan=4, pady=10)
            self.navtool_GS = NavigationToolbar2Tk(self.canvas_GS, self.frame_fig, pack_toolbar=False)
            self.navtool_GS.grid(row=1, column=5, columnspan=4, sticky=tkinter.constants.W)
            self.extract_config = {
                "length": self.extract_length.get(),
                "upper": self.upper.get(),
                "lower": self.lower.get(),
                "method": self.direction.get(),
                "zero_point": self.zero_point.get(),
                "x_conversion": self.points_per_nm.get(),
                'G_scale': self.G_scale.get(),
                'X_scale': self.X_scale.get(),
                'recursive': self.directory_recursive.get()
            }
            self.add_data(path)
            tk.Button(self.frame_conf, text='Export', command=self.export).grid(row=5, column=7, padx=10)
            if isinstance(path, list): return
            self.observer = Observer()
            self.observer.schedule(self, path=self.directory_path.get(), recursive=self.directory_recursive.get())
            self.observer.start()
            atexit.register(self.observer.stop)
            self.run_button.config(text='Stop', bg='red')
            self.is_run = True
        else:
            self.run_button.config(text='Run', bg='lime')
            self.is_run = False

    def on_created(self, event):
        if self.is_run is False:
            self.observer.stop()
            return
        if isinstance(event, FileCreatedEvent):
            if (event.src_path.endswith('.txt')):
                try:
                    if os.path.getsize(event.src_path) == 0: time.sleep(0.1)
                    self.add_data(event.src_path)
                except Exception as E:
                    tkinter.messagebox.showwarning('Warning', f'{type(E).__name__}: {E.args}')

    def add_data(self, path: str | list):
        if isinstance(path, str):
            self.stat_lastfile.config(text=path)
            if not os.listdir(path): return  # empty directory
        else:
            self.stat_lastfile.config(text="(Multiple)")
        extracted = STM_bj.extract_data(path, **self.extract_config, threads=CPU_threads.get())
        self.extract_data = np.vstack([self.extract_data, extracted])
        self.hist_G.add_data(extracted)
        self.hist_GS.add_data(extracted)
        self.canvas_G.draw()
        self.canvas_GS.draw()
        self.stat_traces.config(text=self.hist_G.trace)

    def export(self):
        self.Export_prompt(self.extract_data, self.hist_G, self.hist_GS, **self.extract_config)

    class Export_prompt:

        def __init__(self, G: np.ndarray, hist_G: STM_bj.Hist_G, hist_GS: STM_bj.Hist_GS, **conf) -> None:
            self.window = tk.Toplevel()
            self.window.grab_set()
            self.window.title('Export')
            self.G = G
            self.hist_G = hist_G
            self.hist_GS = hist_GS
            self.conf = conf
            # col 0
            self.export_type = tk.IntVar(value=1)
            tk.Radiobutton(self.window, text='Raw data', variable=self.export_type, value=1).grid(row=0, column=0, sticky='w')
            tk.Radiobutton(self.window, text='1D histogram', variable=self.export_type, value=2).grid(row=1, column=0, sticky='w')
            tk.Radiobutton(self.window, text='2D histogram', variable=self.export_type, value=3).grid(row=2, column=0, sticky='w')
            # row 0
            self.check_raw_X = tk.BooleanVar(value=True)  #disabled
            self.check_raw_G = tk.BooleanVar(value=True)
            self.check_raw_logG = tk.BooleanVar(value=True)
            tk.Checkbutton(self.window, variable=self.check_raw_X, text='X', state='disabled').grid(row=0, column=1, sticky='w')
            tk.Checkbutton(self.window, variable=self.check_raw_G, text='G').grid(row=0, column=2, sticky='w')
            tk.Checkbutton(self.window, variable=self.check_raw_logG, text='logG').grid(row=0, column=3, sticky='w')
            # row 1
            self.check_1D_G = tk.BooleanVar(value=True)  #disabled
            self.check_1D_count = tk.BooleanVar(value=True)  #disabled
            self.check_1D_pertraces = tk.BooleanVar(value=False)
            tk.Checkbutton(self.window, variable=self.check_1D_G, text='G', state='disabled').grid(row=1, column=1, sticky='w')
            tk.Checkbutton(self.window, variable=self.check_1D_count, text='Count', state='disabled').grid(row=1, column=2, sticky='w')
            tk.Checkbutton(self.window, variable=self.check_1D_pertraces, text='per traces').grid(row=1, column=3, sticky='w')
            # row 2
            self.check_2D_axis = tk.BooleanVar(value=False)
            self.check_2D_count = tk.BooleanVar(value=True)  #disabled
            self.check_2D_pertraces = tk.BooleanVar(value=False)
            tk.Checkbutton(self.window, variable=self.check_2D_axis, text='Axis').grid(row=2, column=1, sticky='w')
            tk.Checkbutton(self.window, variable=self.check_2D_count, text='Count', state='disabled').grid(row=2, column=2, sticky='w')
            tk.Checkbutton(self.window, variable=self.check_2D_pertraces, text='per traces').grid(row=2, column=3, sticky='w')
            # row 3
            tk.Button(self.window, text='Export', command=self.run).grid(row=3, columnspan=4)

        def run(self):
            path = tkinter.filedialog.asksaveasfilename(confirmoverwrite=True, defaultextension='.csv', filetypes=[('Comma delimited', '*.csv'), ('All Files', '*.*')])
            if path:
                match self.export_type.get():
                    case 1:
                        X = STM_bj.get_displacement(self.G, **self.conf).ravel()
                        if self.check_raw_G.get(): X = np.vstack([X, self.G.ravel()])
                        if self.check_raw_logG.get(): X = np.vstack([X, np.log10(np.abs(self.G)).ravel()])
                        np.savetxt(path, X.T, delimiter=",")
                    case 2:
                        G = np.log10(np.abs(self.hist_G.x)) if self.conf['G_scale'] == 'log' else self.hist_G.x
                        count = self.hist_G.height_per_trace if self.check_1D_pertraces.get() else self.hist_G.height
                        np.savetxt(path, np.vstack([G, count]).T, delimiter=',')
                    case 3:
                        count = self.hist_GS.height_per_trace.T if self.check_2D_pertraces.get() else self.hist_GS.height.T
                        if self.check_2D_axis.get():
                            df = pd.DataFrame(count)
                            df.columns = np.log10(np.abs(self.hist_GS.x)) if self.conf['X_scale'] == 'log' else self.hist_GS.x
                            df.index = np.log10(np.abs(self.hist_GS.y)) if self.conf['G_scale'] == 'log' else self.hist_GS.y
                            df.to_csv(path, sep=',')
                        else:
                            np.savetxt(path, count, delimiter=",")


class I_Ebias_GUI(FileSystemEventHandler):

    def __init__(self, iframe: tk.Frame) -> None:
        self.window = iframe
        # config frame
        self.frame_conf = tk.Frame(self.window)
        self.frame_conf.pack(side='top', anchor='w')
        # row 0
        self.directory_path = tk.StringVar()
        self.directory_recursive = tk.BooleanVar(value=False)
        self.num_files = tk.IntVar(value=10)
        self.num_segments = tk.IntVar(value=4)
        tk.Label(self.frame_conf, text='Path: ').grid(row=0, column=0)
        tk.Entry(self.frame_conf, textvariable=self.directory_path, width=80).grid(row=0, column=1, columnspan=5)
        tk.Button(self.frame_conf, text="Folder", command=lambda: self.directory_path.set(tkinter.filedialog.askdirectory())).grid(row=0, column=6, padx=5)
        tk.Checkbutton(self.frame_conf, variable=self.directory_recursive, text="Recursive").grid(row=0, column=7, padx=5, sticky='w')
        # row 1
        self.V_upper = tk.DoubleVar(value=1.45)
        self.length = tk.IntVar(value=1200)
        self.I_unit = tk.DoubleVar(value=1e-6)
        self.V_unit = tk.DoubleVar(value=1)
        tk.Label(self.frame_conf, text='E upper: ').grid(row=1, column=0)
        tk.Entry(self.frame_conf, textvariable=self.V_upper, justify='center').grid(row=1, column=1)
        tk.Label(self.frame_conf, text='Length: ').grid(row=1, column=2)
        tk.Entry(self.frame_conf, textvariable=self.length, justify='center').grid(row=1, column=3)
        tk.Label(self.frame_conf, text='Units (I, V): ').grid(row=1, column=4)
        frame_offset = tk.Frame(self.frame_conf)
        frame_offset.grid(row=1, column=5)
        tk.Entry(frame_offset, textvariable=self.I_unit, justify='center', width=10).pack(side='left')
        tk.Entry(frame_offset, textvariable=self.V_unit, justify='center', width=10).pack(side='left')
        tk.Button(self.frame_conf, text="Files", command=lambda: self.directory_path.set(json.dumps(tkinter.filedialog.askopenfilenames(), ensure_ascii=False))).grid(row=1, column=6)
        tk.Label(self.frame_conf, text='#Files/#Segments: ').grid(row=1, column=7)
        frame_folderconf = tk.Frame(self.frame_conf)
        frame_folderconf.grid(row=1, column=8)
        tk.Entry(frame_folderconf, textvariable=self.num_files, justify='center', width=10).pack(side='left')
        tk.Entry(frame_folderconf, textvariable=self.num_segments, justify='center', width=10).pack(side='left')

        # row 2
        self.V_range = tk.DoubleVar(value=0.1)
        self.I_limit = tk.DoubleVar(value=1e-5)
        self.direction = tk.StringVar(value='both')
        tk.Label(self.frame_conf, text='I min@V< ').grid(row=2, column=0)
        tk.Entry(self.frame_conf, textvariable=self.V_range, justify='center').grid(row=2, column=1)
        tk.Label(self.frame_conf, text='I limit: ').grid(row=2, column=2)
        tk.Entry(self.frame_conf, textvariable=self.I_limit, justify='center').grid(row=2, column=3)
        tk.Label(self.frame_conf, text='Direction: ').grid(row=2, column=4)
        tk.OptionMenu(self.frame_conf, self.direction, *['both', '-→+', '+→-']).grid(row=2, column=5)
        self.check_zeroing = tk.BooleanVar(value=True)
        tk.Checkbutton(self.frame_conf, variable=self.check_zeroing, text='Zeroing').grid(row=2, column=6, sticky='w')
        # row 3
        self.V_min = tk.DoubleVar(value=-1.5)
        self.V_max = tk.DoubleVar(value=1.5)
        self.V_bins = tk.IntVar(value=300)
        self.V_scale = tk.StringVar(value='linear')
        tk.Label(self.frame_conf, text='E min: ').grid(row=3, column=0)
        tk.Entry(self.frame_conf, textvariable=self.V_min, justify='center').grid(row=3, column=1)
        tk.Label(self.frame_conf, text='E max: ').grid(row=3, column=2)
        tk.Entry(self.frame_conf, textvariable=self.V_max, justify='center').grid(row=3, column=3)
        tk.Label(self.frame_conf, text='E #bins: ').grid(row=3, column=4)
        tk.Entry(self.frame_conf, textvariable=self.V_bins, justify='center').grid(row=3, column=5)
        tk.Label(self.frame_conf, text='E scale: ').grid(row=3, column=6)
        tk.OptionMenu(self.frame_conf, self.V_scale, *['log', 'linear']).grid(row=3, column=7)
        # row 4
        self.G_min = tk.DoubleVar(value=1e-5)
        self.G_max = tk.DoubleVar(value=1e-1)
        self.G_bins = tk.IntVar(value=400)
        self.G_scale = tk.StringVar(value='log')
        tk.Label(self.frame_conf, text='G min: ').grid(row=4, column=0)
        tk.Entry(self.frame_conf, textvariable=self.G_min, justify='center').grid(row=4, column=1)
        tk.Label(self.frame_conf, text='G max: ').grid(row=4, column=2)
        tk.Entry(self.frame_conf, textvariable=self.G_max, justify='center').grid(row=4, column=3)
        tk.Label(self.frame_conf, text='G #bins: ').grid(row=4, column=4)
        tk.Entry(self.frame_conf, textvariable=self.G_bins, justify='center').grid(row=4, column=5)
        tk.Label(self.frame_conf, text='G scale: ').grid(row=4, column=6)
        tk.OptionMenu(self.frame_conf, self.G_scale, *['log', 'linear']).grid(row=4, column=7)
        # row 5
        self.I_min = tk.DoubleVar(value=1e-11)
        self.I_max = tk.DoubleVar(value=1e-5)
        self.I_bins = tk.IntVar(value=600)
        self.I_scale = tk.StringVar(value='log')
        tk.Label(self.frame_conf, text='I min: ').grid(row=5, column=0)
        tk.Entry(self.frame_conf, textvariable=self.I_min, justify='center').grid(row=5, column=1)
        tk.Label(self.frame_conf, text='I max: ').grid(row=5, column=2)
        tk.Entry(self.frame_conf, textvariable=self.I_max, justify='center').grid(row=5, column=3)
        tk.Label(self.frame_conf, text='I #bins: ').grid(row=5, column=4)
        tk.Entry(self.frame_conf, textvariable=self.I_bins, justify='center').grid(row=5, column=5)
        tk.Label(self.frame_conf, text='I scale: ').grid(row=5, column=6)
        tk.OptionMenu(self.frame_conf, self.I_scale, *['log', 'linear']).grid(row=5, column=7)
        # row 6
        tk.Label(self.frame_conf, text='Colorbar: ').grid(row=6, column=0)
        self.colorbar_conf = tk.Text(self.frame_conf, height=3, wrap='none')
        self.colorbar_conf.grid(row=6, column=1, columnspan=5, sticky='w')
        self.colorbar_conf.insert(
            '0.0', '{"red":   [[0, 1, 1], [0.05, 0, 0], [0.1, 0, 0], [0.15, 1, 1], [0.3, 1, 1], [1, 1, 1]],\n "green": [[0, 1, 1], [0.05, 0, 0], [0.1, 1, 1], [0.15, 1, 1], [0.3, 0, 0], [1, 0, 0]],\n "blue":  [[0, 1, 1], [0.05, 1, 1], [0.1, 0, 0], [0.15, 0, 0], [0.3, 0, 0], [1, 1, 1]]}')
        self.run_button = tk.Button(self.frame_conf, text='Run', bg='lime', command=self.run)
        self.run_button.grid(row=6, column=6, padx=10)
        self.is_run = False
        # figure frame
        self.frame_fig = tk.Frame(self.window)
        self.frame_fig.pack(side='top', anchor='w')
        # status frame
        self.frame_stat = tk.Frame(self.window)
        self.frame_stat.pack(side='bottom', anchor='w')
        tk.Label(self.frame_stat, text='#Segments: ').pack(side='left')
        self.stat_traces = tk.Label(self.frame_stat, text=0)
        self.stat_traces.pack(side='left')
        tk.Label(self.frame_stat, text='File: ', padx=20).pack(side='left')
        self.stat_lastfile = tk.Label(self.frame_stat, text='Waiting')
        self.stat_lastfile.pack(side='left')

    def run(self):
        if not self.is_run:
            if os.path.isdir(self.directory_path.get()):
                path = self.directory_path.get()
            else:
                try:
                    path = json.loads(self.directory_path.get())
                except:
                    tkinter.messagebox.showerror('Error', 'Invalid directory')
                    return
            plt.close()
            for item in self.frame_fig.winfo_children():
                item.destroy()
            self.I = np.empty((0, self.length.get()))
            self.V = np.empty((0, self.length.get()))
            self.hist_GV = I_Ebias.Hist_GV([self.V_min.get(), self.V_max.get()], [self.G_min.get(), self.G_max.get()], self.V_bins.get(), self.G_bins.get(), self.V_scale.get(), self.G_scale.get())
            self.hist_IV = I_Ebias.Hist_IV([self.V_min.get(), self.V_max.get()], [self.I_min.get(), self.I_max.get()], self.V_bins.get(), self.I_bins.get(), self.V_scale.get(), self.I_scale.get())
            try:
                colorbar_conf = self.colorbar_conf.get('0.0', 'end')
                if colorbar_conf != "\n":
                    cmap = LinearSegmentedColormap('Cmap', segmentdata=json.loads(colorbar_conf), N=256)
                    self.hist_GV.plot.set_cmap(cmap=cmap)
                    self.hist_IV.plot.set_cmap(cmap=cmap)
            except:
                tkinter.messagebox.showwarning('Warning', 'Invalid colorbar setting')
            self.canvas_GV = FigureCanvasTkAgg(self.hist_GV.fig, self.frame_fig)
            self.canvas_GV.get_tk_widget().grid(row=0, column=0, columnspan=4, pady=10)
            self.navtool_GV = NavigationToolbar2Tk(self.canvas_GV, self.frame_fig, pack_toolbar=False)
            self.navtool_GV.grid(row=1, column=0, columnspan=4, sticky=tkinter.constants.W)
            self.canvas_IV = FigureCanvasTkAgg(self.hist_IV.fig, self.frame_fig)
            self.canvas_IV.get_tk_widget().grid(row=0, column=5, columnspan=4, pady=10)
            self.navtool_IV = NavigationToolbar2Tk(self.canvas_IV, self.frame_fig, pack_toolbar=False)
            self.navtool_IV.grid(row=1, column=5, columnspan=4, sticky=tkinter.constants.W)
            self.extract_config = {
                "height": self.V_upper.get(),
                "length": self.length.get(),
                "units": (self.I_unit.get(), self.V_unit.get()),
                "V_range": self.V_range.get(),
                "I_max": self.I_limit.get(),
                'V_scale': self.V_scale.get(),
                'G_scale': self.G_scale.get(),
                'I_scale': self.I_scale.get(),
                'recursive': self.directory_recursive.get(),
                'num_files': self.num_files.get(),
                'zeroing': self.check_zeroing.get()
            }
            self.pending = list()
            self.add_data(path)
            tk.Button(self.frame_conf, text='Export', command=self.export).grid(row=6, column=7, padx=10)
            if isinstance(path, list): return
            self.observer = Observer()
            self.observer.schedule(self, path=self.directory_path.get(), recursive=self.directory_recursive.get())
            self.observer.start()
            atexit.register(self.observer.stop)
            self.run_button.config(text='Stop', bg='red')
            self.is_run = True
        else:
            self.run_button.config(text='Run', bg='lime')
            self.is_run = False

    def on_created(self, event):
        if self.is_run is False:
            self.observer.stop()
            return
        if isinstance(event, FileCreatedEvent):
            if (event.src_path.endswith('.txt')):
                try:
                    if os.path.getsize(event.src_path) == 0: time.sleep(0.1)
                    self.add_data(event.src_path)
                except Exception as E:
                    tkinter.messagebox.showwarning('Warning', f'{type(E).__name__}: {E.args}')

    def add_data(self, path: str | list):
        if isinstance(path, str):
            self.stat_lastfile.config(text=path)
            if not os.listdir(path): return  # empty directory
        else:
            self.stat_lastfile.config(text="(Multiple)")
        self.pending.append(path)
        I, V = I_Ebias.extract_data(self.pending[-self.num_files.get():], **self.extract_config, threads=CPU_threads.get())
        if I.shape[0] < self.num_segments.get(): return
        else:
            I, V = I_Ebias.noise_remove(I, V, **self.extract_config)
            if self.extract_config['zeroing']: I, V = I_Ebias.zeroing(I, V)
            if I.size:
                match self.direction.get():
                    case '-→+':
                        I, V = I_Ebias.split_scan_direction(I, V)[0]
                    case '+→-':
                        I, V = I_Ebias.split_scan_direction(I, V)[1]
                self.hist_GV.add_data(I, V)
                self.hist_IV.add_data(I, V)
                self.canvas_GV.draw()
                self.canvas_IV.draw()
                self.stat_traces.config(text=self.hist_GV.trace)
                self.I = np.vstack([self.I, I])
                self.V = np.vstack([self.V, V])
            self.pending.clear()

    def export(self):
        self.Export_prompt(self.I, self.V, self.hist_GV, self.hist_IV, **self.extract_config)

    class Export_prompt:

        def __init__(self, I: np.ndarray, V: np.ndarray, hist_GV: I_Ebias.Hist_GV, hist_IV: I_Ebias.Hist_IV, **conf) -> None:
            self.window = tk.Toplevel()
            self.window.grab_set()
            self.window.title('Export')
            self.I = I
            self.V = V
            self.hist_GV = hist_GV
            self.hist_IV = hist_IV
            self.conf = conf
            # col 0
            self.export_type = tk.IntVar(value=1)
            tk.Radiobutton(self.window, text='Raw data', variable=self.export_type, value=1).grid(row=0, column=0, rowspan=2, sticky='w')
            tk.Radiobutton(self.window, text='GV histogram', variable=self.export_type, value=2).grid(row=2, column=0, sticky='w')
            tk.Radiobutton(self.window, text='IV histogram', variable=self.export_type, value=3).grid(row=3, column=0, sticky='w')
            # row 0
            self.check_raw_V = tk.BooleanVar(value=True)  #disabled
            self.check_raw_G = tk.BooleanVar(value=True)
            self.check_raw_logG = tk.BooleanVar(value=True)
            tk.Checkbutton(self.window, variable=self.check_raw_V, text='V', state='disabled').grid(row=0, column=1, sticky='w')
            tk.Checkbutton(self.window, variable=self.check_raw_G, text='G').grid(row=0, column=2, sticky='w')
            tk.Checkbutton(self.window, variable=self.check_raw_logG, text='logG').grid(row=0, column=3, sticky='w')
            # row 1
            self.check_raw_I = tk.BooleanVar(value=True)
            self.check_raw_absI = tk.BooleanVar(value=True)
            self.check_raw_logI = tk.BooleanVar(value=True)
            tk.Checkbutton(self.window, variable=self.check_raw_I, text='I').grid(row=1, column=1, sticky='w')
            tk.Checkbutton(self.window, variable=self.check_raw_absI, text='| I |').grid(row=1, column=2, sticky='w')
            tk.Checkbutton(self.window, variable=self.check_raw_logI, text='logI').grid(row=1, column=3, sticky='w')
            # row 2
            self.check_GV_axis = tk.BooleanVar(value=False)
            self.check_GV_count = tk.BooleanVar(value=True)  #disabled
            self.check_GV_pertraces = tk.BooleanVar(value=False)
            tk.Checkbutton(self.window, variable=self.check_GV_axis, text='Axis').grid(row=2, column=1, sticky='w')
            tk.Checkbutton(self.window, variable=self.check_GV_count, text='Count', state='disabled').grid(row=2, column=2, sticky='w')
            tk.Checkbutton(self.window, variable=self.check_GV_pertraces, text='per traces').grid(row=2, column=3, sticky='w')
            # row 3
            self.check_IV_axis = tk.BooleanVar(value=False)
            self.check_IV_count = tk.BooleanVar(value=True)  #disabled
            self.check_IV_pertraces = tk.BooleanVar(value=False)
            tk.Checkbutton(self.window, variable=self.check_IV_axis, text='Axis').grid(row=3, column=1, sticky='w')
            tk.Checkbutton(self.window, variable=self.check_IV_count, text='Count', state='disabled').grid(row=3, column=2, sticky='w')
            tk.Checkbutton(self.window, variable=self.check_IV_pertraces, text='per traces').grid(row=3, column=3, sticky='w')
            # row 4
            tk.Button(self.window, text='Export', command=self.run).grid(row=4, columnspan=4)

        def run(self):
            path = tkinter.filedialog.asksaveasfilename(confirmoverwrite=True, defaultextension='.csv', filetypes=[('Comma delimited', '*.csv'), ('All Files', '*.*')])
            if path:
                match self.export_type.get():
                    case 1:
                        V = self.V.ravel()
                        if self.check_raw_I.get(): V = np.vstack([V, self.I.ravel()])
                        if self.check_raw_absI.get(): V = np.vstack([V, np.abs(self.I.ravel())])
                        if self.check_raw_logI.get(): V = np.vstack([V, np.log10(np.abs(self.I.ravel()))])
                        G = I_Ebias.conductance(self.I, self.V).ravel()
                        if self.check_raw_G.get(): V = np.vstack([V, G])
                        if self.check_raw_logG.get(): V = np.vstack([V, np.log10(np.abs(G))])
                        np.savetxt(path, V.T, delimiter=",")
                    case 2:
                        count = self.hist_GV.height_per_trace.T if self.check_GV_pertraces.get() else self.hist_GV.height.T
                        if self.check_GV_axis.get():
                            df = pd.DataFrame(count)
                            df.columns = np.log10(np.abs(self.hist_GV.x)) if self.conf['V_scale'] == 'log' else self.hist_GV.x
                            df.index = np.log10(np.abs(self.hist_GV.y)) if self.conf['G_scale'] == 'log' else self.hist_GV.y
                            df.to_csv(path, sep=',')
                        else:
                            np.savetxt(path, count, delimiter=",")
                    case 3:
                        count = self.hist_IV.height_per_trace.T if self.check_IV_pertraces.get() else self.hist_IV.height.T
                        if self.check_IV_axis.get():
                            df = pd.DataFrame(count)
                            df.columns = np.log10(np.abs(self.hist_IV.x)) if self.conf['V_scale'] == 'log' else self.hist_IV.x
                            df.index = np.log10(np.abs(self.hist_IV.y)) if self.conf['I_scale'] == 'log' else self.hist_IV.y
                            df.to_csv(path, sep=',')
                        else:
                            np.savetxt(path, count, delimiter=",")


if __name__ == '__main__':
    multiprocessing.freeze_support()  # PyInstaller
    matplotlib.use('TkAgg')
    GUI = Main()
