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


class STM_bj_GUI(FileSystemEventHandler):

    def __init__(self) -> None:
        self.window = tk.Tk()
        self.window.title('STM-bj')
        self.window.protocol("WM_DELETE_WINDOW", sys.exit)
        self.frame_conf = tk.Frame(self.window)
        self.frame_conf.pack(side='top', anchor='w')
        self.frame_fig = tk.Frame(self.window)
        self.frame_fig.pack(side='top', anchor='w')
        self.frame_stat = tk.Frame(self.window)
        self.frame_stat.pack(side='bottom', anchor='w')
        self.directory_path = tk.StringVar()
        self.directory_recursive = tk.BooleanVar(value=False)
        self.extract_length = tk.IntVar(value=2000)
        self.lower = tk.DoubleVar(value=3.2)
        self.upper = tk.DoubleVar(value=1e-6)
        self.zero_point = tk.DoubleVar(value=0.5)
        self.points_per_nm = tk.DoubleVar(value=800)
        self.direction = tk.StringVar(value='pull')
        self.G_min = tk.DoubleVar(value=0.00001)
        self.G_max = tk.DoubleVar(value=3.16)
        self.G_bins = tk.IntVar(value=550)
        self.G_scale = tk.StringVar(value='log')
        self.X_min = tk.DoubleVar(value=-0.4)
        self.X_max = tk.DoubleVar(value=0.6)
        self.X_bins = tk.IntVar(value=1000)
        self.X_scale = tk.StringVar(value='linear')
        self.isrun = False

    def start(self):
        tk.Label(self.frame_conf, text='Path: ').grid(row=0, column=0)
        tk.Entry(self.frame_conf, textvariable=self.directory_path, width=80).grid(row=0, column=1, columnspan=5)
        tk.Button(self.frame_conf, text="Folder", command=lambda: self.directory_path.set(tkinter.filedialog.askdirectory())).grid(row=0, column=6, padx=5)
        tk.Checkbutton(self.frame_conf, variable=self.directory_recursive, text="Recursive").grid(row=1, column=6, columnspan=2, padx=5, sticky='w')
        tk.Button(self.frame_conf, text="Files", command=lambda: self.directory_path.set(json.dumps(tkinter.filedialog.askopenfilenames(), ensure_ascii=False))).grid(row=0, column=7)
        tk.Label(self.frame_conf, text='Length: ').grid(row=1, column=0)
        tk.Entry(self.frame_conf, textvariable=self.extract_length, justify='center').grid(row=1, column=1)
        tk.Label(self.frame_conf, text='Upper: ').grid(row=1, column=2)
        tk.Entry(self.frame_conf, textvariable=self.lower, justify='center').grid(row=1, column=3)
        tk.Label(self.frame_conf, text='Lower: ').grid(row=1, column=4)
        tk.Entry(self.frame_conf, textvariable=self.upper, justify='center').grid(row=1, column=5)
        tk.Label(self.frame_conf, text='X=0@G= ').grid(row=2, column=0)
        tk.Entry(self.frame_conf, textvariable=self.zero_point, justify='center').grid(row=2, column=1)
        tk.Label(self.frame_conf, text='Points/nm: ').grid(row=2, column=2)
        tk.Entry(self.frame_conf, textvariable=self.points_per_nm, justify='center').grid(row=2, column=3)
        tk.Label(self.frame_conf, text='Direction: ').grid(row=2, column=4)
        tk.OptionMenu(self.frame_conf, self.direction, *['pull', 'crash', 'both']).grid(row=2, column=5)
        tk.Label(self.frame_conf, text='Gmin: ').grid(row=3, column=0)
        tk.Entry(self.frame_conf, textvariable=self.G_min, justify='center').grid(row=3, column=1)
        tk.Label(self.frame_conf, text='Gmax: ').grid(row=3, column=2)
        tk.Entry(self.frame_conf, textvariable=self.G_max, justify='center').grid(row=3, column=3)
        tk.Label(self.frame_conf, text='#Gbins: ').grid(row=3, column=4)
        tk.Entry(self.frame_conf, textvariable=self.G_bins, justify='center').grid(row=3, column=5)
        tk.Label(self.frame_conf, text='Gscale: ').grid(row=3, column=6)
        tk.OptionMenu(self.frame_conf, self.G_scale, *['log', 'linear']).grid(row=3, column=7)
        tk.Label(self.frame_conf, text='Xmin: ').grid(row=4, column=0)
        tk.Entry(self.frame_conf, textvariable=self.X_min, justify='center').grid(row=4, column=1)
        tk.Label(self.frame_conf, text='Xmax: ').grid(row=4, column=2)
        tk.Entry(self.frame_conf, textvariable=self.X_max, justify='center').grid(row=4, column=3)
        tk.Label(self.frame_conf, text='#Xbins: ').grid(row=4, column=4)
        tk.Entry(self.frame_conf, textvariable=self.X_bins, justify='center').grid(row=4, column=5)
        tk.Label(self.frame_conf, text='Xscale: ').grid(row=4, column=6)
        tk.OptionMenu(self.frame_conf, self.X_scale, *['log', 'linear']).grid(row=4, column=7)
        tk.Label(self.frame_conf, text='Colorbar: ').grid(row=5, column=0)
        self.colorbar_conf = tk.Text(self.frame_conf, height=3, wrap='none')
        self.colorbar_conf.grid(row=5, column=1, columnspan=5, sticky='w')
        self.colorbar_conf.insert(
            '0.0', '{"red":   [[0, 1, 1], [0.05, 0, 0], [0.1, 0, 0], [0.15, 1, 1], [0.3, 1, 1], [1, 1, 1]],\n "green": [[0, 1, 1], [0.05, 0, 0], [0.1, 1, 1], [0.15, 1, 1], [0.3, 0, 0], [1, 0, 0]],\n "blue":  [[0, 1, 1], [0.05, 1, 1], [0.1, 0, 0], [0.15, 0, 0], [0.3, 0, 0], [1, 1, 1]]}')
        self.run_button = tk.Button(self.frame_conf, text='Run', bg='lime', command=self.run)
        self.run_button.grid(row=5, column=6, padx=10)
        tk.Label(self.frame_stat, text='#Traces: ').pack(side='left')
        self.stat_traces = tk.Label(self.frame_stat, text=0)
        self.stat_traces.pack(side='left')
        tk.Label(self.frame_stat, text='File: ', padx=20).pack(side='left')
        self.stat_lastfile = tk.Label(self.frame_stat)
        self.stat_lastfile.pack(side='left')
        tk.mainloop()

    def run(self):
        if not self.isrun:
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
            self.isrun = True
        else:
            self.run_button.config(text='Run', bg='lime')
            self.isrun = False

    def on_created(self, event):
        if self.isrun is False:
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
        extracted = STM_bj.extract_data(path, **self.extract_config)
        self.extract_data = np.vstack([self.extract_data, extracted])
        self.hist_G.add_data(extracted)
        self.hist_GS.add_data(extracted)
        self.canvas_G.draw()
        self.canvas_GS.draw()
        self.stat_traces.config(text=self.hist_G.trace)

    def export(self):
        Export_prompt(self.extract_data, self.hist_G, self.hist_GS, **self.extract_config)


class Export_prompt:

    def __init__(self, G: np.ndarray, hist_G: STM_bj.Hist_G, hist_GS: STM_bj.Hist_GS, **conf) -> None:
        self.window = tk.Toplevel()
        self.window.grab_set()
        self.window.title('Export')
        self.G = G
        self.hist_G = hist_G
        self.hist_GS = hist_GS
        self.conf = conf

        self.export_type = tk.IntVar(value=1)
        tk.Radiobutton(self.window, text='Raw data', variable=self.export_type, value=1).grid(row=0, column=0, sticky='w')
        tk.Radiobutton(self.window, text='1D histogram', variable=self.export_type, value=2).grid(row=1, column=0, sticky='w')
        tk.Radiobutton(self.window, text='2D histogram', variable=self.export_type, value=3).grid(row=2, column=0, sticky='w')

        self.check_raw_X = tk.BooleanVar(value=True)  #disabled
        self.check_raw_G = tk.BooleanVar(value=True)
        self.check_raw_logG = tk.BooleanVar(value=True)
        tk.Checkbutton(self.window, variable=self.check_raw_X, text='X', state='disabled').grid(row=0, column=1, sticky='w')
        tk.Checkbutton(self.window, variable=self.check_raw_G, text='G').grid(row=0, column=2, sticky='w')
        tk.Checkbutton(self.window, variable=self.check_raw_logG, text='logG').grid(row=0, column=3, sticky='w')

        self.check_1D_G = tk.BooleanVar(value=True)  #disabled
        self.check_1D_count = tk.BooleanVar(value=True)  #disabled
        self.check_1D_pertraces = tk.BooleanVar(value=False)
        tk.Checkbutton(self.window, variable=self.check_1D_G, text='G', state='disabled').grid(row=1, column=1, sticky='w')
        tk.Checkbutton(self.window, variable=self.check_1D_count, text='Count', state='disabled').grid(row=1, column=2, sticky='w')
        tk.Checkbutton(self.window, variable=self.check_1D_pertraces, text='per traces').grid(row=1, column=3, sticky='w')

        self.check_2D_axis = tk.BooleanVar(value=False)
        self.check_2D_count = tk.BooleanVar(value=True)  #disabled
        self.check_2D_pertraces = tk.BooleanVar(value=False)
        tk.Checkbutton(self.window, variable=self.check_2D_axis, text='Axis').grid(row=2, column=1, sticky='w')
        tk.Checkbutton(self.window, variable=self.check_2D_count, text='Count', state='disabled').grid(row=2, column=2, sticky='w')
        tk.Checkbutton(self.window, variable=self.check_2D_pertraces, text='per traces').grid(row=2, column=3, sticky='w')

        tk.Button(self.window, text='Export', bg='lime', command=self.run).grid(row=3, columnspan=4)

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
                    G = np.log10(np.abs(self.hist_G.x_bins)) if self.conf['G_scale'] == 'log' else self.hist_G.x_bins
                    G = (G[1:] + G[:-1]) / 2
                    count = self.hist_G.height_per_trace if self.check_1D_pertraces.get() else self.hist_G.height
                    np.savetxt(path, np.vstack([G, count]).T, delimiter=',')
                case 3:
                    count = self.hist_GS.height_per_trace.T if self.check_2D_pertraces.get() else self.hist_GS.height.T
                    if self.check_2D_axis.get():
                        df = pd.DataFrame(count)
                        X = np.log10(np.abs(self.hist_GS.x_bins)) if self.conf['X_scale'] == 'log' else self.hist_GS.x_bins
                        G = np.log10(np.abs(self.hist_GS.y_bins)) if self.conf['G_scale'] == 'log' else self.hist_GS.y_bins
                        df.columns = (X[1:] + X[:-1]) / 2
                        df.index = (G[1:] + G[:-1]) / 2
                        df.to_csv(path, sep=',')
                    else:
                        np.savetxt(path, count, delimiter=",")


if __name__ == '__main__':
    multiprocessing.freeze_support()  # PyInstaller
    matplotlib.use('TkAgg')
    GUI = STM_bj_GUI()
    GUI.start()
