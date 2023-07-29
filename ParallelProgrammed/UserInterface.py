import PySimpleGUI as sg
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import cv2


class BetterUI:
    window = ""
    img_bytes = ""
    figure_canvas_agg = ""
    emotion_labels = ""
    emotion_colors = ""
    canvas = ""
    no_canvas = False

    def __init__(self, no_canvas):
        sg.theme("reds")
        self.no_canvas = no_canvas
        if no_canvas:
            layout = [
                [sg.Image(key='CAMERA')]
            ]
        else:
            layout = [
                [sg.Image(key='CAMERA'), sg.Canvas(size=(500, 400), key='GRAPH')]
            ]

        self.window = sg.Window('Camera', layout, finalize=True)

    def configure_figure(self, emotion_labels=None, emotion_colors=None):
        self.canvas = matplotlib.figure.Figure()
        self.canvas.add_subplot(111).plot([], [])

        if not self.no_canvas:
            self.emotion_labels = emotion_labels
            self.emotion_colors = self.bgr2rgb(emotion_colors)

            self.figure_canvas_agg = FigureCanvasTkAgg(self.canvas, self.window['GRAPH'].TKCanvas)
            self.figure_canvas_agg.draw()
            self.figure_canvas_agg.get_tk_widget().pack()

        self.move_center()

    def update_camera(self, frame):
        event, values = self.window.read(timeout=1)
        if event == sg.WIN_CLOSED:
            self.destroy_window()
        self.img_bytes = cv2.imencode('.png', frame)[1].tobytes()
        self.window['CAMERA'].update(data=self.img_bytes)

    def update_emotion_graph(self, client_profile):
        event, values = self.window.read(timeout=1)
        if event == sg.WIN_CLOSED:
            self.destroy_window()

        axes = self.canvas.axes
        axes[0].cla()

        x_pos = np.arange(len(self.emotion_labels))
        total_frames = sum(client_profile)
        client_profile = [(i / total_frames) * 100 for i in client_profile]

        ax = axes[0]
        ax.bar(x_pos, client_profile, color=self.emotion_colors, width=.4)
        ax.grid(axis="y")

        ax.set_ylim(top=110)
        ax.set_yticks([i * 10 for i in range(0, 11)])
        ax.set_xticks(x_pos, self.emotion_labels)

        self.figure_canvas_agg.draw()
        self.figure_canvas_agg.get_tk_widget().pack()

    def bgr2rgb(self, bgr_colors):
        rgb_colors = []
        for color in bgr_colors:
            rgb_colors.append([color[2]/256, color[1]/256, color[0]/256])
        return rgb_colors

    def move_center(self):
        screen_width, screen_height = self.window.get_screen_dimensions()
        win_width, win_height = self.window.size
        x, y = (screen_width - win_width) // 2, (screen_height - win_height) // 2
        self.window.move(x, y)

    def destroy_window(self):
        self.window.close()
