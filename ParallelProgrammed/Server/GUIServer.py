import EmotionPercentDynamicServer
from NewtonFunctionFile import NewtonFunction
from FullFunctionFile import FullFunction
from keras.models import model_from_json
from ParallelProgrammed import FaceRecognition
import mediapipe as mp
import cv2 as cv
import multiprocessing
from multiprocessing.connection import Listener
from multiprocessing.managers import BaseManager
import PySimpleGUI as sg
import NewGui2
import socket
import pickle
import numpy as np


def format_result_percent(first_profile, second_profile, index):
    res = int(second_profile[index] / sum(second_profile) * 100) - int(first_profile[index] / sum(first_profile) * 100)
    if res > 0:
        return "+ {}%".format(res)
    elif res < 0:
        return "- {}%".format(abs(res))
    else:
        return " {}%".format(abs(res))


def update_faces():
    return clients_ids, clients_faces


def start_camera():
    return cv.VideoCapture(0), NewGui2.BetterUI()


def camera_capture(ids, emotions, faces):
    local_capture, local_ui = start_camera()

    emotion_dict_fullname_eng = {0: 'Angry', 1: "Happy", 2: "Neutral", 3: "Sad"}
    emotion_array = ['Гнев', "Радость", "Нейтральность", "Грусть"]
    client_profile = [0, 0, 0, 0]
    emotion_color = [[0, 0, 175], [0, 200, 0], [75, 75, 75], [175, 0, 0]]
    emotion_color_rgb = [[0.684, 0, 0], [0, 0.781, 0], [0.293, 0.293, 0.293], [0, 0, 0.684]]

    age_list = ['25-30', '42-48', '6-20', '60-98']

    direct_to = ""

    timer = 20  # In seconds

    # time function definition
    funcs = [NewtonFunction("../functions/anger_p1.csv"), NewtonFunction("../functions/anger_p2.csv"),
             NewtonFunction("../functions/anger_p3.csv"), NewtonFunction("../functions/anger_p4.csv")]
    ranges = [[0, .2], [.2, .6], [.6, .9], [.9, 1.01]]

    func = FullFunction(funcs, ranges)

    # Initializing face detector
    face_detector = mp.solutions.face_detection
    face_detection = face_detector.FaceDetection(.75)

    # Loading Emotion Detection Model
    # load json and create model
    print("Loading Emotion Detection Model")
    json_file = open('../fer70.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    emotion_model = model_from_json(loaded_model_json)

    # load weights into new model
    emotion_model.load_weights('../fer70.h5')
    print("Emotion Detection Model Loaded")
    ###########

    EmotionPercentDynamicServer.emotion_dict = emotion_dict_fullname_eng
    EmotionPercentDynamicServer.emotion_dict_fullname = emotion_dict_fullname_eng
    EmotionPercentDynamicServer.emotion_color = emotion_color
    EmotionPercentDynamicServer.capture = local_capture
    EmotionPercentDynamicServer.face_detection = face_detection
    EmotionPercentDynamicServer.emotion_model = emotion_model
    EmotionPercentDynamicServer.func = func
    EmotionPercentDynamicServer.emotion_array = emotion_array
    EmotionPercentDynamicServer.emotion_colors = emotion_color_rgb

    direct_to, total_frames = EmotionPercentDynamicServer.detect_emotions(ui=local_ui, local_timer=timer,
                                                                          client_ids=ids, face_data=faces,
                                                                          starting_emotions=emotions)
    # ,
    #                                                                   emotion_dict=emotion_dict_fullname_eng,
    #                                                                   emotion_dict_fullname=emotion_dict_fullname_eng,
    #                                                                   emotion_color=emotion_color,
    #                                                                   capture=local_capture,
    #                                                                   face_detection=face_detection,
    #                                                                   emotion_model=emotion_model,
    #                                                                   func=func,
    #                                                                   emotion_array=emotion_array,
    #                                                                   emotion_colors=emotion_color_rgb,)


if __name__ == "__main__":
    # camera_capture([0, 0, 0, 0])
    manager = multiprocessing.Manager()
    clients_faces = manager.list()
    clients_ids = manager.list()
    clients_start_emotions = manager.list()
    TRESS = multiprocessing.Process(target=camera_capture, args=[clients_ids, clients_start_emotions, clients_faces])
    TRESS.start()

    HOST = '192.168.43.180'
    PORT = 11111

    sock = Listener((HOST, PORT))
    # sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # sock.bind((HOST, PORT))
    # sock.listen(1)

    while True:
        conn = sock.accept()
        data = conn.recv()
        clients_ids.append(data[0])
        clients_start_emotions.append(data[1])
        clients_faces.append(data[2])
        # print(data)
        # conn, addr = sock.accept()
        # print("Connected by", addr)
        #
        # id = b''
        # emotions = b''
        # face = b''
        # while True:
        #     data = conn.recv(8)
        #     if not data:
        #         break
        #     id += data
        #
        # while True:
        #     data = conn.recv(32)
        #     if not data:
        #         break
        #     emotions += data
        #
        # while True:
        #     data = conn.recv(4096)
        #     if not data:
        #         break
        #     face += data
        #
        # received_array = pickle.loads(face)
        # print("Received array:", received_array)
        #
        # clients_faces.append(received_array)
        # # Process the received array as needed
        #
        # EmotionPercentDynamicServer.clients_emotion_array = np.array(list(clients_faces))
        # print(clients_faces)
        #
        # conn.close()
        # print("Connection closed\n")
