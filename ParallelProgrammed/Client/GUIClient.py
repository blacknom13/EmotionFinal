import multiprocessing
from multiprocessing.connection import Client
import EmotionPercentDynamicClient
import ParallelProgrammed.UserInterface
from keras.models import model_from_json
import mediapipe as mp
import cv2 as cv
import PySimpleGUI as sg
import numpy as np

from ParallelProgrammed import UserInterface

HOST = '192.168.0.14'  # "192.168.43.180"
PORT = 11111


def start_camera():
    return cv.VideoCapture(0), UserInterface.BetterUI(False)


def send_data(face_id, client_emotions, data, age):
    sock = Client((HOST, PORT))
    sock.send([face_id, client_emotions, data, age])
    sock.close()


def camera_capture(button_pressed, client_id, camera_ready):
    local_capture, local_ui = start_camera()

    emotion_dict_fullname_eng = {0: 'Angry', 1: "Happy", 2: "Neutral", 3: "Sad"}
    emotion_array_rus = ['Гнев', "Радость", "Нейтральность", "Грусть"]
    emotion_array_eng = ['Angry', 'Happy', 'Neutral', 'Sad']
    client_profile = [0, 0, 0, 0]
    emotion_colors = [[0, 0, 175], [0, 200, 0], [75, 75, 75], [175, 0, 0]]

    age_list = ['25-30', '42-48', '6-20', '60-98']

    INITIAL_CLIENT_STATE_TIMER = 6000000000  # In nanoseconds

    # Initializing face detector
    face_detector = mp.solutions.face_detection
    face_detection = face_detector.FaceDetection(.95)

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

    ## Loading Age model
    # Load our model json
    print("Loading Age Model")
    json_file = open('../AgeModelJson.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    age_model = model_from_json(loaded_model_json)

    # Load weights
    age_model.load_weights("../AgeModelWeights.h5")
    print("Age Model Loaded")

    init_timer = INITIAL_CLIENT_STATE_TIMER

    camera_ready.acquire()
    camera_ready.notify()
    camera_ready.release()
    emotion_model.predict(np.zeros((1, 48, 48, 1)))

    while True:
        ret, frame = local_capture.read()
        if not ret:
            break
        local_ui.update_camera(frame)
        if button_pressed.value == 1:
            button_pressed.value = 0
            face_data, age = EmotionPercentDynamicClient.detect_emotions(init_timer, local_ui, 0,
                                                                         client_profile,
                                                                         emotion_colors,
                                                                         local_capture,
                                                                         face_detection,
                                                                         emotion_model,
                                                                         age_model, age_list,
                                                                         client_id.value.decode(),
                                                                         emotion_array=emotion_array_eng)

            send_data(client_id.value.decode(), client_profile, face_data, age)
            client_profile = [0, 0, 0, 0]


if __name__ == "__main__":
    # setting up two threads:
    # the main one runs the 3 buttons interface for giving tickets
    # the second runs the recognition operations and sending the data to the server

    # the recognition thread
    button_pressed = multiprocessing.Value('i', 0)
    client_id = multiprocessing.Array('c', b"0000")
    camera_ready = multiprocessing.Condition()
    thread = multiprocessing.Process(target=camera_capture, args=(button_pressed, client_id, camera_ready,))
    thread.start()

    ###########

    # the main thread
    layout = [
        [sg.Button("Оформить карту", size=20, key="A", metadata=0)],
        [sg.Button("Оплата услуг", size=20, key="B", metadata=0)],
        [sg.Button("Оформить кредит", size=20, key="C", metadata=0)]
    ]

    sg.theme("LightGreen")
    x = {}

    with camera_ready:
        camera_ready.wait()
        window = sg.Window("Банковский интерфейс", layout, (100, 50))

    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED:
            break
        else:
            temp_id = str(window[event].key) + str(window[event].metadata)
            client_id.value = bytes(temp_id, 'UTF-8')
            window[event].metadata = (window[event].metadata + 1) % 10
            button_pressed.value = 1

    window.close()
