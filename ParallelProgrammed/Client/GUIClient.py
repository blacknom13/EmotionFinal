import multiprocessing
from multiprocessing.connection import Client
import EmotionPercentDynamicClient
import NewGui2
from keras.models import model_from_json
import mediapipe as mp
import cv2 as cv
import PySimpleGUI as sg
import pickle
import socket

HOST = "192.168.43.180"
PORT = 11111


def format_result_percent(first_profile, second_profile, index):
    res = int(second_profile[index] / sum(second_profile) * 100) - int(first_profile[index] / sum(first_profile) * 100)
    if res > 0:
        return "+ {}%".format(res)
    elif res < 0:
        return "- {}%".format(abs(res))
    else:
        return " {}%".format(abs(res))


def start_camera():
    return cv.VideoCapture(0), NewGui2.BetterUI()


def send_data(data, face_id, client_emotions):
    # sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # sock.connect((HOST, PORT))
    # print("Connected successfully to: {} on the port {}".format(HOST, PORT))
    # sock.send(face_id)
    # sock.send(data)
    # sock.send(client_emotions)
    sock=Client((HOST,PORT))
    sock.send([face_id,client_emotions,data])
    sock.close()


def camera_capture(button_pressed, client_id, camera_ready):
    local_capture, local_ui = start_camera()

    emotion_dict = {0: 'Agr', 1: "Hpy", 2: "Ntl", 3: "Sad"}
    emotion_dict_fullname_eng = {0: 'Angry', 1: "Happy", 2: "Neutral", 3: "Sad"}
    emotion_array_eng = ['Angry', "Happy", "Neutral", "Sad"]
    emotion_array = ['Гнев', "Радость", "Нейтральность", "Грусть"]
    client_profile = [0, 0, 0, 0]
    emotion_color = [[0, 0, 175], [0, 200, 0], [75, 75, 75], [175, 0, 0]]
    emotion_color_rgb = [[0.684, 0, 0], [0, 0.781, 0], [0.293, 0.293, 0.293], [0, 0, 0.684]]

    age_list = ['25-30', '42-48', '6-20', '60-98']

    direct_to = ""

    INITIAL_CLIENT_STATE_TIMER = 6000000000  # In nanoseconds

    # time function definition
    total_frames = 0

    # Initializing face detector
    face_detector = mp.solutions.face_detection
    face_detection = face_detector.FaceDetection(.75)

    # Loading Emotion Detection Model
    # load json and create model
    print("Loading Emotion Detection Model")
    json_file = open('fer70.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    emotion_model = model_from_json(loaded_model_json)

    # load weights into new model
    emotion_model.load_weights('fer70.h5')
    print("Emotion Detection Model Loaded")
    ###########

    ## Loading Age model
    # Load our model json
    print("Loading Age Model")
    json_file = open('AgeModelJson.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    age_model = model_from_json(loaded_model_json)

    # Load weights
    age_model.load_weights("AgeModelWeights.h5")
    print("Age Model Loaded")

    init_timer = INITIAL_CLIENT_STATE_TIMER

    camera_ready.acquire()
    camera_ready.notify()
    camera_ready.release()
    while True:
        ret, frame = local_capture.read()
        if not ret:
            break
        local_ui.update_camera(frame)
        if button_pressed.value == 1:
            button_pressed.value = 0
            direct_to, total_frames, face_data = EmotionPercentDynamicClient.detect_emotions(init_timer, local_ui, 0,
                                                                                           client_profile,
                                                                                           emotion_dict_fullname_eng,
                                                                                           emotion_dict_fullname_eng,
                                                                                           emotion_color,
                                                                                           local_capture,
                                                                                           face_detection,
                                                                                           emotion_model,
                                                                                           age_model, age_list,
                                                                                           client_id.value.decode(),
                                                                                           emotion_array=emotion_array,
                                                                                           emotion_colors=emotion_color_rgb)

            #data = pickle.dumps(face_data)
            #face_name=pickle.dumps(client_id.value.decode())
            #client_emotions=pickle.dumps(client_profile)
            #send_data(data, face_name, client_emotions)
            send_data(face_data, client_id.value.decode(), client_profile)

            client_profile = [0, 0, 0, 0]


if __name__ == "__main__":
    button_pressed = multiprocessing.Value('i', 0)
    client_id = multiprocessing.Array('c', b"0000")
    camera_ready = multiprocessing.Condition()
    thread = multiprocessing.Process(target=camera_capture, args=(button_pressed, client_id, camera_ready,))
    thread.start()

    ###########

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
