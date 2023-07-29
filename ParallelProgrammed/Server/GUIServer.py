import EmotionPercentDynamicServer
from ParallelProgrammed.NewtonFunctionFile import NewtonFunction
from ParallelProgrammed.FullFunctionFile import FullFunction
from keras.models import model_from_json
import mediapipe as mp
import cv2 as cv
import multiprocessing
from multiprocessing.connection import Listener
from ParallelProgrammed import UserInterface
import numpy as np

HOST = '192.168.0.14'  # '192.168.43.180'
PORT = 11111


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
    return cv.VideoCapture(0), UserInterface.BetterUI(True)


def camera_capture(ids, emotions, faces):
    local_capture, local_ui = start_camera()

    emotion_dict_fullname_eng = {0: 'Angry', 1: "Happy", 2: "Neutral", 3: "Sad"}
    emotion_array_rus = ['Гнев', "Радость", "Нейтральность", "Грусть"]
    emotion_array_eng = ['Angry', 'Happy', 'Neutral', 'Sad']
    emotion_colors = [[0, 0, 175], [0, 200, 0], [75, 75, 75], [175, 0, 0]]

    direct_to = ""

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
    EmotionPercentDynamicServer.emotion_colors = emotion_colors
    EmotionPercentDynamicServer.capture = local_capture
    EmotionPercentDynamicServer.face_detection = face_detection
    EmotionPercentDynamicServer.emotion_model = emotion_model
    EmotionPercentDynamicServer.func = func
    EmotionPercentDynamicServer.emotion_array = emotion_array_eng

    emotion_model.predict(np.zeros((1, 48, 48, 1)))
    EmotionPercentDynamicServer.detect_emotions(ui=local_ui,
                                                client_ids=ids, face_data=faces,
                                                starting_emotions=emotions,client_age=client_age)


if __name__ == "__main__":

    manager = multiprocessing.Manager()
    clients_faces = manager.list()
    clients_ids = manager.list()
    clients_start_emotions = manager.list()
    client_age=manager.list()
    TRESS = multiprocessing.Process(target=camera_capture, args=[clients_ids, clients_start_emotions, clients_faces])
    TRESS.start()

    sock = Listener((HOST, PORT))

    while True:
        conn = sock.accept()
        data = conn.recv()
        clients_ids.append(data[0])
        clients_start_emotions.append(data[1])
        clients_faces.append(data[2])
        client_age.append(data[3])
        # print (clients_faces[:])
