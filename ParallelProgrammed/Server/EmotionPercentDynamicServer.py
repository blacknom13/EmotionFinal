from ParallelProgrammed import FaceRecognition
import ParallelProgrammed.HelperLibrary as help
import cv2 as cv
import numpy as np
import time

face_detection = None
capture = None
emotion_model = None
emotion_dict = None
emotion_dict_fullname = None
func = None
name = None
emotion_array = None
emotion_colors = None
age_list = ['25-30', '42-48', '6-20', '60-98']
DEFAULT_TIMER = 20  # In seconds


def send_face_to_specialist(client_id, face_data):
    # should send the data to the specialist that will be working with our client
    pass


def client_should_be_directed_by_age(client_age):
    # checks if the client should be sent directly to the specialist due to being an elder
    if age_list.index(client_age) > 2:
        return True
    else:
        return False


def add_new_client_to_list(face_data, client_ids, starting_emotions, client_age, clients_dict):
    if client_should_be_directed_by_age(client_age):
        direct_to(age=True)
        send_face_to_specialist(client_ids, face_data)
    else:
        local_total_frames = count_num_of_frames(starting_emotions)
        FaceRecognition.store_face_name_with_encoding(face_data, client_ids)
        clients_dict[client_ids] = [starting_emotions.copy(), local_total_frames, DEFAULT_TIMER,
                                    time.time_ns()]
    print(clients_dict)


def clear_sent_data(face_data, client_ids, starting_emotions, client_age):
    face_data[:] = []
    client_ids[:] = []
    starting_emotions[:] = []
    client_age[:] = []


def new_client_entered(list_of_sent_faces):
    return len(list_of_sent_faces) != 0


def update_client_state(clients_dict, client_id, anger_percent, sad_percent):
    clients_dict[client_id][2] -= func.fx((anger_percent + sad_percent) / 100.0)
    if clients_dict[client_id][2] > 20:
        clients_dict[client_id][2] = 20
    elif clients_dict[client_id][2] <= 0:
        real_time = time.time_ns() - clients_dict[client_id][3]
        real_time /= 1000000000
        print(direct_to(real_time=real_time))
        send_face_to_specialist(client_id, FaceRecognition.delete_face_by_name(client_id))


def direct_to(real_time=None, age=False):
    if 6 <= real_time < 10:
        destination = "психлогу"
    elif real_time >= 10 or age:
        destination = "старшему сотруднику"
    return destination


def count_num_of_frames(emotion_list):
    return sum(emotion_list)


def detect_emotions(ui, client_ids, face_data, starting_emotions, client_age):
    current = time.time_ns()
    id_to_info = {}
    fps = 0
    FPS = 0
    ui.configure_figure(emotion_array, emotion_colors)

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        faces = face_detection.process(img)

        if faces.detections and len(id_to_info) != 0:

            for idd, detection in enumerate(faces.detections):
                temp = detection.location_data.relative_bounding_box

                intx, inty, intWidth, intHeight = help.extract_face_boundaries(frame, temp)
                roi, roi_resized = help.extract_rois(frame, intx, inty, intWidth, intHeight)

                recognized_client = FaceRecognition.recognize_face(roi_resized)

                if recognized_client != 'No Face' and recognized_client is not None:
                    cropped_img = np.expand_dims(np.expand_dims(cv.resize(roi, (48, 48)), -1), 0)
                    prediction = emotion_model.predict(cropped_img)[0]
                    anger_percent, happy_percent, neutral_percent, sad_percent, id_to_info[recognized_client][
                        1] = help.update_client_emotions(prediction,
                                                         id_to_info[recognized_client][0],
                                                         id_to_info[recognized_client][1])
                    label_position = (intx + 20, inty - 20)

                    # Predict the emotions

                    cv.rectangle(frame, (intx, inty),
                                 (intx + intWidth, inty + intHeight),
                                 (0, 255, 255), 1)

                    # Fancy frame
                    help.draw_rect_frame(frame, intx, inty, intWidth, intHeight, 8, 20, (0, 255, 0))

                    help.resize_gray_rect_around_face(frame, intx, inty, intWidth, intHeight)

                    cv.putText(frame,
                               str(format("%.2f" % id_to_info[recognized_client][2])) + " sec",
                               (intx + int(intWidth / 2) - 50, label_position[1] + 10), cv.FONT_HERSHEY_COMPLEX_SMALL,
                               1,
                               (0, 0, 175), 1)

                    cv.putText(frame, recognized_client,
                               (intx - 120, inty + intHeight // 4), cv.FONT_HERSHEY_COMPLEX_SMALL, 3,
                               (0, 0, 0), 4)

                    help.print_emotion_percents(frame, emotion_array,
                                                [anger_percent, happy_percent, neutral_percent, sad_percent],
                                                emotion_colors, intx, inty, intHeight, 20)

                    if (time.time_ns() - current) / 1000000000 > 1:
                        update_client_state(id_to_info, recognized_client, anger_percent, sad_percent)

        if (time.time_ns() - current) / 1000000000 > 1:
            current = time.time_ns()
            FPS = fps
            fps = 0
        fps += 1

        if new_client_entered(face_data[:]):
            add_new_client_to_list(face_data[:][0], client_ids[:][0], starting_emotions[:][0], client_age[:][0],
                                   id_to_info)
            clear_sent_data(face_data, client_ids, starting_emotions, client_age)

        cv.rectangle(frame, (0, 0), (100, 40), (200, 200, 200), 40)
        cv.putText(frame, "FPS: " + str(FPS), (10, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))
        ui.update_camera(frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
