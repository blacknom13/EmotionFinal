import matplotlib.pyplot as plt
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
DEFAULT_TIMER = 20  # In seconds


def update_client_emotions(img, recognized_client, clients_dict):
    prediction = emotion_model.predict(img)[0]

    clients_dict[recognized_client][0][prediction.argmax()] += 1
    clients_dict[recognized_client][1] += 1
    local_total_frames = clients_dict[recognized_client][1]
    anger_percent = int(
        clients_dict[recognized_client][0][0] * 100 / local_total_frames)
    happy_percent = int(
        clients_dict[recognized_client][0][1] * 100 / local_total_frames)
    neutral_percent = int(
        clients_dict[recognized_client][0][2] * 100 / local_total_frames)
    sad_percent = int(
        clients_dict[recognized_client][0][3] * 100 / local_total_frames)

    return anger_percent, happy_percent, neutral_percent, sad_percent


def add_new_client_to_list(face_data, client_ids, starting_emotions, clients_dict):
    local_total_frames = count_num_of_frames(starting_emotions)

    FaceRecognition.store_face_name_with_encoding(face_data, client_ids)

    clients_dict[client_ids] = [starting_emotions.copy(), local_total_frames, DEFAULT_TIMER,
                                time.time_ns()]
    print(clients_dict)


def clear_sent_data(face_data, client_ids, starting_emotions):
    face_data[:] = []
    client_ids[:] = []
    starting_emotions[:] = []


def new_client_entered(list_of_sent_faces):
    return len(list_of_sent_faces) != 0


def update_client_state(clients_dict, client_id, anger_percent, sad_percent):
    clients_dict[client_id][2] -= func.fx((anger_percent + sad_percent) / 100.0)
    if clients_dict[client_id][2] > 20:
        clients_dict[client_id][2] = 20
    elif clients_dict[client_id][2] <= 0:
        real_time = time.time_ns() - clients_dict[client_id][3]
        real_time /= 1000000000
        print(direct_to(real_time))


def direct_to(real_time):
    if 6 <= real_time < 10:
        destination = "психлогу"
    else:
        destination = "старшему сотруднику"
    return destination


def count_num_of_frames(emotion_list):
    return sum(emotion_list)


def detect_emotions(ui, client_ids, face_data, starting_emotions):
    current = time.time_ns()
    id_to_info = {}
    fps = 0
    FPS = 0
    ui.configure_figure(emotion_array, emotion_colors)

    while True:
        # find haar cascade to draw bounding box around face
        ret, frame = capture.read()
        if not ret:
            break

        img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        faces = face_detection.process(img)

        if faces.detections and len(id_to_info) != 0:

            for id, detection in enumerate(faces.detections):

                temp = detection.location_data.relative_bounding_box

                intx = max(0, int(len(frame[0]) * temp.xmin))
                inty = max(0, int(len(frame) * temp.ymin))
                inty -= int(.1 * inty)
                intWidth = int(len(frame[0]) * temp.width)
                intHeight = int(len(frame) * temp.height)
                intHeight += int(.1 * intHeight)

                roi = frame[inty:inty + intHeight, intx:intx + intWidth]
                roi = cv.resize(roi, (48, 48), interpolation=cv.INTER_AREA)
                roi_resized = cv.resize(roi, (80, 80), interpolation=cv.INTER_AREA)
                roi = cv.cvtColor(roi, cv.COLOR_RGB2GRAY)

                recognized_client = FaceRecognition.recognize_face(roi_resized)

                if recognized_client != 'No Face' and recognized_client is not None:
                    cropped_img = np.expand_dims(np.expand_dims(cv.resize(roi, (48, 48)), -1), 0)
                    anger_percent, happy_percent, neutral_percent, sad_percent = update_client_emotions(cropped_img,
                                                                                                        recognized_client,
                                                                                                        id_to_info)
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
            add_new_client_to_list(face_data[:][0], client_ids[:][0], starting_emotions[:][0], id_to_info)
            clear_sent_data(face_data, client_ids, starting_emotions)

        cv.rectangle(frame, (0, 0), (100, 40), (200, 200, 200), 40)
        cv.putText(frame, "FPS: " + str(FPS), (10, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))
        ui.update_camera(frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break


def make_bar_chart(figure_name, client_profile, compare_to_client_profile, total_frames, emotion_array,
                   emotion_color_rgb):
    y_pos = np.arange(len(emotion_array))
    color = np.array(emotion_color_rgb)
    client_profile = [(x / total_frames) * 100 for x in client_profile]
    if len(compare_to_client_profile) != 0:
        compare_to_client_profile = [(x / sum(compare_to_client_profile)) * 100 for x in compare_to_client_profile]

    width = .35
    fig, ax = plt.subplots()
    first_prof = ax.bar(y_pos - width / 2, client_profile, color=color, width=.3)

    if len(compare_to_client_profile) != 0:
        second_prof = ax.bar(y_pos + width / 2, compare_to_client_profile, color=color / 2, width=.3)
    ax.set_xticks(y_pos, emotion_array)
    ax.set_ylabel('Проценты')
    ax.set_title('Эмоции')

    if len(compare_to_client_profile) != 0:
        ax.bar_label(first_prof, labels=["После", "После", "После", "После"], padding=3)
        ax.bar_label(second_prof, labels=["До", "До", "До", "До"], padding=3)
    fig.tight_layout()
    plt.ylim(top=110)
    plt.yticks([i * 10 for i in range(0, 11)])
    plt.get_current_fig_manager().set_window_title(figure_name)
    plt.show()


def make_bar_chart_list(figure_name, client_profiles, profile_labels, emotion_array, emotion_color_rgb):
    y_pos = np.arange(len(emotion_array))
    color = np.array(emotion_color_rgb)
    num_of_profiles = len(client_profiles)
    for i in range(num_of_profiles):
        client_profiles[i] = [(x / sum(client_profiles[i])) * 100 for x in client_profiles[i]]

    width = 1 / (num_of_profiles + 1)
    fig, ax = plt.subplots()
    list_of_profs = []
    position = (-width / num_of_profiles) * (num_of_profiles / 2)
    for i in range(num_of_profiles):
        list_of_profs.append(
            ax.bar(y_pos + position + (width / num_of_profiles) * (i + 2 * i), client_profiles[i],
                   color=color / (i + 1),
                   width=1 / (num_of_profiles + 1)))

    ax.set_xticks(y_pos, emotion_array)
    ax.set_ylabel('Проценты')
    ax.set_title('Эмоции')

    for i in range(num_of_profiles):
        ax.bar_label(list_of_profs[i], labels=profile_labels[i], padding=3)

    fig.tight_layout()
    plt.ylim(top=110)
    plt.get_current_fig_manager().set_window_title(figure_name)
    plt.show()
