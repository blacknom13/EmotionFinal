import copy
import GUIServer
import matplotlib.pyplot as plt
from ParallelProgrammed import FaceRecognition
import cv2 as cv
import numpy as np
import time

face_detection = None
capture = None
emotion_color = None
emotion_model = None
emotion_dict = None
emotion_dict_fullname = None
func = None
name = None
emotion_array = None
emotion_colors = None
DEFAULT_TIMER = 20  # In seconds


def draw_rect_frame(frame, x, y, width, height, thickness, length, color):
    cv.line(frame, (x, y), (x + length, y), color, thickness)
    cv.line(frame, (x + width, y), (x + width - length, y), color, thickness)

    cv.line(frame, (x + width, y), (x + width, y + length), color, thickness)
    cv.line(frame, (x + width, y + height), (x + width, y + height - length), color, thickness)

    cv.line(frame, (x + width, y + height), (x + width - length, y + height), color, thickness)
    cv.line(frame, (x, y + height), (x + length, y + height), color, thickness)

    cv.line(frame, (x, y + height), (x, y + height - length), color, thickness)
    cv.line(frame, (x, y), (x, y + length), color, thickness)


def count_num_of_frames(emotion_list):
    return sum(emotion_list)


def detect_emotions(ui, client_ids, face_data, starting_emotions):
    local_timer = 20
    current = time.time_ns()
    direct_to = ""
    real_time = 0
    id_to_info = {}
    fps = 0
    FPS = 0
    ui.configure_figure(emotion_array, emotion_colors)
    anger_percent=0
    sad_percent=0

    while True:
        # find haar cascade to draw bounding box around face
        ret, frame = capture.read()
        if not ret:
            break

        recognized_client = ""
        # clients_emotion_array = starting_emotions[:]
        img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        faces = face_detection.process(img)

        if faces.detections and len(id_to_info) != 0:

            for id, detection in enumerate(faces.detections):
                print(id)

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
                if recognized_client != 'No Face' and recognized_client != '' and recognized_client is not None:
                    print(FaceRecognition.faces_names)

                    cropped_img = np.expand_dims(np.expand_dims(cv.resize(roi, (48, 48)), -1), 0)

                    prediction = emotion_model.predict(cropped_img)[0]

                    print(prediction)
                    # label = emotion_dict[prediction.argmax()]
                    id_to_info[recognized_client][0][prediction.argmax()] += 1
                    local_total_frames = count_num_of_frames(id_to_info[recognized_client][0])
                    anger_percent = int(
                        id_to_info[recognized_client][0][0] * 100 / local_total_frames)
                    happy_percent = int(
                        id_to_info[recognized_client][0][1] * 100 / local_total_frames)
                    neutral_percent = int(
                        id_to_info[recognized_client][0][2] * 100 / local_total_frames)
                    sad_percent = int(
                        id_to_info[recognized_client][0][3] * 100 / local_total_frames)

                    label_position = (intx + 20, inty - 20)

                    # Predict the emotions

                    cv.rectangle(frame, (intx, inty),
                                 (intx + intWidth, inty + intHeight),
                                 (0, 255, 255), 1)

                    # Fancy frame
                    draw_rect_frame(frame, intx, inty, intWidth, intHeight, 8, 20, (0, 255, 0))

                    cv.rectangle(frame, (0, 0), (640, 40), (200, 200, 200), 40)
                    cv.putText(frame, "FPS: " + str(FPS), (20, 40), cv.FONT_HERSHEY_TRIPLEX, 1.2, (0, 0, 255))
                    cv.putText(frame, "Timer: " + str(format("%.2f" % local_timer)), (300, 40), cv.FONT_HERSHEY_TRIPLEX,
                               1.2,
                               (255, 0, 255))

                    sub_frame = frame[max(0, inty - 40): max(0, inty + intHeight),
                                max(0, intx - 150): max(0, intx + intWidth)]
                    white_rect = np.ones(sub_frame.shape, dtype=np.uint8) * 255
                    temp_face = copy.deepcopy(frame[inty:inty + intHeight, intx:intx + intWidth])
                    sub_frame = cv.addWeighted(sub_frame, 0.5, white_rect, 0.5, 1.0)
                    if sub_frame is not None:
                        frame[max(0, inty - 40): max(0, inty + intHeight),
                        max(0, intx - 150): max(0, intx + intWidth)] = sub_frame
                        frame[inty:inty + intHeight, intx:intx + intWidth] = temp_face

                    cv.putText(frame,
                               emotion_dict_fullname[np.argmax(id_to_info[recognized_client][0][prediction.argmax()])],
                               (intx + int(intWidth / 2) - 50, label_position[1] + 10), cv.FONT_HERSHEY_COMPLEX_SMALL,
                               1,
                               emotion_color[np.argmax(id_to_info[recognized_client][0][prediction.argmax()])], 1)

                    cv.putText(frame, recognized_client,
                               (intx - 120, inty + intHeight // 4), cv.FONT_HERSHEY_COMPLEX_SMALL, 4,
                               (0, 0, 0), 4)

                    # Putting the labels of emotions on the left
                    cv.putText(frame, emotion_dict[0] + ":" + str(anger_percent) + '%',
                               (intx - 150, inty + int(intHeight / 2)),
                               cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 175), 1)
                    cv.putText(frame, emotion_dict[1] + ":" + str(happy_percent) + '%',
                               (intx - 150, inty + int(intHeight / 2) + 20), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                               (0, 200, 0),
                               1)
                    cv.putText(frame, emotion_dict[2] + ":" + str(neutral_percent) + '%',
                               (intx - 150, inty + int(intHeight / 2) + 40), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                               (0, 0, 0), 1)
                    cv.putText(frame, emotion_dict[3] + ":" + str(sad_percent) + '%',
                               (intx - 150, inty + int(intHeight / 2) + 60), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                               (175, 0, 0),
                               1)

                    cv.putText(frame, recognized_client,
                               (intx + int(intWidth / 2) - 20, label_position[1] + 10), cv.FONT_HERSHEY_COMPLEX_SMALL,
                               1,
                               # (intx - 75, inty), cv.FONT_HERSHEY_COMPLEX_SMALL, 4,
                               emotion_color[np.argmax(id_to_info[recognized_client][0][prediction.argmax()])], 1)
                # cv.imshow('Emotion Detector', frame)
            if (time.time_ns() - current) / 1000000000 > 1:
                current = time.time_ns()
                FPS = fps
                fps = 0
                local_timer -= func.fx((anger_percent + sad_percent) / 100.0)
                if local_timer > 20:
                    local_timer = 20
                real_time += 1

            fps += 1

            x = time.time_ns()

        if len(face_data[:]) != 0:
            local_face_data = face_data[:]
            local_client_ids = client_ids[:][0]
            FaceRecognition.store_face_name_with_encoding(local_face_data, local_client_ids)
            id_to_info[local_client_ids] = [starting_emotions[:][0].copy(), DEFAULT_TIMER, time.time_ns()]
            face_data = []
            client_ids = []
            starting_emotions = []
            print(id_to_info)

        # for t in local_timer:

        ui.update_camera(frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            if real_time >= 6 and real_time < 10:
                direct_to = "психлогу"
            else:  # elif real_time >= 10 and real_time < 14:
                direct_to = "старшему сотруднику"
            # else:
            #     direct_to = "обычному сотруднику"
            break
    return local_total_frames, direct_to


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
