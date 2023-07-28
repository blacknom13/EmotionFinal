import copy
import cv2 as cv
import ParallelProgrammed.FaceRecognition as fr
import ParallelProgrammed.HelperLibrary as help
import numpy as np
import time

def detect_emotions(preparation_timer, ui, local_timer, local_client_profile, emotion_dict, emotion_dict_fullname,
                    emotion_color, capture, face_detection, emotion_model, age_model=None, age_labels=None, name=None,
                    emotion_array=None, emotion_colors=None):
    current = time.time_ns()
    fps = 0
    FPS = 0
    start_minute = False
    once = False
    face_stored = False
    end_minute = current + preparation_timer
    local_total_frames = 0
    recognized_client = ""
    ui.configure_figure(emotion_array, emotion_colors)
    while True:
        # find haar cascade to draw bounding box around face
        ret, frame = capture.read()
        if not ret:
            break

        predicted_age = []
        img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        faces = face_detection.process(img)

        if faces.detections:
            intx, inty, intWidth, intHeight = 0, 0, 0, 0
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

                if face_stored:
                    recognized_client = fr.recognize_face(roi_resized)
                else:
                    fr.store_current_face(roi_resized, name)
                    face_stored = True

                cropped_img = np.expand_dims(np.expand_dims(cv.resize(roi, (48, 48)), -1), 0)
                prediction = emotion_model.predict(cropped_img)[0]

                if age_model is not None:
                    roi_resized = roi_resized.astype('float') / 255.0
                    cropped_img = np.expand_dims(roi_resized, axis=0)
                    predicted = age_model.predict(cropped_img)[0]
                    predicted_age = age_labels[predicted.argmax()]

                local_client_profile[prediction.argmax()] += 1
                local_total_frames += 1
                anger_percent = int(local_client_profile[0] * 100 / local_total_frames)
                happy_percent = int(local_client_profile[1] * 100 / local_total_frames)
                neutral_percent = int(local_client_profile[2] * 100 / local_total_frames)
                sad_percent = int(local_client_profile[3] * 100 / local_total_frames)

                label_position = (intx + 20, inty - 20)

                # Predict the emotions

                cv.rectangle(frame, (intx, inty),
                             (intx + intWidth, inty + intHeight),
                             (0, 255, 255), 1)

                # Fancy frame
                help.draw_rect_frame(frame, intx, inty, intWidth, intHeight, 8, 20, (0, 255, 0))

            if (time.time_ns() - current) / 1000000000 > 1:
                current = time.time_ns()
                FPS = fps
                fps = 0
                if not once and not start_minute:
                    start_minute = True
                ui.update_emotion_graph(client_profile=local_client_profile)
            fps += 1

            x = time.time_ns()

            if end_minute < x:
                once = True
                start_minute = False

            # if once:
            sub_frame = frame[max(0, inty - 40): max(0, inty + intHeight),
                        max(0, intx - 150): max(0, intx + intWidth)]
            white_rect = np.ones(sub_frame.shape, dtype=np.uint8) * 255
            temp_face = copy.deepcopy(frame[inty:inty + intHeight, intx:intx + intWidth])
            sub_frame = cv.addWeighted(sub_frame, 0.5, white_rect, 0.5, 1.0)
            if sub_frame is not None:
                frame[max(0, inty - 40): max(0, inty + intHeight),
                max(0, intx - 150): max(0, intx + intWidth)] = sub_frame
                frame[inty:inty + intHeight, intx:intx + intWidth] = temp_face

            # cv.putText(frame, emotion_dict_fullname[np.argmax(local_client_profile)],
            #            (intx + int(intWidth / 2) - 50, label_position[1] + 10), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
            #            emotion_color[np.argmax(local_client_profile)], 1)

            cv.putText(frame, recognized_client,
                       (intx - 120, inty + intHeight // 4), cv.FONT_HERSHEY_COMPLEX_SMALL, 4,
                       (0, 0, 0), 4)

            # Putting the labels of emotions on the left
            cv.putText(frame, emotion_dict[0] + ":" + str(anger_percent) + '%',
                       (intx - 150, inty + int(intHeight / 2)),
                       cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 175), 1)
            cv.putText(frame, emotion_dict[1] + ":" + str(happy_percent) + '%',
                       (intx - 150, inty + int(intHeight / 2) + 20), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 200, 0),
                       1)
            cv.putText(frame, emotion_dict[2] + ":" + str(neutral_percent) + '%',
                       (intx - 150, inty + int(intHeight / 2) + 40), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
            cv.putText(frame, emotion_dict[3] + ":" + str(sad_percent) + '%',
                       (intx - 150, inty + int(intHeight / 2) + 60), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (175, 0, 0),
                       1)

        cv.rectangle(frame, (0, 0), (100, 40), (200, 200, 200), 40)
        cv.putText(frame, "FPS: " + str(FPS), (10, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))

        if len(predicted_age) != 0:
            cv.putText(frame, predicted_age,
                       (intx + int(intWidth / 2) - 20, label_position[1] + 10), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       (0, 0, 255), 1)

        ui.update_camera(frame)
        if cv.waitKey(1) & 0xFF == ord('q') or local_timer <= 0 and once:
            break

    face_data=fr.return_face_by_name(name).copy()
    fr.delete_face_by_name(name)
    return face_data
