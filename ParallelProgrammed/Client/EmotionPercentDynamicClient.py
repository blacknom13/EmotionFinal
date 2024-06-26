import cv2 as cv
import ParallelProgrammed.FaceRecognition as fr
import ParallelProgrammed.HelperLibrary as help
import numpy as np
import time


def detect_emotions(preparation_timer, ui, local_timer, local_client_profile,
                    emotion_colors, capture, face_detection, emotion_model, age_model=None, age_labels=None, name=None,
                    emotion_array=None):
    current = time.time_ns()
    fps = 0
    FPS = 0
    start_minute = False
    once = False
    face_stored = False
    end_minute = current + preparation_timer
    local_total_frames = 0
    recognized_client = ""
    if not ui.graph_is_valid():
        ui.configure_figure(emotion_array, emotion_colors)

    while True:
        # find haar cascade to draw bounding box around face
        ret, frame = capture.read()
        if not ret:
            break

        predicted_age = [0, 0, 0, 0]
        img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        faces = face_detection.process(img)

        if faces.detections:
            intx, inty, intWidth, intHeight = 0, 0, 0, 0
            for idd, detection in enumerate(faces.detections):
                temp = detection.location_data.relative_bounding_box

                intx, inty, intWidth, intHeight = help.extract_face_boundaries(frame, temp)
                roi, roi_resized = help.extract_rois(frame, intx, inty, intWidth, intHeight)

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
                    predicted_age[predicted.argmax()] += 1

                anger_percent, happy_percent, neutral_percent, sad_percent, local_total_frames = (
                    help.update_client_emotions(prediction, local_client_profile, local_total_frames))

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

            help.resize_gray_rect_around_face(frame, intx, inty, intWidth, intHeight)

            cv.putText(frame, recognized_client,
                       (intx - 120, inty + intHeight // 4), cv.FONT_HERSHEY_COMPLEX_SMALL, 4,
                       (0, 0, 0), 4)

            help.print_emotion_percents(frame, emotion_array,
                                        [anger_percent, happy_percent, neutral_percent, sad_percent],
                                        emotion_colors, intx, inty, intHeight, 20)

            cv.rectangle(frame, (0, 0), (100, 40), (200, 200, 200), 40)
            cv.putText(frame, "FPS: " + str(FPS), (10, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))

            if len(predicted_age) != 0:
                cv.putText(frame, age_labels[predicted_age.index(max(predicted_age))],
                           (intx + int(intWidth / 2) - 20, label_position[1] + 10), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           (0, 0, 255), 1)

        ui.update_camera(frame)
        if cv.waitKey(1) & 0xFF == ord('q') or local_timer <= 0 and once:
            break

    face_data = fr.return_face_by_name(name).copy()
    fr.delete_face_by_name(name)
    print(age_labels[predicted_age.index(max(predicted_age))])
    return face_data, age_labels[predicted_age.index(max(predicted_age))]
