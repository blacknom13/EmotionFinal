import cv2 as cv
import numpy as np
import copy


def draw_rect_frame(frame, x, y, width, height, thickness, length, color):
    cv.line(frame, (x, y), (x + length, y), color, thickness)
    cv.line(frame, (x + width, y), (x + width - length, y), color, thickness)

    cv.line(frame, (x + width, y), (x + width, y + length), color, thickness)
    cv.line(frame, (x + width, y + height), (x + width, y + height - length), color, thickness)

    cv.line(frame, (x + width, y + height), (x + width - length, y + height), color, thickness)
    cv.line(frame, (x, y + height), (x + length, y + height), color, thickness)

    cv.line(frame, (x, y + height), (x, y + height - length), color, thickness)
    cv.line(frame, (x, y), (x, y + length), color, thickness)


def resize_gray_rect_around_face(frame, x, y, width, height):
    sub_frame = frame[max(0, y - 40): max(0, y + height),
                max(0, x - 150): max(0, x + width)]
    white_rect = np.ones(sub_frame.shape, dtype=np.uint8) * 255
    face_box_face = copy.deepcopy(frame[y:y + height, x:x + width])
    sub_frame = cv.addWeighted(sub_frame, 0.5, white_rect, 0.5, 1.0)

    if sub_frame is not None:
        frame[max(0, y - 40): max(0, y + height),
        max(0, x - 150): max(0, x + width)] = sub_frame
        frame[y:y + height, x:x + width] = face_box_face


def print_emotion_percents(frame, emotion_names, emotion_values, emotion_colors, x, y, height, offset):
    # Putting the labels of emotions on the left

    # emotion names are Angry, Happy, Neutral, Sad in that order
    # emotion values are the corresponding for the emotions above in the same order
    # offset is the distance between two adjacent emotion texts in the frame
    for i in range(len(emotion_names)):
        cv.putText(frame, emotion_names[i] + ":" + str(emotion_values[i]) + '%',
                   (x - 150, y + int(height / 2) + offset * i),
                   cv.FONT_HERSHEY_COMPLEX_SMALL, 1, emotion_colors[i], 1)


def update_client_emotions(prediction, emotion_list, frame_counter):
    # updates client's emotions
    emotion_list[prediction.argmax()] += 1
    frame_counter += 1
    local_total_frames = frame_counter
    anger_percent, happy_percent, neutral_percent, sad_percent = [int(x * 100 / local_total_frames) for x in
                                                                  emotion_list]
    return anger_percent, happy_percent, neutral_percent, sad_percent, frame_counter


def extract_face_boundaries(frame, face_box):
    # gets the face coordinates and size
    x = max(0, int(len(frame[0]) * face_box.xmin))
    y = max(0, int(len(frame) * face_box.ymin))
    y -= int(.1 * y)
    width = int(len(frame[0]) * face_box.width)
    height = int(len(frame) * face_box.height)
    height += int(.1 * height)
    return x, y, width, height


def extract_rois(frame, x, y, width, height):
    # extracts and resizes regions of interest
    roi = frame[y:y + height, x:x + width]
    roi = cv.resize(roi, (48, 48), interpolation=cv.INTER_AREA)
    roi_for_recognition = cv.resize(roi, (80, 80), interpolation=cv.INTER_AREA)
    roi = cv.cvtColor(roi, cv.COLOR_RGB2GRAY)
    return roi, roi_for_recognition
