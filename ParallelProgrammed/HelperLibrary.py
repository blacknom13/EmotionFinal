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


def resize_gray_rect_around_face(frame, intx, inty, int_width, int_height):
    sub_frame = frame[max(0, inty - 40): max(0, inty + int_height),
                max(0, intx - 150): max(0, intx + int_width)]
    white_rect = np.ones(sub_frame.shape, dtype=np.uint8) * 255
    temp_face = copy.deepcopy(frame[inty:inty + int_height, intx:intx + int_width])
    sub_frame = cv.addWeighted(sub_frame, 0.5, white_rect, 0.5, 1.0)

    if sub_frame is not None:
        frame[max(0, inty - 40): max(0, inty + int_height),
        max(0, intx - 150): max(0, intx + int_width)] = sub_frame
        frame[inty:inty + int_height, intx:intx + int_width] = temp_face


def print_emotion_percents(frame, emotion_names, emotion_values, emotion_colors, intx, inty, int_height, offset):
    # Putting the labels of emotions on the left

    # emotion names are Angry, Happy, Neutral, Sad in that order
    # emotion values are the corresponding for the emotions above in the same order
    # offset is the distance between two adjacent emotion texts in the frame
    for i in range(len(emotion_names)):
        cv.putText(frame, emotion_names[i] + ":" + str(emotion_values[i]) + '%',
                   (intx - 150, inty + int(int_height / 2) + offset * i),
                   cv.FONT_HERSHEY_COMPLEX_SMALL, 1, emotion_colors[i], 1)


def update_client_emotions(prediction, emotion_list, frame_counter):
    # updates client's emotions
    emotion_list[prediction.argmax()] += 1
    frame_counter += 1
    local_total_frames = frame_counter
    anger_percent, happy_percent, neutral_percent, sad_percent = [int(x * 100 / local_total_frames) for x in
                                                                  emotion_list]
    return anger_percent, happy_percent, neutral_percent, sad_percent, frame_counter
