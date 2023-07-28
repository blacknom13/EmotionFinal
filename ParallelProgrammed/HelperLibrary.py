import cv2 as cv
def draw_rect_frame(frame, x, y, width, height, thickness, length, color):
    cv.line(frame, (x, y), (x + length, y), color, thickness)
    cv.line(frame, (x + width, y), (x + width - length, y), color, thickness)

    cv.line(frame, (x + width, y), (x + width, y + length), color, thickness)
    cv.line(frame, (x + width, y + height), (x + width, y + height - length), color, thickness)

    cv.line(frame, (x + width, y + height), (x + width - length, y + height), color, thickness)
    cv.line(frame, (x, y + height), (x + length, y + height), color, thickness)

    cv.line(frame, (x, y + height), (x, y + height - length), color, thickness)
    cv.line(frame, (x, y), (x, y + length), color, thickness)