import cv2

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    ### predict output
    # draw rectangle around face
    x = 313
    y = 82
    w = 451
    h = 568
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 3)
    width = int(video_capture.get(3) + 0.5)
    height = int(video_capture.get(4) + 0.5)
    # annotate main image with a label
    cv2.putText(frame, text="A", org=(width//2 + 250, height//2 + 75),
                fontFace= cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=17, color=(255, 255, 0),
                thickness=15, lineType=cv2.LINE_AA)

    cv2.putText(frame, text="B", org=(width//2 + width//5+40, (360 + 240)),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=6, color=(0, 0, 255),
                thickness=6, lineType=cv2.LINE_AA)

    cv2.putText(frame, text="C", org=(width//2 + width//3+5, (360 + 240)),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=6, color=(0, 0, 255),
                thickness=6, lineType=cv2.LINE_AA)


    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
