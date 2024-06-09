import supervision as sv
import numpy as np
from ultralytics import YOLO
import cv2
COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B"])

if __name__ == "__main__":

    # Load the model
    model = YOLO("./data/traffic_analysis.pt")

    # Open a connection to the webcam
    cap = cv2.VideoCapture('./vdo.avi')

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set window dimensions (Only works on my laptop's webcam??)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    # Area of Interest parameters
    # Polygon for area of detections
    polygon = np.array([
        [0, 200],    # A
        [1800, 200],  # B
        [1800, 900],  # C
        [0, 900]     # D
    ])

    # TEST LINE
    testline_start = sv.Point(974, 401)
    testline_end = sv.Point(1085, 763)
    testline_zone = sv.LineZone(start=testline_start, end=testline_end)
    testline_annotator = sv.LineZoneAnnotator()

    Polygon = sv.PolygonZone(polygon=polygon, frame_resolution_wh=[
                             frame_width, frame_height])

    # Initialize bounding boxes annotator
    Box_Annotator = sv.LabelAnnotator(
        color=COLORS, text_color=sv.Color.BLACK)

    # Define the window name
    window_name = 'Video file'
    tracker = sv.ByteTrack()
    while True:
        # Read a frame from the video file
        ret, frame = cap.read()

        # If the frame was successfully read
        if ret:
            result = model(
                frame, verbose=False, conf=0.3, iou=0.5
            )[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = tracker.update_with_detections(detections)
            xyxy = detections.xyxy
            mask = detections.mask
            confidence = detections.confidence
            class_id = detections.class_id
            tracker_id = detections.tracker_id
            labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
            print(detections)
            frame = Box_Annotator.annotate(
                frame, detections=detections, labels=labels)
            testline_annotator.annotate(frame, testline_zone)
            testline_zone.trigger(detections=detections)
            print(testline_zone.in_count, testline_zone.out_count)
            cv2.imshow(window_name, frame)

            # If the user presses 'q', or the window is closed, exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        else:
            print("Unable to read frame")
            break

    # Release the webcam and destroy all windows
    cap.release()
    cv2.destroyAllWindows()
