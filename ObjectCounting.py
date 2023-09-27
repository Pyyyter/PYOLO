import cv2
import argparse

from ultralytics import YOLO
import supervision as sv
import numpy as np

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Py(ter)olo")
    parser.add_argument(
        "--webcam-resolution", 
        default=[640, 480], 
        nargs=2, 
        type=int
    )
    parser.add_argument(
        "--count-objects", 
        default=0, 
        nargs=1, 
        type=int
    )
    args = parser.parse_args()
    return args

def filterResults(results):
    persons = []
    boats = []
    for result in results:
        if result.names[0] == 'person':
            persons.append(result)
        elif result.names[0] == 'boat':
            boats.append(result)
    areaOfPersons = []
    for person in persons:
        areaOfPersons.append(person.xyxy[2] * person.xyxy[3])
    areaOfBoats = []
    for boat in boats:
        areaOfBoats.append(boat.xyxy[2] * boat.xyxy[3])

    return areaOfPersons, areaOfBoats

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("yolov8s.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    if args.count_objects:
        # The area that will be counted ( this is a square in the middle of the screen )
        ZONE_POLYGON = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ])
        zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
        zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=tuple(args.webcam_resolution))
        zone_annotator = sv.PolygonZoneAnnotator(
            zone=zone, 
            color=sv.Color.red(),
            thickness=2,
            text_thickness=4,
            text_scale=2
        )
    
    decisionMakerCount = 0
    
    while True:
        ret, frame = cap.read()

        results = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(results)
        try:
            labels = [
                f"{model.model.names[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, _,_,_
                in detections
            ]
        except:
            print(detections[0])
            labels = []
        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections, 
            labels=labels
        )

        if decisionMakerCount >= 30:
            decisionMakerCount = 0
            areaOfPersons, areaOfBoats = filterResults(results)
            

        if args.count_objects:
            zone.trigger(detections=detections)
            frame = zone_annotator.annotate(scene=frame)      

        cv2.imshow("yolov8", frame)

        if (cv2.waitKey(30) == 27):
            break

        decisionMakerCount += 1

if __name__ == "__main__":
    main()