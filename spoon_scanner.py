import os
import cv2
import numpy as np
import argparse


def arg_parser():
    desc =  """
            An inspection system that scans baby food can in order to ensure
            that a single spoon has been placed in the can.
            """
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--image", dest="img_path", default=None, type=str,
                        help="path to image/images")
    parser.add_argument("--template", dest="templ_path", default=None, type=str,
                        help="path to template image")
    parser.add_argument("--metric", dest="metric", default=0, type=int,
                        help="Metric used by the algorithm. 0 -> squared difference,\
                        1 -> correlation")

    return parser.parse_args()

def nms(bboxes, thresh):
    if not len(bboxes): return []

    picked = [] #selected indices

    #sort bboxes according to their correlation score with the temlate
    idxs = np.argsort(bboxes[:, 4])
    bboxes = bboxes[idxs]

    #Get the coordinates of bounding boxes
    b_x1 = bboxes[:, 0]
    b_y1 = bboxes[:, 1]
    b_x2 = bboxes[:, 2]
    b_y2 = bboxes[:, 3]

    idxs = np.sort(idxs)

    area = (b_x2 - b_x1 + 1)*(b_y2 - b_y1 + 1) #Union Area

    while len(idxs) > 0:
        i = idxs[-1] #the last is one with maximum correlation score
        picked.append(i)

        #Get coord of intersecting rectangle
        inter_rect_x1 = np.maximum(b_x1[i], b_x1[idxs[:i]])
        inter_rect_y1 = np.maximum(b_y1[i], b_y1[idxs[:i]])
        inter_rect_x2 = np.minimum(b_x2[i], b_x2[idxs[:i]])
        inter_rect_y2 = np.minimum(b_y2[i], b_y2[idxs[:i]])

        #Intersection Area
        w = np.maximum(0, inter_rect_x2 - inter_rect_x1 + 1)
        h = np.maximum(0, inter_rect_y2 - inter_rect_y1 + 1)
        inter_area = (w*h)

        ious = inter_area/area[idxs[:i]]

        #delete bboxes detecting same object
        #print(idxs, i)
        idxs = np.delete(idxs,
                np.concatenate((np.where(idxs==i)[0], np.where(ious >= thresh)[0]))
            )

    return bboxes[picked].astype(int)


def match(img, templ, metric):
    result = cv2.matchTemplate(img, templ, metric)#cv2.TM_CCORR_NORMED
    (Ys, Xs) = np.where(result >=0.9)
    bboxes = []
    for (x, y) in zip(Xs, Ys):
        bboxes.append([x, y, x+templ.shape[1], y+templ.shape[0], result[y][x]])

    picked = nms(np.array(bboxes), 0.3)
    if len(picked)!=1:
        cv2.putText(img, "ANORMAL",
                    (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, [0, 0, 255], 2
                )
        print("\a")
    for (x1, y1, x2, y2, _) in picked:
        cv2.rectangle(
            img, (x1, y1), (x2, y2), (0, 255, 0), 2, 8, 0
            )
        t_size = cv2.getTextSize("spoon", cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        cv2.putText(img, "spoon",
                    (x1, y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 1, [255, 0, 0], 2
                )
##    Single object detection
##    cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1)
##    _minVal, _maxVal, minLoc, maxLoc = cv2.minMaxLoc(result, None)
##  
##    
##    matchLoc = minLoc if metric==cv2.TM_SQDIFF_NORMED else maxLoc
##

##        cv2.rectangle(
##            img, matchLoc, (matchLoc[0]+templ.shape[1], matchLoc[1]+templ.shape[0]),
##            (0, 255, 0), 2, 8, 0
##            )


def main(img_path, templ_path, metric):
    m = {0:cv2.TM_CCORR_NORMED, 1:cv2.TM_SQDIFF_NORMED}
    templ = cv2.imread(templ_path, cv2.IMREAD_COLOR)
    if not os.path.isdir(img_path):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        match(img, templ, m[metric])
        cv2.imshow("output", img)
        cv2.waitKey(0)
    else:
        cap = cv2.VideoCapture(img_path+"/BabyFood-Test%02d.jpg")
        fps = 2
        v_writer = None
        assert cap.isOpened(), "Unable to read from path"
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                w, h = frame.shape[:2]
                match(frame, templ, m[metric])

                if not v_writer:
                    v_writer = cv2.VideoWriter('./anormaly_detection.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, (h,w))
                v_writer.write(frame)
        v_writer.release()

if __name__ == "__main__":
    args = arg_parser()
    img_path = args.img_path
    templ_path = args.templ_path
    metric = args.metric

    main(img_path, templ_path, metric)
