import cv2
import numpy as np

def draw_lane_status(frame, lane_info, threshold_offset = 0.6):

    font = cv2.FONT_HERSHEY_SIMPLEX
    info_road = "Lane Status"
    info_lane = "Direction: {0}".format(lane_info['curve_direction'])
    info_cur = "Curvature {:6.1f} m".format(lane_info['curvature'])
    info_offset = "Off center: {0} {1:3.1f}m".format(lane_info['dev_dir'], lane_info['offset'])

    l_uper = (10,10)

    cv2.line(frame,(l_uper[0] + 265,0),(l_uper[0] + 265,155),(255,0,0),5)

    cv2.putText(frame, info_road, (50,32+5), font, 0.8, (255,255,0), 2,cv2.LINE_AA)
    cv2.putText(frame, info_lane, (16,60+10), font, 0.6, (255,255,0), 1,cv2.LINE_AA)
    cv2.putText(frame, info_cur, (16,80+25), font, 0.6, (255,255,0), 1,cv2.LINE_AA)

    if lane_info['offset'] >= threshold_offset:
        cv2.putText(frame, info_offset, (16,100+40), font, 0.6, (255,0,0), 1,cv2.LINE_AA)
    else:
        cv2.putText(frame, info_offset, (16,100+40), font, 0.6, (255,255,0), 1,cv2.LINE_AA)

def draw_speed(img_cp, fps, w):

    fps_info = "{0:4.1f} fps".format(fps)
    cv2.putText(img_cp, 'Speed', (w - 120,37), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2, cv2.LINE_AA)
    cv2.putText(img_cp, fps_info, (w - 130,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 1, cv2.LINE_AA)
    cv2.line(img_cp,(w-160,0),(w-160,155),(255,0,0),5)


def draw_thumbnails(img_cp, img, window_list, thumb_w=100, thumb_h=80, off_x=30, off_y=30):

    cv2.putText(img_cp, 'Detected viehicles', (400,37), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2, cv2.LINE_AA)
    for i, bbox in enumerate(window_list):
        thumbnail = img[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
        vehicle_thumb = cv2.resize(thumbnail, dsize=(thumb_w, thumb_h))
        start_x = 300 + (i+1) * off_x + i * thumb_w
        img_cp[off_y + 30:off_y + thumb_h + 30, start_x:start_x + thumb_w, :] = vehicle_thumb


def draw_background_highlight(image, draw_img, w):

    mask = cv2.rectangle(np.copy(image), (0, 0), (w, 155), (0, 0, 0), thickness=cv2.FILLED)
    draw_img = cv2.addWeighted(src1=mask, alpha=0.3, src2=draw_img, beta=0.8, gamma=0)

    return draw_img
