from moviepy.editor import VideoFileClip
from svn_pipeline import *
from yolo_pipeline import *
from lane import *


def pipeline_yolo(img):

    img_undist, img_lane_augmented, lane_info = lane_process(img)
    output = vehicle_detection_yolo(img_undist, img_lane_augmented, lane_info)

    return output

def pipeline_svn(img):

    img_undist, img_lane_augmented, lane_info = lane_process(img)
    output = vehicle_detection_svn(img_undist, img_lane_augmented, lane_info)

    return output


if __name__ == "__main__":

    demo = 3  # 1:image (YOLO and SVN), 2: video (YOLO Pipeline), 3: video (SVN pipeline)

    if demo == 1:
        filename = 'examples/test4.jpg'
        image = mpimg.imread(filename)

        #(1) Yolo pipeline
        yolo_result = pipeline_yolo(image)
        plt.figure()
        plt.imshow(yolo_result)
        plt.title('yolo pipeline', fontsize=30)

        #(2) SVN pipeline
        draw_img = pipeline_svn(image)
        fig = plt.figure()
        plt.imshow(draw_img)
        plt.title('svn pipeline', fontsize=30)
        plt.show()

    elif demo == 2:
        # YOLO Pipeline
        video_output = 'examples/project_YOLO.mp4'
        clip1 = VideoFileClip("examples/project_video.mp4").subclip(30,32)
        clip = clip1.fl_image(pipeline_yolo)
        clip.write_videofile(video_output, audio=False)

    else:
        # SVN pipeline
        video_output = 'examples/project_svn.mp4'
        clip1 = VideoFileClip("examples/project_video.mp4").subclip(30,32)
        clip = clip1.fl_image(pipeline_svn)
        clip.write_videofile(video_output, audio=False)


