#Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#PDX-License-Identifier: MIT-0 (For details, see https://github.com/awsdocs/amazon-rekognition-developer-guide/blob/master/LICENSE-SAMPLECODE.)

import cv2
import os
import shutil
import json
# from google.cloud import videointelligence
import matplotlib.pyplot as plt
import scenedetect
from scenedetect.frame_timecode import FrameTimecode
from scenedetect import VideoManager
from scenedetect import SceneManager
import datetime

# For content-aware scene detection:
from scenedetect.detectors import ContentDetector

def display(*imgs):
    n = len(imgs)
    f = plt.figure()
    for i, img in enumerate(imgs):
        f.add_subplot(1, n, i + 1)
        plt.imshow(img)
    plt.show()


def get_frames(frames_path, start_time, end_time):
    txt = open(frames_path,"r+").read()
    frames = json.loads(txt)
    output = []
    for f in frames:
        t = f['timestamp']
        if t > end_time:
            break
        if t > start_time and t < end_time:
            output.append(f)
    return output 

def empty_dir(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))  
               
def _save_frames(video, scene_times, out_folder):
    out_folder_path = os.path.join('scenes',out_folder)
    empty_dir(out_folder_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    for i,t in enumerate(scene_times):
        ftc = FrameTimecode(timecode = t, fps =fps )
        frame_num = ftc.get_frames()
        video.set(1,frame_num)
        _,frame = video.read()
        try:
            cv2.imwrite( f'{out_folder_path}/{i}.jpg', frame)
        except:
            a=1
    a=1        


def frame_reduction(video_path, threshold=0.20):
    
    video=cv2.VideoCapture(video_path)
    fps=int(video.get(cv2.CAP_PROP_FPS))
    increment = fps
    
    i=1
    ref_frame = None
    frames_dict = []
    shots = []
    scene_times = []
    start_time = 0
    end_time = None
    
    while(video.isOpened()):
        if ref_frame is None:
            video.set(1,0)
            _, ref_frame= video.read()
        video.set(1,i)
        _,current_frame =video.read()
        
        if current_frame is None:
            break
        diff = frame_diff(current_frame, ref_frame)
        if diff > threshold:
#             print(f"writing frame num {i}")
#             filtered_frames.append(i)
            timestamp = video.get(cv2.CAP_PROP_POS_MSEC)
            t = round(timestamp/1000,4)
            frame_name = f'{i}.jpg'
            frames_dict.append({"frame_id":frame_name, "time":t})
            
            shots.append({"start":start_time, "end":t})
            scene_times.append((start_time+t)/2)
            start_time = t
            
            ref_frame = current_frame
        i+=increment
    print("Frame Reduction Shots")
    for i,s in enumerate(shots):
        print(f'Shot {i}: {s["start"]} to {s["end"]}')
    _save_frames(video, scene_times, 'frame_reduction')

def frame_diff(cf, rf):
    hsv_base = cv2.cvtColor(rf, cv2.COLOR_BGR2HSV)
    hsv_test1 = cv2.cvtColor(cf, cv2.COLOR_BGR2HSV)
    
    h_bins = 50
    s_bins = 60
    histSize = [h_bins, s_bins]
    
    # hue varies from 0 to 179, saturation from 0 to 255
    h_ranges = [0, 180]
    s_ranges = [0, 256]
    ranges = h_ranges + s_ranges # concat lists
    # Use the 0-th and 1-st channels
    channels = [0, 1]
    
    hist_base = cv2.calcHist([hsv_base], channels, None, histSize, ranges, accumulate=False)
    cv2.normalize(hist_base, hist_base, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    hist_test1 = cv2.calcHist([hsv_test1], channels, None, histSize, ranges, accumulate=False)
    cv2.normalize(hist_test1, hist_test1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    
    diffs = []
    for compare_method in range(4):
        base_base = cv2.compareHist(hist_base, hist_base, compare_method)
        base_test1 = cv2.compareHist(hist_base, hist_test1, compare_method)
        diffs.append((base_base,base_test1))
    return diffs[3][1]

def format_time(seconds):
    a = str(datetime.timedelta(seconds=round(seconds,2)))
    try:
        ind = a.index('.')
    except ValueError:
        return a
    return a[:ind+3]

def google_shot_change(video):
    """ Detects camera shot changes. """
#     path = "gs://dbmo-sandbox/test-videos/0195f5ae-e9d2-4e16-be3e-e34de6748645_test-videos_dn2020-0429_vid_1min.mp4"
    path = "gs://dbmo-sandbox/test-videos/9d6f8252-d905-49cc-bcc5-ccb10d607a33_Showtime_60m_720p_Billions_S2_E12.mp4"
    video_client = videointelligence.VideoIntelligenceServiceClient()
    features = [videointelligence.enums.Feature.SHOT_CHANGE_DETECTION]
    operation = video_client.annotate_video(input_uri=path, features=features)
    print("\nProcessing video for shot change annotations:")
    
    result = operation.result()    
    scene_times = []
    # first result is retrieved because a single video was processed
    for i, shot in enumerate(result.annotation_results[0].shot_annotations):
        start_time = shot.start_time_offset.seconds + shot.start_time_offset.nanos / 1e9
        end_time = shot.end_time_offset.seconds + shot.end_time_offset.nanos / 1e9
        print("\tShot {}: {} to {}".format(i, format_time(start_time), format_time(end_time)))
        scene_times.append( (start_time+ end_time)/2)
    _save_frames(video, scene_times, "google")

def scene_detect(video_path, video , threshold=50):
        # Create our video & scene managers, then add the detector.
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold))

    # Base timestamp at frame 0 (required to obtain the scene list).
    base_timecode = video_manager.get_base_timecode()

    # Improve processing speed by downscaling before processing.
    video_manager.set_downscale_factor()

    # Start the video manager and perform the scene detection.
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    # Each returned scene is a tuple of the (start, end) timecode.
    scenes = scene_manager.get_scene_list(base_timecode)
    num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    rate_of_scene_change = len(scenes)/num_frames
    print(f'Rate of scene change {rate_of_scene_change*100}')
    with open('scene.txt',"w") as f:
        f.write(f'Rate of scene change {rate_of_scene_change*100}')
    return
    scene_times = []
    for i,scene in enumerate(scenes):
        t1 = scene[0].get_seconds()
        t2 = scene[1].get_seconds()
        print(f'Shot {i+1} {format_time(t1)} to {format_time(t2)}')
        scene_times.append((t1+t2)/2)
    _save_frames(video, scene_times, 'scene_detect')    
    a=1
    

def main():
    video_path='tmp/test1.mp4'
#     photo = "gs://dbmo-sandbox/test-output/99a6d39b-93b7-4da9-85b5-e412e36ad57f_test-videos_dn2020-0429_vid_1min.mp4/frames/755.jpg"
    bucket='objdetectionvideos'
    
    video=cv2.VideoCapture(video_path)
    fps=int(video.get(cv2.CAP_PROP_FPS))
    
#     get_frames("frames.json", 20,30)   
#     print("Labels detected: " + str(label_count))
#     google_shot_change(video)
#     frame_reduction(video_path, 0.75)
    scene_detect(video_path, video)
    
if __name__ == "__main__":
    main()