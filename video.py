#Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#PDX-License-Identifier: MIT-0 (For details, see https://github.com/awsdocs/amazon-rekognition-developer-guide/blob/master/LICENSE-SAMPLECODE.)
from PIL import Image
import imagehash
import cv2
import os
import shutil
from shutil import copyfile
import json
import vptree
# from google.cloud import videointelligence
import matplotlib.pyplot as plt
import scenedetect
from scenedetect.frame_timecode import FrameTimecode
from scenedetect import VideoManager, SceneManager, StatsManager
from pathlib import Path
import numpy as np
import datetime
import argparse
# For content-aware scene detection:
from scenedetect.detectors import ContentDetector
import plotly.express as px
# import dash
# import dash_core_components as dcc
# import dash_html_components as html
from statistics import mean
import logging

# from skimage import io
from my_scene import MyScene
from scene_matcher import feature_match
tmp = 'tmp'
out_folder = 'output'




        
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
    if not os.path.isdir(folder):
        os.makedirs(folder)
        return
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))  
               
def _save_frames(video, scenes, out_folder_path):
    empty_dir(out_folder_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    for i,scene in enumerate(scenes):
        frame_num = scene.center_frame_num
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

def init_dhash(scenes, video, video_name):
    for i,scene in enumerate(scenes):
        frame_num = scene.center_frame_num
        video.set(1,frame_num)
        _,frame = video.read()
        hash_val = dhash(frame)
        scene.dhash = hash_val
    x = list(range(0,len(scenes)))
    y = [s.dhash for s in scenes]
    with open(f'{out_folder}/{video_name}/dhash.json', 'w') as f:
        json.dump(y, f)
    plot_bar_chart(x, y, f'{out_folder}/{video_name}/{video_name}_dhash.png', 'DHash values')

def dhash(image, hashSize=8):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(image, (hashSize + 1, hashSize))
    diff = resized[:, 1:] > resized[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def plot_scene_graph(my_scenes, video_out_dir, video_name):
    y = [1]* len(my_scenes)
    x = [s.center.get_seconds() for s in my_scenes]
    plt.figure()
    plt.scatter(x,y, s=1)
    plt.suptitle(video_name, fontsize=20)
    plt.savefig(f'{video_out_dir}/{video_name}_scene_times.png', dpi=300)
    plt.close()
#     fig = px.scatter(x=x, y=y)
#     app = dash.Dash()
#     app.layout = html.Div([
#         dcc.Graph(figure=fig)
#     ])
#     
#     app.run_server(debug=True, use_reloader=False)  # Turn off reloader if inside Jupyter
    a=1;

def plot_bar_chart(x,y,out_path, title):
    plt.figure()
    plt.bar(x,y)
    plt.suptitle(title, fontsize=10)
    plt.savefig(out_path)
    plt.close()


def reduce_scenes(my_scenes, threshold = 2.0):
    black = 0
    scenes = []
    vals = {}
    for scene in my_scenes:
        cval = scene.content_val
        if cval is black:
            continue
        try:
            vals[cval]
        except KeyError:
            scenes.append(scene)
            vals[cval] = True
    return scenes

def scene_detect(video_path, video , scene_length = 2.0 ,threshold=50):
        # Create our video & scene managers, then add the detector.
    video_manager = VideoManager([video_path])
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    scene_manager.add_detector(
        ContentDetector(threshold=threshold))
    base_timecode = video_manager.get_base_timecode()

    video_name = Path(video_path).stem
    video_out_dir = f'{out_folder}/{video_name}/'
    stats_file_path = f'{video_out_dir}/{video_name}.stats.csv'
    if os.path.exists(stats_file_path):
        with open(stats_file_path, 'r') as stats_file:
            stats_manager.load_from_csv(stats_file, base_timecode)
    if not os.path.exists(f'{video_out_dir}'):
        os.makedirs(video_out_dir)
        
    logging.basicConfig(filename=f'{video_out_dir}/{video_name}.log', level=logging.INFO, filemode='w')
    # Base timestamp at frame 0 (required to obtain the scene list).

    # Improve processing speed by downscaling before processing.
    video_manager.set_downscale_factor()

    # Start the video manager and perform the scene detection.
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    # Each returned scene is a tuple of the (start, end) timecode.
#     scenes = scene_manager.get_scene_list(base_timecode)
    my_scenes = MyScene.initialize(scene_manager.get_scene_list(base_timecode), stats_manager, video, scene_length)
    scene_out_path = os.path.join(video_out_dir,'scenes')
    _save_frames(video, my_scenes, scene_out_path)  
    MyScene.scenes_to_csv(my_scenes, os.path.join(video_out_dir, 'scene_details.csv'))
#     reduce_frames(scene_out_path)
#     plot_content_vals(my_scenes, video_name)
#     init_dhash(my_scenes, video, video_name)
    num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    rate_of_scene_change = len(my_scenes)/num_frames
    print(f'Rate of scene change {rate_of_scene_change*100}')
    logging.info(f'Rate of scene change {rate_of_scene_change*100}')
    
    if stats_manager.is_save_required():
        with open(stats_file_path, 'w') as stats_file:
            stats_manager.save_to_csv(stats_file, base_timecode)
    
    with open('scene.txt',"w") as f:
        f.write(f'Rate of scene change {rate_of_scene_change*100} \n')
        f.write(f'Video name {video_path.split("/")[-1][:-4]}')
#     return
#     scene_times = []
#     for i,scene in enumerate(scenes):
#         t1 = scene[0].get_seconds()
#         t2 = scene[1].get_seconds()
#         print(f'Shot {i+1} {format_time(t1)} to {format_time(t2)}')
#         scene_times.append((t1+t2)/2)
    avg_scene_duration = mean([s.duration for s in my_scenes])
    print(f'average scene duration {avg_scene_duration}')
    logging.info(f'average scene duration {avg_scene_duration}')
    plot_scene_graph(my_scenes, video_out_dir, video_name)
#     _save_frames(video, my_scenes, video_name)    
    
def hamming(a, b):
    # compute and return the Hamming distance between the integers
    return a-b
#    return bin(int(a) ^ int(b)).count("1")

def convert_hash(image):
    # convert the hash to NumPy's 64-bit float and then back to
    # Python's built in int
#     h = dhash(image)
#     return int(np.array(h, dtype="float64"))
    return imagehash.average_hash(image)

def test_tree(video_dir= 'output/desus/'):
    scenes_dir = os.path.join(video_dir, 'scenes')
    image_names = os.listdir(scenes_dir)
    hashes = {}
    for i, file in enumerate(image_names):
#         img = cv2.imread(os.path.join(scenes_dir, file))
        img = Image.open(os.path.join(scenes_dir, file))
        h = convert_hash(img)
         
        l = hashes.get(h, [])
        l.append(i)
        hashes[h] = l
    
    points = list(hashes.keys())
    tree = vptree.VPTree(points, hamming)
    res = {}
    for i,h in enumerate(hashes):
        sim_hashes = tree.get_all_in_range(h,5)
        if len(sim_hashes):
            res[i] = []
        for sh in sim_hashes:
            res[i].append(hashes[sh[1]])
#         res[i] = list(np.concatenate((res[i])))
        
    sim = {}
    for k,v in res.items():
        a=1
    a=1
        

def test_hash(video_dir= 'output/desus/'):
    image_dir = os.path.join(video_dir , 'scenes' )
    
    i1 = '9.jpg'
    i2 = '11.jpg'
    i3 = '138.jpg'
    i4 = '140.jpg'
    i5 = '143.jpg'
    
    images = os.listdir(image_dir)
#     images = [i1,i2,i3,i4,i5]
    duplicate = [False]* len(images)
    hashes = []
    for i, img in enumerate(images):
#         hashes.append(convert_hash(cv2.imread(i)))
        hashes.append(imagehash.average_hash(Image.open(os.path.join(image_dir,img)).convert('LA')))
    
    threshold = 15
    similars = {}
    for i1, img1 in enumerate(hashes):
        if duplicate[i1] is True:
            continue
        for i2, img2 in enumerate(hashes):
            if i1 == i2:
                continue
            diff = img1 -img2
            if (diff < threshold):
                try:
                    similars[i1]
                except KeyError:
                    similars[i1] = []
                similars[i1].append(i2)
                duplicate[i2] = True
            
#     t = vptree.VPTree(hashes, hamming)
    x = np.array(duplicate)
    imgs = np.array(images)
    res = imgs[x==False]
#     unique_scenes_dir = os.path.join(video_dir, 'unique_scenes')
#     if not os.path.isdir(unique_scenes_dir):
#         os.mkdir(unique_scenes_dir)
#     else:
#         empty_dir(unique_scenes_dir)
#     for r in res:
#         copyfile(os.path.join(root, r), os.path.join(unique_scenes_dir, r))
    a = 1

def show_img(path):
    img = imread(path, mode="RGB")
    plt.imshow(img)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Process Videos')
    parser.add_argument('-vp','--video_path', type=str,
                    default='desus',help='Video Path')
    
    args = parser.parse_args()
    video_path=f'{tmp}/{args.video_path}.mp4'
#     photo = "gs://dbmo-sandbox/test-output/99a6d39b-93b7-4da9-85b5-e412e36ad57f_test-videos_dn2020-0429_vid_1min.mp4/frames/755.jpg"
    bucket='objdetectionvideos'
    
    video=cv2.VideoCapture(video_path)
    fps=int(video.get(cv2.CAP_PROP_FPS))
    
#     get_frames("frames.json", 20,30)   
#     print("Labels detected: " + str(label_count))
#     google_shot_change(video)
#     frame_reduction(video_path, 0.75)
#     test_tree()
#     test_hash()
    video_name = Path(video_path).stem
    pickle_file_path = os.path.join(out_folder, video_name, 'features.pck')
    feature_match(video_name, pickle_file_path, out_folder)
    return
#     scene_detect(video_path, video)
    
if __name__ == "__main__":
    main()