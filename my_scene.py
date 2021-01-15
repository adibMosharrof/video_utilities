import scenedetect
from scenedetect.frame_timecode import FrameTimecode
from scenedetect import VideoManager, SceneManager, StatsManager
import csv
import datetime

class MyScene:
    
    def __init__(self, scene_tuple, stats_manager):
        self.start = scene_tuple[0]
        self.end = scene_tuple[1]
        center_frame_num = (self.start.get_frames() + self.end.get_frames())//2
        self.center_frame_num = center_frame_num
        self.center = FrameTimecode(center_frame_num, self.start.get_framerate())
        self.duration = MyScene.get_scene_duration(scene_tuple)
        self.content_val = stats_manager.get_metrics(center_frame_num, ['content_val'])[0]
        self.dhash = None
    
    def format_time(self, seconds):
        a = str(datetime.timedelta(seconds=round(seconds,2)))
        try:
            ind = a.index('.')
        except ValueError:
            return a
        return a[:ind+3]
    
        
    @staticmethod
    def get_scene_duration(scene):
        return (scene[1] - scene[0]).get_seconds()
    
    
    @staticmethod
    def filter_scene_length(scene,scene_length):
        if MyScene.get_scene_duration(scene)> scene_length:
            return True
        return False
    
    @staticmethod
    def initialize(scenes, stats_manager, video, scene_length):
        my_scenes = []
        for s in scenes:
            if not MyScene.filter_scene_length(s, scene_length):
                continue
            my_scenes.append(MyScene(s, stats_manager)) 
        return my_scenes

    
    
    @staticmethod
    def scenes_to_csv(scenes, out_file):
        headers = ['number',  'start_time', 'center_time','end_time', 'duration', 'start_frame', 'center_frame', 'end_frame']
        rows = []
        for i,s in enumerate(scenes):
            rows.append([i, s.format_time(s.start.get_seconds()), s.format_time(s.center.get_seconds()), s.format_time(s.end.get_seconds()), s.format_time(s.duration), s.start.get_frames(), s.center_frame_num, s.end.get_frames()])
        with open(out_file, 'w') as csv_file:
            writer = csv.writer(csv_file, lineterminator='\n')
            writer.writerow(headers);
            writer.writerows(rows);