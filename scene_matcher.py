
import pickle
import os
import cv2
from scipy.misc import imread
import numpy as np
import scipy
import random
import csv
from shutil import copyfile

class Matcher(object):

    def __init__(self, pickled_db_path="features.pck"):
        with open(pickled_db_path, 'rb') as fp:
            self.data = pickle.load(fp)
        self.names = []
        self.matrix = []
        for k, v in self.data.items():
            self.names.append(k)
            self.matrix.append(v)
        self.matrix = np.array(self.matrix)
        self.names = np.array(self.names)

    def cos_cdist(self, vector):
        # getting cosine distance between search image and images database
        v = vector.reshape(1, -1)
        return scipy.spatial.distance.cdist(self.matrix, v, 'cosine').reshape(-1)

    def match(self, image_path, topn=5):
        features = extract_features(image_path)
        img_distances = self.cos_cdist(features)
        # getting top 5 records
        nearest_ids = np.argsort(img_distances)[:topn].tolist()
        nearest_img_paths = self.names[nearest_ids].tolist()

        return nearest_img_paths, img_distances[nearest_ids].tolist()
   
def get_image_name(path):
    return path.split("\\")[-1][:-4]

def feature_match(video_name, pickle_file_path, out_folder):

    images_path = os.path.join(out_folder , video_name, 'scenes' )
    files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]
    # getting 3 random images 
#     test = [6, 9, 138]
#     sample = [os.path.join(images_path, f'{t}.jpg') for t in test]
    
#     sample = random.sample(files, 10)
    sample = files
    
    if not os.path.exists(pickle_file_path):
        batch_extractor(images_path, pickle_file_path)

    ma = Matcher(pickle_file_path)
    result = []
    similar_scenes = set([])
    for s in sample:
#         print (f'Query image {get_image_name(s)}==========================================')
#         show_img(s)
        names, match = ma.match(s, topn=10)
        qname = get_image_name(s)
        similar_scenes.add(qname)
        for i in range(3):
            # we got cosine distance, less cosine distance between vectors
            # more they similar, thus we subtruct it from 1 to get match value
            score = 1-match[i]
            if score > 0.81 and score < 0.99:
#                 res.setdefault(qname, []).append([get_image_name(names[i]), 1-match[i]])
                print (f'Query Image {get_image_name(s)}==========================================')
                print(f'Result {get_image_name(names[i])}, Match {1-match[i]}' )
                result.append([get_image_name(s),get_image_name(names[i]), round(1-match[i],3)])
                similar_scenes.add(get_image_name(names[i]))
#     for k,vals in res.items():
#         print(f'Query image {k}')
#         print(f'Results')
#         for v in vals:
#             print(f'Image {v[0]}, match {v[1]}')
    write_scene_match_csv(result, os.path.join(out_folder, video_name, 'similar_scenes.csv'))
    sim_scenes_path = os.path.join(out_folder, video_name, 'similar_scenes')
    if not os.path.exists(sim_scenes_path):
        os.makedirs(sim_scenes_path)
    else:
        empty_dir(sim_scenes_path)
    write_similar_scenes(similar_scenes,os.path.join(out_folder, video_name, 'scenes') ,sim_scenes_path)

def write_similar_scenes(scenes, source_path, dest_path):
    for s in scenes:
        copyfile(os.path.join(source_path, f'{s}.jpg'), os.path.join(dest_path, f'{s}.jpg'))


def write_scene_match_csv(rows, out_file):
    header = ['Query Scene', 'Similar Scene', 'Similarity Score']
    with open(out_file, 'w') as csv_file:
        writer = csv.writer(csv_file, lineterminator='\n')
        writer.writerow(header);
        writer.writerows(rows);


def extract_features(image_path, vector_size=32):
    image = imread(image_path, mode="RGB")
    try:
        # Using KAZE, cause SIFT, ORB and other was moved to additional module
        # which is adding addtional pain during install
        alg = cv2.KAZE_create()
        # Dinding image keypoints
        kps = alg.detect(image)
        # Getting first 32 of them. 
        # Number of keypoints is varies depend on image size and color pallet
        # Sorting them based on keypoint response value(bigger is better)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        # computing descriptors vector
        kps, dsc = alg.compute(image, kps)
        # Flatten all of them in one big vector - our feature vector
        dsc = dsc.flatten()
        # Making descriptor of same size
        # Descriptor vector size is 64
        needed_size = (vector_size * 64)
        if dsc.size < needed_size:
            # if we have less the 32 descriptors then just adding zeros at the
            # end of our feature vector
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
    except cv2.error as e:
        print ('Error: ', e)
        return np.empty(vector_size*64)
    except AttributeError:
        print('Attribute error in image ', image_path)
        return np.empty(vector_size*64)

    return dsc

def batch_extractor(images_path, pickled_db_path="features.pck"):
    files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]

    result = {}
    for f in files:
        print ('Extracting features from image %s' % f)
        name = f.split('/')[-1].lower()
        result[name] = extract_features(f)
    with open(pickled_db_path, 'wb') as fp:
        pickle.dump(result, fp)
        
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