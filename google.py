import io
import os

import sys
sys.path.append("..")

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types

# Instantiates a client
client = vision.ImageAnnotatorClient()
import concurrent.futures
import json
import time

def all(img_url):
    labels = detect_labels_uri(img_url)
    objects = localize_objects_uri(img_url)
    faces = detect_faces(img_url)
    return {
        "labels": labels,
        "objects":objects,
        "faces":faces
    }

    
def detect_labels_uri(uri):
    client = vision.ImageAnnotatorClient()
    image = vision.types.Image()
    image.source.image_uri = uri

    response = client.label_detection(image=image)
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    labels = response.label_annotations
    output = []
    for label in labels:
#         output.append({"description":label.description, "confidence":label.score*100})
        output.append(label.description)
    
    return output

def detect_faces(uri):
    client = vision.ImageAnnotatorClient()
    image = vision.types.Image()
    image.source.image_uri = uri

    response = client.face_detection(image=image)
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    faces = response.face_annotations

    # Names of likelihood from google.cloud.vision.enums
    likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                       'LIKELY', 'VERY_LIKELY')
    output = []
    for face in faces:
        out= {}
        out['anger'] = likelihood_name[face.anger_likelihood]
        out['joy']=likelihood_name[face.joy_likelihood]
        out['surprise']=likelihood_name[face.surprise_likelihood]
        for vertex in face.bounding_poly.normalized_vertices:
            out["bounding_box"].append({"x":vertex.x, "y":vertex.y})
        output.append(out)
    return output
        
def localize_objects_uri(uri):
    client = vision.ImageAnnotatorClient()

    image = vision.types.Image()
    image.source.image_uri = uri
    output = []
    response = client.object_localization(
        image=image)
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    
    objects =response.localized_object_annotations 
    for object in objects:
        out = object.name
#         out = {"description":object.name, "confidence":object.score, "bounding_box":[]}
#         for vertex in object.bounding_poly.normalized_vertices:
#             out["bounding_box"].append({"x":vertex.x, "y":vertex.y})
        output.append(out)
    return output

def call_google(uri):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        labels = executor.submit(detect_labels_uri, uri)
        objects = executor.submit(localize_objects_uri,uri)
    return labels.result(),objects.result()

def parallel_call():
    base_url = "gs://data-development/test-output/18baf93d-fc7b-4719-b05c-881dd9ce7f7e_Showtime_30m_720p_The_Circus_S5_E13/frames/";
    with open('frames.json') as f:
        frames_json = json.load(f)
    frames_json = frames_json[:10]
    start_time = time.time()
    result = []
#     for f in frames_json[:10]:
#         url = base_url + f['filename']
#         output = detect_labels_uri(url)
    with concurrent.futures.ThreadPoolExecutor() as executor:
#     with concurrent.futures.ProcessPoolExecutor() as executor:
    # Start the load operations and mark each future with its URL
        future_to_url = {executor.submit(call_google, base_url+frame['filename']): frame for frame in frames_json}
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                labels,objects = future.result()
                result.append({"labels":labels, "objects":objects})
            except Exception as exc:
                print('%r generated an exception: %s' % (url, exc))
#             else:
#                 print('%r page is %d bytes' % (url, len(labels)))

    print("--- %s seconds ---" % (time.time() - start_time))
    a=1

def main():
    #just change the url here and test on different images
    
    url = "gs://dbmo-sandbox/test-output/99a6d39b-93b7-4da9-85b5-e412e36ad57f_test-videos_dn2020-0429_vid_1min.mp4/frames/755.jpg"
#     detect_labels_uri(url)
#     detect_faces(url)
#     localize_objects_uri(url)
#     output = all(url)
    parallel_call()
#     print(output)


if __name__ == "__main__":
    main()