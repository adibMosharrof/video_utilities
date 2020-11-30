#Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#PDX-License-Identifier: MIT-0 (For details, see https://github.com/awsdocs/amazon-rekognition-developer-guide/blob/master/LICENSE-SAMPLECODE.)

import boto3
import json
import wget

def detect_labels(photo, bucket):

    client=boto3.client('rekognition')
#     dl_from_url(photo)
#     image_filename = wget.download(photo)
    with open(photo, 'rb') as image:
        response = client.recognize_celebrities(Image={'Bytes': image.read()})

#     response = client.detect_labels(Image={'S3Object':{'Bucket':bucket,'Name':photo}},
#         MaxLabels=10)

    for label in response['Labels']:
        print(f"{label['Name']}, {label['Confidence']}")
#         print ("Label: " + label['Name'])
#         print ("Confidence: " + str(label['Confidence']))
#         print ("Instances:")
#         for instance in label['Instances']:
#             print ("  Bounding box")
#             print ("    Top: " + str(instance['BoundingBox']['Top']))
#             print ("    Left: " + str(instance['BoundingBox']['Left']))
#             print ("    Width: " +  str(instance['BoundingBox']['Width']))
#             print ("    Height: " +  str(instance['BoundingBox']['Height']))
#             print ("  Confidence: " + str(instance['Confidence']))
#             print()

#         print ("Parents:")
#         for parent in label['Parents']:
#             print ("   " + parent['Name'])
#         print ("----------")
#         print ()
    return len(response['Labels'])

def detect_celeb(photo, bucket):

    client=boto3.client('rekognition')

    response = client.recognize_celebrities(Image={'S3Object':{'Bucket':bucket,'Name':photo}},
        )

    for celebrity in response['CelebrityFaces']:
        print ('Name: ' + celebrity['Name'])
        print ('Id: ' + celebrity['Id'])
        print ('Position:')
        print ('   Left: ' + '{:.2f}'.format(celebrity['Face']['BoundingBox']['Height']))
        print ('   Top: ' + '{:.2f}'.format(celebrity['Face']['BoundingBox']['Top']))
        print ('Info')
        for url in celebrity['Urls']:
            print ('   ' + url)
        print
    return len(response['CelebrityFaces'])
    
def detect_facedetail(photo, bucket):
    client=boto3.client('rekognition')
    
    response = client.detect_faces(Image={'S3Object':{'Bucket':bucket,'Name':photo}},Attributes=['ALL'])
    
    for faceDetail in response['FaceDetails']:
        print('The detected face is between ' + str(faceDetail['AgeRange']['Low']) 
              + ' and ' + str(faceDetail['AgeRange']['High']) + ' years old')
        print('Here are the other attributes:')
        print(json.dumps(faceDetail, indent=4, sort_keys=True))
    return len(response['FaceDetails'])


def main():
    photo='tmp/755.jpg'
#     photo = "gs://dbmo-sandbox/test-output/99a6d39b-93b7-4da9-85b5-e412e36ad57f_test-videos_dn2020-0429_vid_1min.mp4/frames/755.jpg"
    bucket='objdetectionvideos'
    label_count=detect_labels(photo, bucket)
    print("Labels detected: " + str(label_count))

if __name__ == "__main__":
    main()