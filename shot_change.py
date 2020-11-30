from google.cloud import videointelligence

""" Detects camera shot changes. """
path = "gs://dbmo-sandbox/test-videos/0195f5ae-e9d2-4e16-be3e-e34de6748645_test-videos_dn2020-0429_vid_1min.mp4"
video_client = videointelligence.VideoIntelligenceServiceClient()
features = [videointelligence.enums.Feature.SHOT_CHANGE_DETECTION]
operation = video_client.annotate_video(input_uri=path, features=features)
print("\nProcessing video for shot change annotations:")

result = operation.result(timeout=90)
print("\nFinished processing.")

# first result is retrieved because a single video was processed
for i, shot in enumerate(result.annotation_results[0].shot_annotations):
    start_time = shot.start_time_offset.seconds + shot.start_time_offset.nanos / 1e9
    end_time = shot.end_time_offset.seconds + shot.end_time_offset.nanos / 1e9
    print("\tShot {}: {} to {}".format(i, start_time, end_time))
# [END video_analyze_shots]