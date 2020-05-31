
import cv2
import numpy as np
import random
import sys


# The dimensions of any input frame
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# Dimensions of the various window frames used during prediction
IMAGE_DIM = 50
WINDOW_DIM = 80
MODEL_OUTPUT_DIM = 76
IMAGE_WINDOW_START = (WINDOW_DIM - IMAGE_DIM) // 2
MODEL_WINDOW_START = (WINDOW_DIM - MODEL_OUTPUT_DIM) // 2
IMAGE_MODEL_START = (MODEL_OUTPUT_DIM - IMAGE_DIM) // 2

# The threshold value which will dictate whether we use the technique of frame averaging or not
AVERAGE_FRAME_DISTANCE_THRESHOLD_VALUE = 150

# Definition of important directories
root_directory = './'
model_directory = root_directory + 'Model/'



''' Load the model to be used for prediction '''
json_file = open(model_directory + 'model80_76.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

from keras.models import model_from_json
model = model_from_json(loaded_model_json)
model.load_weights(model_directory + 'model80_76.h5')



''' The function defined below essentially figures out whether to use the technique 
    of frame averaging for intermediate window generation or not 
'''
def use_frame_averaging(frame1_window, frame2_window):
    average_window = (frame1_window + frame2_window) // 2
    frame1_window_average_window_dist = np.mean((frame1_window - average_window) ** 2)
    frame2_window_average_window_dist = np.mean((frame2_window - average_window) ** 2)
    if (frame1_window_average_window_dist < AVERAGE_FRAME_DISTANCE_THRESHOLD_VALUE and 
                frame2_window_average_window_dist < AVERAGE_FRAME_DISTANCE_THRESHOLD_VALUE):
        return True
    return False



''' The function defined below computes an intermediate frame for a given pair of frames. 
    The intermediate frame is generated through composition of different intermediate windows, 
    obtained through averaging or model prediction. 
'''
def insert_frame(frame1, frame2):
    height, width = frame1.shape[0], frame1.shape[1]
    pad_h = 0 if height % IMAGE_DIM == 0 else (-height) % IMAGE_DIM
    pad_w = 0 if width % IMAGE_DIM == 0 else (-width) % IMAGE_DIM
    frame1 = np.pad(frame1, ((IMAGE_WINDOW_START, IMAGE_WINDOW_START + pad_h), (IMAGE_WINDOW_START, IMAGE_WINDOW_START + pad_w), (0, 0)))
    frame2 = np.pad(frame2, ((IMAGE_WINDOW_START, IMAGE_WINDOW_START + pad_h), (IMAGE_WINDOW_START, IMAGE_WINDOW_START + pad_w), (0, 0)))

    new_frame = np.zeros(frame1.shape)
    prediction_queue_ids = []
    prediction_queue_frames = []
    for i in range(IMAGE_WINDOW_START, IMAGE_WINDOW_START + height + pad_h, IMAGE_DIM):
        for j in range(IMAGE_WINDOW_START, IMAGE_WINDOW_START + width + pad_w, IMAGE_DIM):
            frame1_window = frame1[i:i + IMAGE_DIM, j:j + IMAGE_DIM, :]
            frame2_window = frame2[i:i + IMAGE_DIM, j:j + IMAGE_DIM, :]
            if (use_frame_averaging(frame1_window, frame2_window)):
                window_pred = (frame1_window + frame2_window) // 2
                new_frame[i:i + IMAGE_DIM, j:j + IMAGE_DIM, :] = window_pred
            else:
                window = np.zeros((WINDOW_DIM, WINDOW_DIM, 6))
                window[:, :, 0:3] = frame1[(i - IMAGE_WINDOW_START):(i + IMAGE_DIM + IMAGE_WINDOW_START),
                                        (j - IMAGE_WINDOW_START):(j + IMAGE_DIM + IMAGE_WINDOW_START), :]
                window[:, :, 3:6] = frame2[(i - IMAGE_WINDOW_START):(i + IMAGE_DIM + IMAGE_WINDOW_START),
                                        (j - IMAGE_WINDOW_START):(j + IMAGE_DIM + IMAGE_WINDOW_START), :]
                prediction_queue_ids.append((i, j))
                prediction_queue_frames.append(window)
    
    if(len(prediction_queue_ids) > 0):
        predicted_queue = model.predict(np.array(prediction_queue_frames, dtype=np.float32))
        for idx in range(len(prediction_queue_ids)):
            prediction = predicted_queue[idx, IMAGE_MODEL_START:MODEL_OUTPUT_DIM - IMAGE_MODEL_START, 
                                            IMAGE_MODEL_START:MODEL_OUTPUT_DIM - IMAGE_MODEL_START, :]
            prediction = np.maximum(0, prediction)
            prediction = np.minimum(255, prediction)
            i, j = prediction_queue_ids[idx]
            new_frame[i:i + IMAGE_DIM, j:j + IMAGE_DIM, :] = prediction

    return new_frame[IMAGE_WINDOW_START:IMAGE_WINDOW_START + height, IMAGE_WINDOW_START:IMAGE_WINDOW_START + width, :]



''' The following function inserts multiple frames between a pair of frames, by using 
    a divide and conquer approach.
'''
def insert_frames_recursively(first_frame, last_frame, frame_count):
    if frame_count <= 0:
        return []
    else:
        mid = (frame_count - 1) // 2
        mid_frame = insert_frame(first_frame, last_frame)
        frames = insert_frames_recursively(first_frame, mid_frame, mid)
        frames.append(mid_frame)
        frames.extend(insert_frames_recursively(mid_frame, last_frame, frame_count - 1 - mid))
        return frames



''' The following function scales up the FPS of any video by a given factor, by 
    leveraging the functions we defined above 
'''
def sample_up_fps(path, factor):
    assert type(factor) == type(0) and factor > 1
    video = cv2.VideoCapture(path)
    out = cv2.VideoWriter('clip_u.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
            video.get(cv2.CAP_PROP_FPS) * factor, (320, 240))
    count = 0
    last_frame = None
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            count += 1
            if (count % 2 == 0):
                print(count)
            if type(last_frame) == type(None):
                out.write(frame)
                last_frame = frame
            else:
                frames = insert_frames_recursively(np.array(cv2.resize(last_frame, (FRAME_WIDTH, FRAME_HEIGHT)), dtype=np.uint64), 
                                np.array(cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT)), dtype=np.uint64), factor - 1)
                for new_frame in frames:
                    out.write(np.array(new_frame, dtype=np.uint8))
                out.write(frame)
                last_frame = frame
        else:
            break

    video.release()
    cv2.destroyAllWindows()



''' The following function retreives the dimensions of any given video '''
def get_dims(video):
    return (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))



''' The following function retreives the effective FPS for a video which is to be sampled down '''
def get_reduced_fps(video, reduction_factor):
    org_frames_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    org_fps = video.get(cv2.CAP_PROP_FPS)
    duration = org_frames_count / org_fps
    new_frames_count = org_frames_count * reduction_factor
    new_fps = new_frames_count / duration
    return new_fps



''' The following function samples down the FPS of any video by a given factor '''
def sample_down_fps(path, factor):
    assert factor > 0 and factor <= 1
    video = cv2.VideoCapture(path)
    new_fps = get_reduced_fps(video, factor)
    out = cv2.VideoWriter('clip_d.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 
            new_fps, (FRAME_WIDTH, FRAME_HEIGHT))
    count = 0
    total_frames = 0
    last_frame = None
    while (video.isOpened()):
        ret, frame = video.read()
        if ret == True:
            count += 1
            if count % (1 / factor) < 1:
                if (count % (1 / factor) > 0.5 and type(last_frame) != type(None)):
                    out.write(cv2.resize(last_frame, (FRAME_WIDTH, FRAME_HEIGHT)))
                else:
                    out.write(cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT)))
                total_frames += 1
            last_frame = frame
        else:
            break
    
    print('FPS downscaled by a factor of', round(total_frames / video.get(cv2.CAP_PROP_FRAME_COUNT), 4))
    video.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    path = sys.argv[1]
    mode = sys.argv[2]
    factor = sys.argv[3]
    assert mode[0] == '-'
    assert mode[-1] == 'u' or mode[-1] == 'd'
    if mode[-1] == 'u':
        sample_up_fps(path, int(factor))
    else:
        sample_down_fps(path, float(factor))