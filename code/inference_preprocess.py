import numpy as np
import cv2
from skimage.util import img_as_float

import matplotlib.pyplot as plt
from scipy.sparse import spdiags


def preprocess_raw_video(video_file_path, dim=36, max_duration=30):

    #########################################################################
    # set up
    t = []
    i = 0
    if video_file_path is None:
        cap = cv2.VideoCapture(0)
        # Check success
        if not cap.isOpened():
            raise Exception("Could not open video device")
        # Set properties. Each returns === True on success (i.e. correct resolution)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 60)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        max_frames = int(frame_rate * max_duration)  # Actual number of captured frames can  be less, if esc key hit while capturing
    else:
        cap = cv2.VideoCapture(video_file_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        max_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), max_duration * frame_rate)
    normalized_frames = np.zeros((max_frames, dim, dim, 3), dtype=np.float32)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    success, img = cap.read()
    dims = img.shape
    print("Original Height", height)
    print("Original width", width)
    print("Frame Rate", frame_rate)
    #########################################################################
    # Crop each frame size into dim x dim
    while success and i < max_frames:
        t.append(cap.get(cv2.CAP_PROP_POS_MSEC))  # current timestamp in millisecond
        frame_as_numpy = cv2.resize(img_as_float(img[:, int(width/2)-int(height/2 + 1):int(height/2)+int(width/2), :]), (dim, dim), interpolation=cv2.INTER_AREA)
        frame_as_numpy = cv2.rotate(frame_as_numpy, cv2.ROTATE_90_CLOCKWISE)  # rotate 90 degree
        frame_as_numpy = cv2.cvtColor(frame_as_numpy.astype('float32'), cv2.COLOR_BGR2RGB)
        frame_as_numpy[frame_as_numpy > 1] = 1
        frame_as_numpy[frame_as_numpy < (1/255)] = 1/255
        normalized_frames[i, :, :, :] = frame_as_numpy
        success, img = cap.read() # read the next one
        if video_file_path is None:
            cv2.imshow(f'Seconds remaining:', img)
            cv2.setWindowTitle(f'Seconds remaining:', f'([Esc] to interrupt) Seconds remaining: {max_duration - i // frame_rate}')
            if cv2.waitKey(1) & 0xFF == 27:
                break

        i = i + 1
    plt.imshow(normalized_frames[0])
    plt.title('Sample Preprocessed Frame')
    plt.show()

    #########################################################################
    # Normalized Frames in the motion branch
    normalized_len = len(t) - 1
    frame_diffs = np.zeros((normalized_len, dim, dim, 3), dtype=np.float32)
    for j in range(normalized_len - 1):
        frame_diffs[j, :, :, :] = (normalized_frames[j+1, :, :, :] - normalized_frames[j, :, :, :]) / (normalized_frames[j+1, :, :, :] + normalized_frames[j, :, :, :])
    frame_diffs = frame_diffs / np.std(frame_diffs)

    #########################################################################
    # Normalize raw frames in the appearance branch
    normalized_frames = normalized_frames - np.mean(normalized_frames)
    normalized_frames = normalized_frames  / np.std(normalized_frames)
    normalized_frames = normalized_frames[:max_frames-1, :, :, :]

    #########################################################################
    # Plot an example of data after preprocess
    # plt.imshow(normalized_frames[0])
    # plt.title('Sample Preprocessed Frame')
    # plt.show()
    frame_diffs = np.concatenate((frame_diffs, normalized_frames), axis=3)
    return frame_diffs, frame_rate


def detrend(signal, Lambda):
    """detrend(signal, Lambda) -> filtered_signal
    This function applies a detrending filter.
    This code is based on the following article "An advanced detrending method with application
    to HRV analysis". Tarvainen et al., IEEE Trans on Biomedical Engineering, 2002.
    *Parameters*
      ``signal`` (1d numpy array):
        The signal where you want to remove the trend.
      ``Lambda`` (int):
        The smoothing parameter.
    *Returns*
      ``filtered_signal`` (1d numpy array):
        The detrended signal.
    """
    signal_length = signal.shape[0]

    # observation matrix
    H = np.identity(signal_length)

    # second-order difference matrix

    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot((H - np.linalg.inv(H + (Lambda ** 2) * np.dot(D.T, D))), signal)
    return filtered_signal
