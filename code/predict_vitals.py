
import argparse

import numpy as np
import scipy.io
from scipy.signal import butter
from matplotlib import pyplot as plt, ticker


from model import MTTS_CAN
from inference_preprocess import preprocess_raw_video, detrend


def predict_vitals(args):
    img_rows = 36
    img_cols = 36
    frame_depth = 10
    model_checkpoint = './mtts_can.hdf5'
    batch_size = args.batch_size
    fs = args.sampling_rate
    sample_data_path = args.video_path

    dXsub = preprocess_raw_video(sample_data_path, dim=36)
    print('dXsub shape', dXsub.shape)

    dXsub_len = (dXsub.shape[0] // frame_depth)  * frame_depth
    dXsub = dXsub[:dXsub_len, :, :, :]

    model = MTTS_CAN(frame_depth, 32, 64, (img_rows, img_cols, 3))
    model.load_weights(model_checkpoint)

    yptest = model.predict((dXsub[:, :, :, :3], dXsub[:, :, :, -3:]), batch_size=batch_size, verbose=1)

    pulse_pred = yptest[0]
    pulse_pred = detrend(np.cumsum(pulse_pred), 100)
    [b_pulse, a_pulse] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
    pulse_pred = scipy.signal.filtfilt(b_pulse, a_pulse, np.double(pulse_pred))

    resp_pred = yptest[1]
    resp_pred = detrend(np.cumsum(resp_pred), 100)
    [b_resp, a_resp] = butter(1, [0.08 / fs * 2, 0.5 / fs * 2], btype='bandpass')
    resp_pred = scipy.signal.filtfilt(b_resp, a_resp, np.double(resp_pred))

    ########## Plot ##################
    ax = plt.subplot(211)
    # fig, ax = plt.subplots()
    # ax.plot(ac[:2 * args.max_lag * downsample_rate])
    # repeat_msg = f"Non-repeating section length: {repeat_period}s"
    # ax.set(title=f'Auto-correlation {noise_category}\n{repeat_msg}', xlabel='Lag (secs)')
    # ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:d}'.format(int(x / downsample_rate)))
    # ax.xaxis.set_major_formatter(ticks_x)
    # if args.output_dir is not None:
    #     plt.savefig(f"{args.output_dir}/autocorrel.png")
    # if args.plot:
    #     plt.show()
    ax.plot(pulse_pred)
    ax.set(title='Pulse Prediction', xlabel='s')
    ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:d}'.format(int(x / args.sampling_rate)))
    ax.xaxis.set_major_formatter(ticks_x)
    plt.subplot(212)
    plt.plot(resp_pred)
    plt.title('Resp Prediction')
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, help='processed video path')
    parser.add_argument('--sampling_rate', type=int, default=30, help='sampling rate of your video')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size (multiplier of 10)')
    args = parser.parse_args()

    predict_vitals(args)

