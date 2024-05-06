import librosa
import numpy as np
from scipy import signal
import subprocess
from pathlib import Path
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--input_audio", help="Input Audio path", required=True)
parser.add_argument("--save_dir", help="Save dir path", required=True)
parser.add_argument('--chunk_size', help='Audio Chunk size', default=30, type=int)

args = parser.parse_args()


def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]


def melspectrogram(wav):
    D = _stft(preemphasis(wav, 0.97, True))
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - 20
    return _normalize(S)


def _linear_to_mel(spectogram):
    _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectogram)


def _build_mel_basis():
    return librosa.filters.mel(16000, 800, n_mels=80, fmin=55, fmax=7600)


def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav


def _normalize(S):
    return np.clip((2 * 4) * ((S + 100) / 100) - 4, -4, 4)


def _amp_to_db(x):
    min_level = np.exp(-100 / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _stft(y):
    return librosa.stft(y=y, n_fft=800, hop_length=200, win_length=800)


def load_audio(audio_path, save_dir, chunk_size=30):
    if os.path.exists(save_dir):
        print("Error: save_dir exist")
        raise OSError

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=False, parents=True)

    if not audio_path.endswith('.wav'):
        print('Extracting raw audio...')
        command = 'ffmpeg -y -i {} -strict -2 {}'.format(audio_path, './temp.wav')

        subprocess.call(command, shell=True)
        audio_path = './temp.wav'

    wav = load_wav(audio_path, 16000)
    mel = melspectrogram(wav)
    print("mel.shape: ", mel.shape)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    # 采样梅尔频谱块
    mel_chunks = []
    mel_step_size = 16
    fps = 30
    print("fps:", fps)
    # 计算采样步长
    mel_idx_multiplier = 80. / fps
    print("mel_idx_multiplier: ", mel_idx_multiplier)
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier) # 采样步长，计算出采样步长之后，在梅尔频谱中进行16k的采样
        #             print("start_idx: ", start_idx)
        #             print("start_idx + mel_step_size: ", start_idx + mel_step_size)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
        i += 1

    print("Length of mel chunks: {}".format(len(mel_chunks)))

    full_length = len(mel_chunks)
    # 分段
    chunks_list = []
    for i in range(full_length):
        start_idx = i * chunk_size # 每一个chunk对应30张图，实际上还是
        if start_idx + chunk_size > full_length:
            chunks_list.append(mel_chunks[start_idx: full_length])
            break
        else:
            chunks_list.append(mel_chunks[start_idx: start_idx + chunk_size])

    print("Number of Chunks: ", len(chunks_list))

    for i in range(len(chunks_list)):
        save_path = str(save_dir.joinpath("chunk_" + str(i) + ".npy"))
        np.save(save_path, chunks_list[i])
    print("Finished Process Video Chunks !")


if __name__ == '__main__':
    load_audio(args.input_audio, args.save_dir)

    ##### 合成片段命令 #####
    ### 合并视频片段 ###
    # ffmpeg -f concat -i ./splits_cat.txt -c copy res_cat_splits.mp4
    ### 添加音频 ###
    # ffmpeg -i ./temp.wav -i ./res_cat_splits.mp4 -strict -2 -q:v 1 final_video.mp4
