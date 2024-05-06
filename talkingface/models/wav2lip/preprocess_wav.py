from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
import audio


def process_video_file(img_path):
    try:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (192, 192))  # from https://github.com/Rudrabha/Wav2Lip
        base_name = str(img_path)[:-4]
        np.save(base_name, img)
    except:
        print("frame process error! img_path: " + str(img_path))


def process_audio_file(wav_path):
    try:
        wav = audio.load_wav(wav_path, 16000)
        orig_mel = audio.melspectrogram(wav).T
        base_name = str(wav_path)[:-4]
        np.save(base_name, orig_mel)
    except:
        print("audio process error! wav_path: " + str(wav_path))


def main():
    # data_types = ["mingren", "cjhs", "zss", "fandeng", "zhangxuefeng", "speak", "silent_1"]
    data_types = [
        "biliUP_add", "biliUP", "sftv1_crop", "0707_speak", "0707_silent_1", "speak", "silent_1",
        "kaoyan", "stars", "zhangxuefeng", "mingren", "cjhs", "zss", "fandeng", "ysxw", "cjg_wenwen"]

    for data_type in tqdm(data_types):
        preprocessed_root = Path("/home/imcs/block_disk/datasets/wav2lip/" + data_type + "_preprocessed")

        print('Started processing for {}'.format(preprocessed_root))

        print("=====processing Audio=====")
        wav_paths = []
        for wav_path in preprocessed_root.rglob("*.wav"):
            wav_paths.append(str(wav_path))

        for wav_path in tqdm(wav_paths):
            process_audio_file(wav_path)

    # print("=====processing Frame=====")
    # img_paths = []
    # for img_path in preprocessed_root.rglob("*.jpg"):
    # 	img_paths.append(str(img_path))

    # for img_path in tqdm(img_paths):
    # 	process_video_file(img_path)


if __name__ == '__main__':
    main()
