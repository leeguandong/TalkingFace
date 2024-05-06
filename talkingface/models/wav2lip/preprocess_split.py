import os
import random
import shutil


def split_folders():
    # 调用函数进行划分
    directory = '/home/imcs/local_disk/Wav2Lip/data/data_601642'
    train_ratio = 0.8

    # 获取目录下的文件夹列表
    folder_list = [folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]

    # 随机打乱文件夹列表
    random.shuffle(folder_list)

    # 计算划分索引
    split_index = int(len(folder_list) * train_ratio)

    # 划分为训练集和测试集
    train_folders = folder_list[:split_index]
    test_folders = folder_list[split_index:]

    # 创建train.txt和test.txt文件
    with open("/home/imcs/local_disk/Wav2Lip/filelists/train_601642.txt", "w") as train_file:
        for folder in train_folders:
            parent_directory = os.path.basename(os.path.normpath(directory))
            train_file.write(f"{parent_directory}/{folder}\n")

    with open("/home/imcs/local_disk/Wav2Lip/filelists/test_601642.txt", "w") as test_file:
        for folder in test_folders:
            parent_directory = os.path.basename(os.path.normpath(directory))
            test_file.write(f"{parent_directory}/{folder}\n")

    print("文件夹已划分为训练集和测试集，并写入train.txt和test.txt文件！")


if __name__ == "__main__":
    split_folders()
