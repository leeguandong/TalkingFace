from os.path import dirname, join, basename, isfile
from tqdm import tqdm

from models import SyncNet_color as SyncNet
import audio

import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np

from glob import glob

import os, random, cv2, argparse
from hparams import hparams, get_image_list

from prefetch_generator import BackgroundGenerator, background  # 数据预加载

parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator')

parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset",
                    default="/home/imcs/local_disk/Wav2Lip/data/")  # required=True)
#
parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory',
                    default="/home/imcs/local_disk/Wav2Lip/exp_ckpts/601642_syncnet")  # required=True, type=str)
#
parser.add_argument('--checkpoint_path', help='Resumed from this checkpoint', default=None, type=str)
parser.add_argument('--pretrained_path', help='Resumed from this checkpoint', default=None, type=str)

args = parser.parse_args()

global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

syncnet_T = 5
syncnet_mel_step_size = 16


# 5/25=0.2s 80x0.2=16  1s 25张图->80    重叠部分

class Dataset(object):
    def __init__(self, split):
        self.all_videos = get_image_list(args.data_root, split)

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):  # 每次选5帧图片
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def crop_audio_window(self, spec, start_frame):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))

        end_idx = start_idx + syncnet_mel_step_size  # 帧长16

        return spec[start_idx: end_idx, :]

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]

            img_names = list(glob(join(vidname, '*.jpg')))
            if len(img_names) <= 3 * syncnet_T:
                continue

            # 参考视频和输入视频，如果取到输入视频，则y=1，因为音频和输入视频是对应的，如果取到是参考视频，和音频不对应，y=0,
            # 但是由于是随机取，因此输入视频可以被取到多次，但是永远达不到n平-n
            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)

            if random.choice([True, False]):
                y = torch.ones(1).float()  # 真
                chosen = img_name
            else:
                y = torch.zeros(1).float()  # 假
                chosen = wrong_img_name

            window_fnames = self.get_window(chosen)
            if window_fnames is None:
                continue

            window = []
            all_read = True
            for fname in window_fnames:
                img = cv2.imread(fname)
                if img is None:
                    all_read = False
                    break
                try:
                    # img = cv2.resize(img, (hparams.img_size, hparams.img_size))
                    img = cv2.resize(img, (192, 288))  # from https://github.com/Rudrabha/Wav2Lip
                    # 96,96
                    # 192,288
                    # 192,192
                    # 288,288
                except Exception as e:
                    all_read = False
                    break

                window.append(img)

            if not all_read: continue

            try:
                ########################################
                # wavpath = join(vidname, "audio.wav")
                # wav = audio.load_wav(wavpath, hparams.sample_rate)

                # orig_mel = audio.melspectrogram(wav).T
                ############### 加速 #############################
                wavnpy = join(vidname, "audio.npy")
                orig_mel = np.load(wavnpy)
            except Exception as e:
                continue

            mel = self.crop_audio_window(orig_mel.copy(), img_name)

            if (mel.shape[0] != syncnet_mel_step_size):
                continue

            # H x W x 3 * T
            x = np.concatenate(window, axis=2) / 255.
            x = x.transpose(2, 0, 1)
            x = x[:, x.shape[1] // 2:]

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)

            return x, mel, y


logloss = nn.BCELoss()


def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss


def train(device, model, train_data_loader, test_data_loader, optimizer, lr_scheduler,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None):
    global global_step, global_epoch
    resumed_step = global_step
    best_loss = 100
    best_step = 0

    while global_epoch < nepochs:
        running_loss = 0.
        prog_bar = tqdm(enumerate(train_data_loader))
        for step, (x, mel, y) in prog_bar:
            model.train()
            optimizer.zero_grad()

            # Transform data to CUDA device
            x = x.to(device)
            mel = mel.to(device)

            a, v = model(mel, x)
            y = y.to(device)

            loss = cosine_loss(a, v, y)
            loss.backward()
            optimizer.step()

            # lr decay
            if lr_scheduler is not None:
                # lr_scheduler.step(global_step) # is ok?
                if resumed_step > 0:  # resume
                    lr_scheduler.step(global_step)
                else:
                    lr_scheduler.step()

            global_step += 1
            cur_session_steps = global_step - resumed_step
            running_loss += loss.item()

            # if global_step == 1 or global_step % checkpoint_interval == 0:
            #     save_checkpoint(
            #         model, optimizer, global_step, checkpoint_dir, global_epoch)

            # 损失最小的将被保留
            if global_step % hparams.syncnet_eval_interval == 0:
                with torch.no_grad():
                    cur_loss = eval_model(test_data_loader, global_step, device, model, checkpoint_dir)
                if cur_loss <= best_loss:
                    best_loss = cur_loss
                    best_step = global_step
                    save_checkpoint(
                        model, optimizer, global_step, checkpoint_dir, global_epoch)
                print("best_loss: ", best_loss)
                print("best_step: ", best_step)
            prog_bar.set_description(
                'Loss: {}, lr: {}, global_step: {}'.format(running_loss / (step + 1), optimizer.param_groups[0]["lr"],
                                                           global_step))

        global_epoch += 1


def eval_model(test_data_loader, global_step, device, model, checkpoint_dir):
    eval_steps = 1400
    print('Evaluating for {} steps'.format(eval_steps))
    losses = []
    while 1:
        for step, (x, mel, y) in enumerate(test_data_loader):

            model.eval()

            # Transform data to CUDA device
            x = x.to(device)
            mel = mel.to(device)

            a, v = model(mel, x)
            y = y.to(device)

            loss = cosine_loss(a, v, y)
            losses.append(loss.item())

            if step > eval_steps: break

        averaged_loss = sum(losses) / len(losses)
        print(averaged_loss)

        return averaged_loss


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):
    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}.pth".format(global_step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    if torch.cuda.device_count() > 1:
        mode_state_dict = model.module.state_dict()
    else:
        mode_state_dict = model.state_dict()
    torch.save({
        "state_dict": mode_state_dict,
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)


def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_checkpoint(path, model, optimizer, reset_optimizer=False):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model_state_dict = model.state_dict()
    for k, v in checkpoint["state_dict"].items():
        if k in model_state_dict:
            name = k
        elif "module." + k in model_state_dict:
            name = "module." + k
        else:
            name = k.replace("module.", "")  # remove `module.`
        model_state_dict[name] = v
    model.load_state_dict(model_state_dict, strict=True)

    # model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]

    return model


class WarmupPolyLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, target_lr=0, max_iters=0, power=0.9, warmup_factor=1.0 / 3,
                 warmup_iters=500, warmup_method="linear", last_epoch=-1, **kwargs):
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted "
                "got {}".format(warmup_method))

        self.target_lr = target_lr
        self.max_iters = max_iters
        self.power = power
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method

        super(WarmupPolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        N = self.max_iters - self.warmup_iters
        T = self.last_epoch - self.warmup_iters
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            else:
                raise ValueError("Unknown warmup type.")
            return [self.target_lr + (base_lr - self.target_lr) * warmup_factor for base_lr in self.base_lrs]
        factor = pow(1 - T / N, self.power)
        return [self.target_lr + (base_lr - self.target_lr) * factor for base_lr in self.base_lrs]


# DataLoaderX
class DataLoaderX(data_utils.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


if __name__ == "__main__":
    checkpoint_dir = args.checkpoint_dir
    checkpoint_path = args.checkpoint_path
    pretrained_path = args.pretrained_path

    if not os.path.exists(checkpoint_dir): os.mkdir(checkpoint_dir)

    # Dataset and Dataloader setup
    train_dataset = Dataset('train_601642')
    test_dataset = Dataset('test_601642')

    # train_data_loader = data_utils.DataLoader(
    #     train_dataset, batch_size=hparams.syncnet_batch_size, shuffle=True,
    #     num_workers=hparams.num_workers)

    # train_data_loader = data_utils.DataLoader(
    #     train_dataset, batch_size=hparams.syncnet_batch_size, shuffle=True,
    #     num_workers=8)

    # test_data_loader = data_utils.DataLoader(
    #     test_dataset, batch_size=hparams.syncnet_batch_size,
    #     num_workers=4)

    # 预加载加速
    train_data_loader = DataLoaderX(
        train_dataset, batch_size=hparams.syncnet_batch_size, shuffle=True,
        num_workers=4)

    test_data_loader = DataLoaderX(
        test_dataset, batch_size=hparams.syncnet_batch_size,
        num_workers=4)

    device = torch.device("cuda" if use_cuda else "cpu")

    # Model
    model = SyncNet().to(device)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    #     import pdb;pdb.set_trace()
    # 多卡训练 DP
    model = torch.nn.DataParallel(model)

    if pretrained_path is not None:
        print("Load pretrained from: {}".format(pretrained_path))
        pretrained_state_dict = _load(pretrained_path)
        model_state_dict = model.state_dict()
        for k, v in pretrained_state_dict["state_dict"].items():
            if k in model_state_dict:
                name = k
            elif "module." + k in model_state_dict:
                name = "module." + k
            else:
                name = k.replace("module.", "")  # remove `module.`
            model_state_dict[name] = v
        model.load_state_dict(model_state_dict, strict=True)

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.syncnet_lr)

    # lr_scheduler = WarmupPolyLR(optimizer, warmup_iters=500, max_iters=200000)
    lr_scheduler = None

    if checkpoint_path is not None:
        # load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False)
        load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=True)

    train(device, model, train_data_loader, test_data_loader, optimizer, lr_scheduler,
          checkpoint_dir=checkpoint_dir,
          checkpoint_interval=hparams.syncnet_checkpoint_interval,
          nepochs=hparams.nepochs)
