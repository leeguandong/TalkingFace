from os.path import dirname, join, basename, isfile
from tqdm import tqdm

from models import SyncNet_color as SyncNet
from models import Wav2Lip as Wav2Lip
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

parser = argparse.ArgumentParser(description='Code to train the Wav2Lip model without the visual quality discriminator')

parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset",
                    default="/home/imcs/local_disk/Wav2Lip/data/")  # required=True, type=str)
parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory',
                    default="/home/imcs/local_disk/Wav2Lip/exp_ckpts/601642_wav2lip")  # required=True, type=str)
parser.add_argument('--syncnet_checkpoint_path', help='Load the pre-trained Expert discriminator',
                    default="/home/imcs/local_disk/Wav2Lip/exp_ckpts/601642_syncnet/checkpoint_step000012800.pth")  # required=True, type=str)
parser.add_argument('--checkpoint_path', help='Resume from this checkpoint', default=None, type=str)

args = parser.parse_args()

global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

syncnet_T = 5
syncnet_mel_step_size = 16


class Dataset(object):
    def __init__(self, split):
        self.all_videos = get_image_list(args.data_root, split)

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def read_window(self, window_fnames):
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                # img = cv2.resize(img, (hparams.img_size, hparams.img_size))
                img = cv2.resize(img, (192, 288))  # from https://github.com/Rudrabha/Wav2Lip
                # 96,96
                # 192,288
                # 192,192
                # 288,288
            except Exception as e:
                return None

            window.append(img)

        return window

    def crop_audio_window(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame)  # 0-indexing ---> 1-indexing
        start_idx = int(80. * (start_frame_num / float(hparams.fps))) # 转到音频的起始点

        end_idx = start_idx + syncnet_mel_step_size  # 80/fps是帧移，syncnet_mel_step_size是帧长，

        return spec[start_idx: end_idx, :]

    def get_segmented_mels(self, spec, start_frame):
        mels = []
        assert syncnet_T == 5
        start_frame_num = self.get_frame_id(start_frame) + 1  # 0-indexing ---> 1-indexing
        if start_frame_num - 2 < 0: return None
        for i in range(start_frame_num, start_frame_num + syncnet_T):
            m = self.crop_audio_window(spec, i - 2)
            if m.shape[0] != syncnet_mel_step_size:
                return None
            mels.append(m.T)

        mels = np.asarray(mels)

        return mels

    def prepare_window(self, window):
        # 3 x T x H x W
        x = np.asarray(window) / 255.
        x = np.transpose(x, (3, 0, 1, 2))

        return x

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]
            img_names = list(glob(join(vidname, '*.jpg')))
            if len(img_names) <= 3 * syncnet_T:  # 总的数据小于15就不采样了
                continue

            img_name = random.choice(img_names)  # 选一帧输入帧和一帧参考帧，后续再取一个5帧的连续帧作为一个窗口
            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)

            window_fnames = self.get_window(img_name) # 视频取到5帧的连续帧作为一个窗口
            wrong_window_fnames = self.get_window(wrong_img_name)
            if window_fnames is None or wrong_window_fnames is None:
                continue

            window = self.read_window(window_fnames) # 进行图像缩放
            if window is None:
                continue

            wrong_window = self.read_window(wrong_window_fnames)
            if wrong_window is None:
                continue

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
            # 采取音频的梅尔频谱，读到全部的梅尔频谱，然后根据视频图片的id
            # 获取到梅尔频谱的起始位置，然后再取16帧梅尔频谱，16帧对应5张图，80维的梅尔频谱对应1s的图片的，训练时25帧，25帧
            # 5/25=0.2s 80x0.2=16 1s 25张图->80

            if (mel.shape[0] != syncnet_mel_step_size):
                continue

            # 计算l1损失时，取了5*16=80维的音频特征，视频侧维度是Bx6x5x96x96,音频侧是Bx5x1x80x16
            indiv_mels = self.get_segmented_mels(orig_mel.copy(), img_name)
            if indiv_mels is None: continue

            window = self.prepare_window(window)
            y = window.copy()
            window[:, :, window.shape[2] // 2:] = 0.

            wrong_window = self.prepare_window(wrong_window)
            x = np.concatenate([window, wrong_window], axis=0) # x是拼接了输入帧和参考帧

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)
            indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1)
            y = torch.FloatTensor(y)
            return x, indiv_mels, mel, y


def save_sample_images(x, g, gt, global_step, checkpoint_dir):
    x = (x.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    g = (g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    gt = (gt.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)

    refs, inps = x[..., 3:], x[..., :3]
    folder = join(checkpoint_dir, "samples_step{:09d}".format(global_step))
    if not os.path.exists(folder): os.mkdir(folder)
    collage = np.concatenate((refs, inps, g, gt), axis=-2)
    for batch_idx, c in enumerate(collage):
        for t in range(len(c)):
            cv2.imwrite('{}/{}_{}.jpg'.format(folder, batch_idx, t), c[t])


logloss = nn.BCELoss()


def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss


device = torch.device("cuda" if use_cuda else "cpu")
syncnet = SyncNet().to(device)

# # 多卡训练 DP
# syncnet = torch.nn.DataParallel(syncnet)

for p in syncnet.parameters():
    p.requires_grad = False

recon_loss = nn.L1Loss()


def get_sync_loss(mel, g):
    # TODO:
    # syncnet.eval()
    g = g[:, :, :, g.size(3) // 2:]
    g = torch.cat([g[:, :, i] for i in range(syncnet_T)], dim=1)
    # B, 3 * T, H//2, W
    # print("mel.device: ", mel.device)
    # print("g.device: ", g.device)
    a, v = syncnet(mel, g)
    y = torch.ones(g.size(0), 1).float().to(device)
    return cosine_loss(a, v, y)


def train(device, model, train_data_loader, test_data_loader, optimizer, lr_scheduler=None,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None):
    global global_step, global_epoch
    resumed_step = global_step

    best_loss = 100
    best_step = 0

    while global_epoch < nepochs:
        print('Starting Epoch: {}'.format(global_epoch))
        running_sync_loss, running_l1_loss = 0., 0.
        prog_bar = tqdm(enumerate(train_data_loader))
        for step, (x, indiv_mels, mel, gt) in prog_bar:
            model.train()
            optimizer.zero_grad()

            # Move data to CUDA device
            x = x.to(device)
            mel = mel.to(device)
            indiv_mels = indiv_mels.to(device)
            gt = gt.to(device)

            g = model(indiv_mels, x) # 5张图

            if hparams.syncnet_wt > 0.:
                sync_loss = get_sync_loss(mel, g)
            else:
                sync_loss = 0.

            l1loss = recon_loss(g, gt) # gt是输入图片

            loss = hparams.syncnet_wt * sync_loss + (1 - hparams.syncnet_wt) * l1loss
            loss.backward()
            optimizer.step()

            # lr decay
            if lr_scheduler is not None:
                # lr_scheduler.step(global_step) # is ok?
                if resumed_step > 0:  # resume
                    lr_scheduler.step(global_step)
                else:
                    lr_scheduler.step()

            if global_step % checkpoint_interval == 0:
                save_sample_images(x, g, gt, global_step, checkpoint_dir)

            global_step += 1
            cur_session_steps = global_step - resumed_step

            running_l1_loss += l1loss.item()
            if hparams.syncnet_wt > 0.:
                running_sync_loss += sync_loss.item()
            else:
                running_sync_loss += 0.

            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch)

            if global_step == 1 or global_step % hparams.eval_interval == 0:
                with torch.no_grad():
                    average_sync_loss = eval_model(test_data_loader, global_step, device, model, checkpoint_dir)

                    if average_sync_loss < .75:
                        hparams.set_hparam('syncnet_wt', 0.01)  # without image GAN a lesser weight is sufficient

                    if average_sync_loss <= best_loss:
                        best_loss = average_sync_loss
                        best_step = global_step

                        save_checkpoint(
                            model, optimizer, global_step, checkpoint_dir, global_epoch, is_best=True)
                    print("best_loss: ", best_loss)
                    print("best_step: ", best_step)
            prog_bar.set_description('L1: {}, Sync Loss: {}, lr: {}'.format(running_l1_loss / (step + 1),
                                                                            running_sync_loss / (step + 1),
                                                                            optimizer.param_groups[0]["lr"]))

        global_epoch += 1


def eval_model(test_data_loader, global_step, device, model, checkpoint_dir):
    eval_steps = 700
    print('Evaluating for {} steps'.format(eval_steps))
    sync_losses, recon_losses = [], []
    step = 0
    while 1:
        for x, indiv_mels, mel, gt in test_data_loader:
            # step += 1
            # print("step: ", step)
            model.eval()

            # Move data to CUDA device
            x = x.to(device)
            gt = gt.to(device)
            indiv_mels = indiv_mels.to(device)
            mel = mel.to(device)

            g = model(indiv_mels, x)

            # print("mel.shape: ", mel.shape)
            # print("g.shape: ", g.shape)

            sync_loss = get_sync_loss(mel, g)
            l1loss = recon_loss(g, gt)

            sync_losses.append(sync_loss.item())
            recon_losses.append(l1loss.item())

            if step > eval_steps:
                break
        averaged_sync_loss = sum(sync_losses) / len(sync_losses)
        averaged_recon_loss = sum(recon_losses) / len(recon_losses)

        print('L1: {}, Sync loss: {}'.format(averaged_recon_loss, averaged_sync_loss))

        return averaged_sync_loss


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch, is_best=False):
    if is_best:
        checkpoint_path = join(
            checkpoint_dir, "checkpoint_best.pth")
    else:
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


def load_checkpoint(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    # s = checkpoint["state_dict"]
    # new_s = {}
    # for k, v in s.items():
    #     new_s[k.replace('module.', '')] = v
    # model.load_state_dict(new_s)

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

    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    if overwrite_global_states:
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

    # Dataset and Dataloader setup
    train_dataset = Dataset('train_601642')
    test_dataset = Dataset('test_601642')

    # train_data_loader = data_utils.DataLoader(
    #     train_dataset, batch_size=hparams.batch_size, shuffle=True,
    #     num_workers=hparams.num_workers)

    # train_data_loader = data_utils.DataLoader(
    #     train_dataset, batch_size=24, shuffle=True,
    #     num_workers=8)

    # test_data_loader = data_utils.DataLoader(
    #     test_dataset, batch_size=24,
    #     num_workers=4)

    # 预加载加速
    train_data_loader = DataLoaderX(
        train_dataset, batch_size=10, shuffle=True,
        num_workers=4, drop_last=True)

    test_data_loader = DataLoaderX(
        test_dataset, batch_size=2,
        num_workers=2, drop_last=True)

    device = torch.device("cuda" if use_cuda else "cpu")

    #     import pdb;pdb.set_trace()
    # Model
    model = Wav2Lip().to(device)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    # 多卡训练 DP
    model = torch.nn.DataParallel(model)

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.initial_learning_rate)

    # lr_scheduler = WarmupPolyLR(optimizer, warmup_iters=500, max_iters=20000000)
    lr_scheduler = None

    if args.checkpoint_path is not None:
        # load_checkpoint(args.checkpoint_path, model, optimizer, reset_optimizer=False)
        load_checkpoint(args.checkpoint_path, model, optimizer, reset_optimizer=True)

    load_checkpoint(args.syncnet_checkpoint_path, syncnet, None, reset_optimizer=True, overwrite_global_states=False)

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    # Train!
    train(device, model, train_data_loader, test_data_loader, optimizer, lr_scheduler,
          checkpoint_dir=checkpoint_dir,
          checkpoint_interval=hparams.checkpoint_interval,
          nepochs=hparams.nepochs)

