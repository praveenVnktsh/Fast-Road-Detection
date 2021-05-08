from __future__ import print_function
from models.litmodel import LitModel
from models.fcn32s import FCN32s
from models.vgg import VGGNet

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
from torchvision.models.vgg import VGG
import pytorch_lightning as pl
from torch.optim.rmsprop import RMSprop


from dataloader import CustomDataset, lit_custom_data
from pytorch_lightning import loggers
from configs import Configs

from icecream import ic


import os

import cv2
import numpy as np

import timeit

avg_iou = []
avg_fps = []


def test(iterator):
    verbose = False

    model.freeze()
    model.eval()
    for i, batch in enumerate(iterator):
        # if i % 2 == 0:
        #     continue

        image = batch['input'].cuda()
        target = batch['target']
        real = batch['real']

        start = timeit.default_timer()
        out = model(image)
        end = timeit.default_timer()

        t = target.detach().cpu().squeeze().numpy()
        o = out.detach().cpu().squeeze().numpy() > 0.5

        iou = np.sum(np.bitwise_and(t.astype(bool), o.astype(bool))) / \
            np.sum(np.bitwise_or(t.astype(bool), o.astype(bool)))
        avg_iou.append(iou)
        avg_fps.append(1/(end-start))

        print(
            f"frametime: {(end-start)*1000}ms, iou: {iou} avg: {np.mean(np.array(avg_iou))}, avgfps:{np.mean(np.array(avg_fps))}")

        if verbose:
            img = np.transpose(
                (real*255)[0].numpy().astype("uint8"), (1, 2, 0))
            overlay = np.zeros(img.squeeze().shape)
            overlay[:, :, 2] = o
            overlay[:, :, 0] = t
            overlay = (overlay*255).astype("uint8")
            out = cv2.addWeighted(img, 1, overlay, 0.5, 0)
            out = cv2.resize(out, (500, 500), interpolation=cv2.INTER_AREA)

            cv2.imshow('Out', out)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

    print("||STATS||")
    ic(np.mean(np.array(avg_fps)))
    ic(np.mean(np.array(avg_iou)))


# class LightningMNISTClassifier(pl.LightningModule):
if __name__ == '__main__':
    hparams = {
        'lr': 0.01
    }
    model = LitModel.load_from_checkpoint(
        "lightning_logs\Resnet_LSTM_FCN_2\checkpoints\epoch=119-step=3839.ckpt").cuda()

    dataset = lit_custom_data()
    # trainer = pl.Trainer(gpus=1, max_epochs=120)
    # trainer.fit(model, dataset)

    dataset.setup()
    test(dataset.test_dataloader())
    # trainer = pl.Trainer(gpus=1, )
    # trainer.test(model, dataset.test_dataloader())

    print("hello")
