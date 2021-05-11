from __future__ import print_function
from models.litmodel import LitModel
from dataloader import lit_custom_data
from runstats import Statistics

import cv2
import timeit
import numpy as np

avg_iou = []
avg_fps = []


def test(model, iterator):
    verbose = True
    iouStats = Statistics()
    fpsStats = Statistics()

    fourcc = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
    outwrite = cv2.VideoWriter('output.mp4', fourcc, 60.0, (500, 500))

    model.freeze()
    model.eval()
    its = 0
    for i, batch in enumerate(iterator):

        image = batch['input'].cuda()
        target = batch['target']
        real = batch['real']

        its += 1
        clear = True if its % 10 == 0 else False
        # clear = True if np.random.randn() > 0.9 else False
        start = timeit.default_timer()
        out = model(image)
        end = timeit.default_timer()

        t = target.detach().cpu().squeeze().numpy()
        o = out.detach().cpu().squeeze().numpy() > 0.5

        iou = np.sum(np.bitwise_and(t.astype(bool), o.astype(bool))) / \
            np.sum(np.bitwise_or(t.astype(bool), o.astype(bool)))
        fpsStats.push(1/(end-start))
        iouStats.push(iou)

        if verbose:
            print(
            f"frametime: {(end-start)*1000}ms, iou: {iou} avg: {np.mean(np.array(avg_iou))}, avgfps:{np.mean(np.array(avg_fps))}")
            img = np.transpose(
                (real*255)[0].numpy().astype("uint8"), (1, 2, 0))
            overlay = np.zeros(img.squeeze().shape)
            overlay[:, :, 2] = o
            overlay[:, :, 0] = t
            overlay = (overlay*255).astype("uint8")
            out = cv2.addWeighted(img, 1, overlay, 0.5, 0)
            out = cv2.resize(out, (500, 500), interpolation=cv2.INTER_AREA)
            outwrite.write(out)

            # cv2.imshow('Out', out)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

    outwrite.release()
    cv2.destroyAllWindows()

    print("||STATS||")
    print('Avg fps', fpsStats.mean(), '+-', fpsStats.variance())
    print('Avg IoU', iouStats.mean(), '+-', iouStats.variance())
    print("Output Video Stored as output.mp4")


# class LightningMNISTClassifier(pl.LightningModule):
if __name__ == '__main__':
    hparams = {
        'lr': 0.01
    }
    model = LitModel.load_from_checkpoint(
        r"I:\dataset\cv\trained\Resnet_LSTM_FCN_2\checkpoints\epoch=119-step=3839.ckpt").cuda()

    dataset = lit_custom_data()

    dataset.setup()
    test(model, dataset.test_dataloader())
