import argparse
import torch
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image

import data
import data.dataset as module_dataset
import data.dataloader as module_dataloader
import model as model_module
from model import *
from parse_config import ConfigParser
from tester import *


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def _test_augment(image):
    img = image.transpose(1, 2, 0)
    img90 = np.array(np.rot90(img))
    img1 = np.concatenate([img[None], img90[None]])
    img2 = np.array(img1)[:, ::-1]
    img3 = np.concatenate([img1, img2])
    img4 = np.array(img3)[:, :, ::-1]
    img5 = np.concatenate([img3, img4]).transpose(0, 3, 1, 2)

    return torch.from_numpy(img5)


def _test_augment_pred(pred):
        pred_1 = pred[:,0,:,:].squeeze()
        pred_2 = pred[:,1,:,:].squeeze()
        #0channel
        pred1_1 = pred_1[:4] + pred_1[4:, :, ::-1]
        pred2_1 = pred1_1[:2] + pred1_1[2:, ::-1]
        pred3_1 = pred2_1[0] + np.rot90(pred2_1[1])[::-1, ::-1]
        pred_1 = pred3_1.copy()/8.
        #1channel
        pred1_2 = pred_2[:4] + pred_2[4:, :, ::-1]
        pred2_2 = pred1_2[:2] + pred1_2[2:, ::-1]
        pred3_2 = pred2_2[0] + np.rot90(pred2_2[1])[::-1, ::-1]
        pred_2 = pred3_2.copy()/8.

        pred_out = np.array([pred_1,pred_2])
        return pred_out


def main(config):
    datasets_root = Path("/home/data/xyj/RoadTracer_ori")
    image_root = datasets_root / "imagery"
    mask_root = datasets_root / "masks"
    output_root = Path("output")
    output_root.mkdir(parents=True, exist_ok=True)

    name_list = [f.stem for f in image_root.glob('*.png')]

    model = MHStackHourglass(block="GABasicBlock", heads="[2, 5]", depth=3, num_stacks=2, num_blocks=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(config.resume)
    model = model.to(device)
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        state_dict[k[len("module."):]] = state_dict[k]
        del state_dict[k]
    model.load_state_dict(state_dict)
    model.eval()

    for name in tqdm(name_list):
        image = cv2.imread(str(image_root/(name+'.png')), cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data = torch.from_numpy((image /255.0).transpose((2, 0, 1)).astype(np.float32))
        normalize = transforms.Normalize([0.396, 0.399, 0.370], [0.181, 0.174, 0.171])
        data = normalize(data).numpy()

        l = 512
        b = 72
        a = l-2*b

        c, H, W = data.shape
        label = np.zeros((H, W))

        W_num = (W-(l-b))//a+2 if (W-(l-b))//a else (W-(l-b))//a+1
        H_num = (H-(l-b))//a+2 if (H-(l-b))//a else (H-(l-b))//a+1

        for i in range(W_num):
            for j in range(H_num):
                x_min = l - 2 * b + (i - 1) * a if i > 0 else 0
                x_max = x_min + l

                y_min = l - 2 * b + (j - 1) * a if j > 0 else 0
                y_max = y_min + l

                # 越界的情况
                if x_max > W:
                    x_max = W
                    x_min = W - l

                    if xx_max != W:
                        xx_min = xx_max
                    xx_max = x_max

                else:
                    xx_min = x_min + b
                    xx_max = x_max - b

                if y_max > H:
                    y_max = H
                    y_min = H - l

                    yy_min = yy_max
                    yy_max = y_max

                else:
                    yy_min = y_min + b
                    yy_max = y_max - b

                if i==0:
                    xx_min = 0

                if j==0:
                    yy_min = 0
                img = data[:, y_min:y_max, x_min:x_max]

                image_augment = _test_augment(img).to(device)

                with torch.no_grad():
                    output_augment = model(image_augment)[0][-1]
                pred_augment = output_augment.cpu().numpy()
                out_l = _test_augment_pred(pred_augment)
                out_l = np.argmax(out_l, axis=0)
                label[yy_min:yy_max, xx_min:xx_max] = out_l[yy_min-y_min:yy_max-y_min, xx_min-x_min:xx_max-x_min].astype(np.int8)

        label = np.asarray(label*255, dtype=np.uint8)
        cv2.imwrite(str(output_root / (name+".png")), label)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str, required=True,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str, required=True,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-s', '--seed', default=1234, type=str)

    config = ConfigParser.from_args(args)
    main(config)