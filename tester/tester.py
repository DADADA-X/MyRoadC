import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from base import BaseEval
from model.loss import *
from model.metrics import *


class SegmentEval(BaseEval):
    def __init__(self, model, config, data_loader, output_dir):
        super(SegmentEval, self).__init__(model, config)
        self.data_loader = data_loader

        self.criterion = soft_iou_loss
        self.metric_ftns = [rIoU]

        self.total_metrics = torch.zeros(len(self.metric_ftns))

        self.output_dir = output_dir
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def test(self):
        self.model.eval()

        with torch.no_grad():
            for i, data in enumerate(tqdm(self.data_loader)):
                image_name, image, mask = data.values()
                image = image.to(self.device)
                mask = mask.to(self.device)

                output = self.model(image)

                b, c, w, h = output.shape

                # save smple images
                predicted_prob = torch.argmax(output, dim=1)
                for i in range(b):
                    predicted_prob_ = predicted_prob[i].squeeze().cpu().numpy() * 255
                    predicted_prob_ = np.asarray(predicted_prob_, dtype=np.uint8)
                    fig, ax = plt.subplots(1, 3)
                    ax[0].imshow(image.cpu().numpy()[0].transpose(1, 2, 0))
                    ax[1].imshow(mask.cpu()[0][0])
                    ax[2].imshow(predicted_prob_)
                    plt.show()
                    # cv2.imwrite(str(self.output_dir / (image_name[i] + '.png')), predicted_prob_)

                loss = self.criterion(output, mask)
                for i, metric in enumerate(self.metric_ftns):
                    self.total_metrics[i] += metric(output, mask) * b

        n_samples = len(self.data_loader.sampler)
        log = {}
        log.update({
            met.__name__: self.total_metrics[i].item() / n_samples for i, met in enumerate(self.metric_ftns)
        })

        self.logger.info(log)


class MTLEval(BaseEval):
    def __init__(self, model, config, data_loader, output_dir):
        super(MTLEval, self).__init__(model, config)
        self.data_loader = data_loader

        self.criterion_mask = soft_iou_loss
        self.criterion_conn = balanced_ce_loss

        self.metric_ftns_mask = [rIoU]
        self.metric_ftns_conn = [mIoU]

        self.total_metrics = torch.zeros(len(self.metric_ftns_mask))

        self.output_dir = output_dir
        for cat in ['mask', 'conn']:
            output_dir_ = output_dir / cat
            if not output_dir_.exists():
                output_dir_.mkdir(parents=True, exist_ok=True)

    def test(self):
        self.model.eval()
        a = []
        with torch.no_grad():
            for i, data in enumerate(tqdm(self.data_loader)):
                image_name, image, mask, conn = data.values()
                image = image.to(self.device)
                mask = mask.to(self.device)
                conn = conn.to(self.device)

                output1, output2 = self.model(image)

                b, c, w, h = output1.shape

                # save smple images
                prob_m = torch.sigmoid(output1)
                prob_c = torch.argmax(output2, dim=1)
                for i in range(b):
                    prob_m_ = prob_m[i].squeeze().cpu().numpy() * 255
                    prob_m_ = np.asarray(prob_m_, dtype=np.uint8)

                    prob_c_ = prob_c[i].squeeze().cpu().numpy()
                    # plt.imshow(prob_c_, cmap='gray')
                    # plt.show()
                    from collections import Counter
                    a.extend(Counter(prob_c_.flatten()).keys())

                    # cv2.imwrite(str(self.output_dir / "mask" / (image_name[i] + '.png')), prob_m_)
                    # cv2.imwrite(str(self.output_dir / "conn" / (image_name[i] + '.png')), prob_l_)

                for i, metric in enumerate(self.metric_ftns_mask):
                    self.total_metrics[i] += metric(output1, mask) * b  # TODO mask only now
        print(set(a))
        n_samples = len(self.data_loader.sampler)
        log = {}
        log.update({
                met.__name__: self.total_metrics[i].item() / n_samples for i, met in enumerate(self.metric_ftns_mask)
            })

        self.logger.info(log)