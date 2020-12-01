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