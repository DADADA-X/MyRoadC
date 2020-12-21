import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter

from base import BaseEval
from model.loss import *
from model.metrics import *


class SegmentEval(BaseEval):
    def __init__(self, model, config, data_loader, output_dir):
        super(SegmentEval, self).__init__(model, config)
        self.data_loader = data_loader

        self.criterion = soft_iou_loss
        self.metric_ftns = [rIoU, relaxed_IoU]

        self.total_loss = torch.tensor(0.).to(self.device)
        self.total_metrics = torch.zeros(len(self.metric_ftns)).to(self.device)

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

                # save smple images
                predicted_prob = torch.argmax(output, dim=1)
                predicted_prob_ = predicted_prob.squeeze().cpu().numpy() * 255
                predicted_prob_ = np.asarray(predicted_prob_, dtype=np.uint8)

                # plot
                # plt.imshow(predicted_prob_)
                # plt.show()

                # save
                cv2.imwrite(str(self.output_dir / (image_name[0] + '.png')), predicted_prob_)

                self.total_loss += self.criterion(output, mask)
                for i, metric in enumerate(self.metric_ftns):
                    self.total_metrics[i] += metric(output, mask)

        n_samples = len(self.data_loader.sampler)
        self.logger.info("loss: {:.4f}".format(self.total_loss.item() / n_samples))
        for i, metric in enumerate(self.metric_ftns):
            self.logger.info("{}: {:.2f}%".format(metric.__name__, self.total_metrics[i].item() / n_samples * 100))


class DSegmentEval(BaseEval):
    def __init__(self, model, config, data_loader, output_dir):
        super(DSegmentEval, self).__init__(model, config)
        self.data_loader = data_loader

        self.criterion = dice_bce_loss
        self.metric_ftns = [IoU, relaxed_IoU]

        self.total_loss = torch.tensor(0.).to(self.device)
        self.total_metrics = torch.zeros(len(self.metric_ftns)).to(self.device)

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

                # save smple images
                predicted_prob = torch.sigmoid(output) > 0.5
                predicted_prob_ = predicted_prob.squeeze().cpu().numpy() * 255
                predicted_prob_ = np.asarray(predicted_prob_, dtype=np.uint8)

                # plot
                # plt.imshow(predicted_prob_)
                # plt.show()

                # save
                cv2.imwrite(str(self.output_dir / (image_name[0] + '.png')), predicted_prob_)

                self.total_loss += self.criterion(output, mask)
                for i, metric in enumerate(self.metric_ftns):
                    self.total_metrics[i] += metric(output, mask)

        n_samples = len(self.data_loader.sampler)
        self.logger.info("loss: {:.4f}".format(self.total_loss.item() / n_samples))
        for i, metric in enumerate(self.metric_ftns):
            self.logger.info("{}: {:.2f}%".format(metric.__name__, self.total_metrics[i].item() / n_samples * 100))


class MTLEval(BaseEval):
    def __init__(self, model, config, data_loader, output_dir):
        super(MTLEval, self).__init__(model, config)
        self.data_loader = data_loader

        self.criterion_mask = soft_iou_loss
        self.criterion_conn = balanced_ce_loss

        self.metric_ftns_mask = [rIoU, relaxed_IoU]
        self.metric_ftns_conn = mIoU

        self.total_loss_mask = torch.tensor(0.).to(self.device)
        self.total_loss_conn = torch.tensor(0.).to(self.device)
        self.total_metrics_mask = torch.zeros(len(self.metric_ftns_mask)).to(self.device)
        self.total_metrics_conn = torch.tensor(0.).to(self.device)

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

                # save smple images
                prob_m = torch.argmax(output1, dim=1)
                prob_m_ = prob_m.squeeze().cpu().numpy() * 255
                prob_m_ = np.asarray(prob_m_, dtype=np.uint8)

                prob_c = torch.argmax(output2, dim=1)
                prob_c_ = prob_c.squeeze().cpu().numpy()
                prob_c_ = np.asarray(prob_c_, dtype=np.uint8)

                # plot
                # fig, ax = plt.subplots(1, 3)
                # ax[0].imshow(mask.cpu()[0][0])
                # ax[1].imshow(prob_m_)
                # ax[2].imshow(prob_c_)
                # plt.show()

                # stat conn
                a.extend(Counter(prob_c_.flatten()).keys())

                # save
                cv2.imwrite(str(self.output_dir / "mask" / (image_name[0] + '.png')), prob_m_)
                # cv2.imwrite(str(self.output_dir / "conn" / (image_name[0] + '.png')), prob_c_)

                self.total_loss_mask += self.criterion_mask(output1, mask)
                self.total_loss_conn += self.criterion_conn(output2, conn)
                for i, metric in enumerate(self.metric_ftns_mask):
                    self.total_metrics_mask[i] += metric(output1, mask)
                self.total_metrics_conn += self.metric_ftns_conn(output2, conn)

        print(set(a))

        n_samples = len(self.data_loader.sampler)
        self.logger.info("loss_mask: {:.4f}".format(self.total_loss_mask.item() / n_samples))
        self.logger.info("loss_conn: {:.4f}".format(self.total_loss_conn.item() / n_samples))
        for i, metric in enumerate(self.metric_ftns_mask):
            self.logger.info("{}: {:.2f}%".format(metric.__name__, self.total_metrics_mask[i].item() / n_samples * 100))
        self.logger.info("IoU_conn: {:.2f}%".format( self.total_metrics_conn.item() / n_samples * 100))


class HGSegmentEval(BaseEval):
    def __init__(self, model, config, data_loader, output_dir):
        super(HGSegmentEval, self).__init__(model, config)
        self.data_loader = data_loader

        self.criterion = soft_iou_loss
        self.metric_ftns = [rIoU, relaxed_IoU]

        self.total_loss = torch.tensor(0.).to(self.device)
        self.total_metrics = torch.zeros(len(self.metric_ftns)).to(self.device)

        self.output_dir = output_dir
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def test(self):
        self.model.eval()

        with torch.no_grad():
            for i, data in enumerate(tqdm(self.data_loader)):
                image_name, image, mask, _ = data.values()
                image = image.to(self.device)
                mask = mask.to(self.device)

                outputs = self.model(image)

                # save smple images
                predicted_prob = torch.argmax(outputs[-1], dim=1)
                predicted_prob_ = predicted_prob.squeeze().cpu().numpy() * 255
                predicted_prob_ = np.asarray(predicted_prob_, dtype=np.uint8)

                # plot
                # fig, ax = plt.subplots(1, 3)
                # ax[0].imshow(image.cpu().numpy()[0].transpose(1, 2, 0))
                # ax[1].imshow(mask.cpu()[0][0])
                # ax[2].imshow(predicted_prob_)
                # plt.show()

                # save
                cv2.imwrite(str(self.output_dir / (image_name[0] + '.png')), predicted_prob_)

                self.total_loss += self.criterion(outputs[-1], mask)
                for i, metric in enumerate(self.metric_ftns):
                    self.total_metrics[i] += metric(outputs[-1], mask)

        n_samples = len(self.data_loader.sampler)
        self.logger.info("loss: {:.4f}".format(self.total_loss.item() / n_samples))
        for i, metric in enumerate(self.metric_ftns):
            self.logger.info("{}: {:.2f}%".format(metric.__name__, self.total_metrics[i].item() / n_samples * 100))


class HGMTLEval(BaseEval):
    def __init__(self, model, config, data_loader, output_dir):
        super(HGMTLEval, self).__init__(model, config)
        self.data_loader = data_loader

        self.criterion_mask = soft_iou_loss
        self.criterion_conn = balanced_ce_loss

        self.metric_ftns_mask = [rIoU, relaxed_IoU]
        self.metric_ftns_conn = mIoU

        self.total_loss_mask = torch.tensor(0.).to(self.device)
        self.total_loss_conn = torch.tensor(0.).to(self.device)
        self.total_metrics_mask = torch.zeros(len(self.metric_ftns_mask)).to(self.device)
        self.total_metrics_conn = torch.tensor(0.).to(self.device)

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
                image_name, image, mask, _, conn, _ = data.values()
                image = image.to(self.device)
                mask = mask.to(self.device)
                conn = conn.to(self.device)

                if image_name[0] == 'RGB-PanSharpen_AOI_2_Vegas_img100_2':
                    print('lalala')

                output1, output2 = self.model(image)

                # save smple images
                prob_m = torch.argmax(output1[-1], dim=1)
                prob_m_ = prob_m.squeeze().cpu().numpy() * 255
                prob_m_ = np.asarray(prob_m_, dtype=np.uint8)

                prob_c = torch.argmax(output2[-1], dim=1)
                prob_c_ = prob_c.squeeze().cpu().numpy()
                prob_c_ = np.asarray(prob_c_, dtype=np.uint8)

                # plot
                # fig, ax = plt.subplots(1, 4)
                # ax[0].imshow(image.cpu().numpy()[0].transpose(1, 2, 0))
                # ax[0].axis('off')
                # ax[1].imshow(mask.cpu()[0][0], cmap='gray')
                # ax[1].axis('off')
                # ax[2].imshow(np.logical_not(prob_m_), cmap='gray')
                # ax[2].axis('off')
                # ax[3].imshow(prob_c_, cmap='jet')
                # ax[3].axis('off')
                # plt.show()

                # stat conn
                a.extend(Counter(prob_c_.flatten()).keys())

                cv2.imwrite(str(self.output_dir / "mask" / (image_name[0] + '.png')), prob_m_)
                # cv2.imwrite(str(self.output_dir / "conn" / (image_name[0] + '.png')), prob_m_)

                self.total_loss_mask += self.criterion_mask(output1[-1], mask)
                self.total_loss_conn += self.criterion_conn(output2[-1], conn)
                for i, metric in enumerate(self.metric_ftns_mask):
                    self.total_metrics_mask[i] += metric(output1[-1], mask)
                self.total_metrics_conn += self.metric_ftns_conn(output2[-1], conn)

        print(set(a))

        n_samples = len(self.data_loader.sampler)
        self.logger.info("loss_mask: {:.4f}".format(self.total_loss_mask.item() / n_samples))
        self.logger.info("loss_conn: {:.4f}".format(self.total_loss_conn.item() / n_samples))
        for i, metric in enumerate(self.metric_ftns_mask):
            self.logger.info("{}: {:.2f}%".format(metric.__name__, self.total_metrics_mask[i].item() / n_samples * 100))
        self.logger.info("IoU_conn: {:.2f}%".format( self.total_metrics_conn.item() / n_samples * 100))


class MTLEval3(BaseEval):
    def __init__(self, model, config, data_loader, output_dir):
        super(MTLEval3, self).__init__(model, config)
        self.data_loader = data_loader

        self.criterion_mask = soft_iou_loss
        self.criterion_cline = mse_loss
        self.criterion_point = mse_loss

        self.metric_ftns_mask = [rIoU, relaxed_IoU]
        self.metric_ftns_cline = MSE
        self.metric_ftns_point = MSE

        self.total_loss_mask = torch.tensor(0.).to(self.device)
        self.total_loss_cline = torch.tensor(0.).to(self.device)
        self.total_loss_point = torch.tensor(0.).to(self.device)
        self.total_metrics_mask = torch.zeros(len(self.metric_ftns_mask)).to(self.device)
        self.total_metrics_cline = torch.tensor(0.).to(self.device)
        self.total_metrics_point = torch.tensor(0.).to(self.device)

        self.output_dir = output_dir
        for cat in ['mask', 'cline', 'point']:
            output_dir_ = output_dir / cat
            if not output_dir_.exists():
                output_dir_.mkdir(parents=True, exist_ok=True)

    def test(self):
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(tqdm(self.data_loader)):
                image_name, image, mask, cline, point = data.values()
                image = image.to(self.device)
                mask = mask.to(self.device)
                cline = cline.to(self.device)
                point = point.to(self.device)

                output1, output2, output3 = self.model(image)

                # save smple images
                prob_m = torch.argmax(output1, dim=1)
                prob_m_ = prob_m.squeeze().cpu().numpy() * 255
                prob_m_ = np.asarray(prob_m_, dtype=np.uint8)

                prob_c = torch.sigmoid(output2)
                prob_c_ = prob_c.squeeze().cpu().numpy() * 255
                prob_c_ = np.asarray(prob_c_, dtype=np.uint8)

                prob_p = torch.sigmoid(output3)
                prob_p_ = prob_p.squeeze().cpu().numpy() * 255
                prob_p_ = np.asarray(prob_p_, dtype=np.uint8)

                # plot
                # fig, ax = plt.subplots(1, 3)
                # ax[0].imshow(prob_m_)
                # ax[1].imshow(prob_c_)
                # ax[2].imshow(prob_p_)
                # plt.show()

                # save
                cv2.imwrite(str(self.output_dir / "mask" / (image_name[0] + '.png')), prob_m_)
                # cv2.imwrite(str(self.output_dir / "cline" / (image_name[0] + '.png')), prob_c_)
                # cv2.imwrite(str(self.output_dir / "point" / (image_name[0] + '.png')), prob_p_)

                self.total_loss_mask += self.criterion_mask(output1, mask)
                self.total_loss_cline += self.criterion_cline(output2, cline)
                self.total_loss_point += self.criterion_point(output3, point)
                for i, metric in enumerate(self.metric_ftns_mask):
                    self.total_metrics_mask[i] += metric(output1, mask)
                self.total_metrics_cline += self.metric_ftns_cline(output2, cline)
                self.total_metrics_point += self.metric_ftns_point(output3, point)

        n_samples = len(self.data_loader.sampler)
        self.logger.info("loss_mask: {:.4f}".format(self.total_loss_mask.item() / n_samples))
        self.logger.info("loss_cline: {:.4f}".format(self.total_loss_cline.item() / n_samples))
        self.logger.info("loss_point: {:.4f}".format(self.total_loss_point.item() / n_samples))
        for i, metric in enumerate(self.metric_ftns_mask):
            self.logger.info("{}: {:.2f}%".format(metric.__name__, self.total_metrics_mask[i].item() / n_samples * 100))
        self.logger.info("MSE_cline: {:.2f}".format( self.total_metrics_cline.item() / n_samples ))
        self.logger.info("MSE_point: {:.2f}".format( self.total_metrics_point.item() / n_samples))


class XGEval(BaseEval):
    def __init__(self, model, config, data_loader, output_dir):
        super(XGEval, self).__init__(model, config)
        self.data_loader = data_loader

        self.criterion_mask = balanced_bce_loss
        self.criterion_edge = balanced_bce_loss
        self.criterion_region = balanced_bce_loss
        self.criterion_direct = balanced_ce_loss

        self.metric_ftns_mask = [IoU, relaxed_IoU]

        self.total_loss_mask = torch.tensor(0.).to(self.device)
        self.total_metrics_mask = torch.zeros(len(self.metric_ftns_mask)).to(self.device)

        self.output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

    def test(self):
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(tqdm(self.data_loader)):
                image_name, image, mask, edge, mini, direct = data.values()
                image = image.to(self.device)
                mask = mask.to(self.device)
                edge = edge.to(self.device)
                mini = mini.to(self.device)
                direct = direct.to(self.device)

                output_p, outputs_e, outputs_r, output_d = self.model(image)

                # save smple images
                prob_m = output_p.sigmoid() > 0.5  # todo
                prob_m_ = prob_m.squeeze().cpu().numpy() * 255
                prob_m_ = np.asarray(prob_m_, dtype=np.uint8)

                prob_d = torch.argmax(output_d, dim=1) > 0
                prob_d_ = prob_d.squeeze().cpu().numpy() * 255
                prob_d_ = np.asarray(prob_d_, dtype=np.uint8)

                # plot
                # fig, ax = plt.subplots(1, 3)
                # ax[0].imshow(mask.cpu().numpy()[0][0])
                # ax[1].imshow(prob_m_)
                # ax[2].imshow(prob_d_)
                # plt.show()

                # save
                # cv2.imwrite(str(self.output_dir / "mask" / (image_name[0] + '.png')), prob_m_)

                self.total_loss_mask += self.criterion_mask(output_p, mask)
                for i, metric in enumerate(self.metric_ftns_mask):
                    self.total_metrics_mask[i] += metric(output_d, mask)

        n_samples = len(self.data_loader.sampler)
        self.logger.info("loss_mask: {:.4f}".format(self.total_loss_mask.item() / n_samples))
        for i, metric in enumerate(self.metric_ftns_mask):
            self.logger.info("{}: {:.2f}%".format(metric.__name__, self.total_metrics_mask[i].item() / n_samples * 100))



# todo just for test
# class SegmentEval(BaseEval):
#     def __init__(self, model, config, data_loader, output_dir):
#         super(SegmentEval, self).__init__(model, config)
#         self.data_loader = data_loader
#
#         self.criterion = soft_iou_loss
#         self.metric_ftns = [rIoU]
#
#         self.total_metrics = torch.zeros(len(self.metric_ftns))
#
#         self.output_dir = output_dir
#         if not self.output_dir.exists():
#             self.output_dir.mkdir(parents=True, exist_ok=True)
#
#     def test(self):
#         self.model.eval()
#         e3_ = []
#         with torch.no_grad():
#             for i, data in enumerate(tqdm(self.data_loader)):
#                 image_name, image, mask = data.values()
#                 image = image.to(self.device)
#                 mask = mask.to(self.device)
#
#                 if 'RGB-PanSharpen_AOI_2_Vegas_img20' in image_name:
#                     print('lalala')
#
#                 output, e_ = self.model(image)
#                 e1, e2, e3, e4 = e_
#
#                 # plt.imshow(torch.mean(e3, dim=1).cpu()[0])
#                 # plt.axis('off')
#                 # plt.savefig(str(self.output_dir / (image_name[0] + '.png')))
#         n_samples = len(self.data_loader.sampler)
#         log = {}
#         log.update({
#             met.__name__: self.total_metrics[i].item() / n_samples for i, met in enumerate(self.metric_ftns)
#         })
#
#         self.logger.info(log)