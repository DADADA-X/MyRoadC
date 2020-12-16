from base import BaseTrainer
from utils import MetricTracker
from model.loss import *
from model.metrics import *


class SegmentTrainer(BaseTrainer):
    def __init__(self, model, optimizer, config, data_loader, valid_data_loader=None, lr_scheduler=None):
        super(SegmentTrainer, self).__init__(model, optimizer, config)
        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader)
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = 1
        self.criterion = soft_iou_loss
        self.metric_ftns_seg = [rIoU]
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns_seg],
                                           writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns_seg],
                                           writer=self.writer)

    def _train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()

        for batch_idx, data in enumerate(self.data_loader):
            step = (epoch - 1) * self.len_epoch + batch_idx
            image_name, image, mask = data.values()
            image = image.to(self.device)
            mask = mask.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(image)  # N, 2, w, h
            loss = self.criterion(output, mask)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step(step)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns_seg:
                self.train_metrics.update(met.__name__, met(output, mask))

            if batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx, self.data_loader),
                    loss.item()))

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_data_loader):
                step = (epoch - 1) * len(self.valid_data_loader) + batch_idx
                image_name, image, mask = data.values()
                image = image.to(self.device)
                mask = mask.to(self.device)

                output = self.model(image)
                loss = self.criterion(output, mask)

                self.writer.set_step(step, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns_seg:
                    self.valid_metrics.update(met.__name__, met(output, mask))

                self.logger.info('Valid Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx, self.valid_data_loader),
                    loss.item()))

        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx, data_loader):
        base = '[{:2d}/{:2d} ({:2.0f}%)]'
        current = batch_idx * data_loader.batch_size
        total = data_loader.n_samples
        return base.format(current, total, 100.0 * current / total)


class DSegmentTrainer(BaseTrainer):
    def __init__(self, model, optimizer, config, data_loader, valid_data_loader=None, lr_scheduler=None):
        super(DSegmentTrainer, self).__init__(model, optimizer, config)
        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader)
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = 1
        self.criterion = dice_bce_loss
        self.metric_ftns_seg = [IoU]
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns_seg],
                                           writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns_seg],
                                           writer=self.writer)

    def _train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()

        for batch_idx, data in enumerate(self.data_loader):
            step = (epoch - 1) * self.len_epoch + batch_idx
            image_name, image, mask = data.values()
            image = image.to(self.device)
            mask = mask.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(image)  # N, 2, w, h
            loss = self.criterion(output, mask)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step(step)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns_seg:
                self.train_metrics.update(met.__name__, met(output, mask))

            if batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx, self.data_loader),
                    loss.item()))

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_data_loader):
                step = (epoch - 1) * len(self.valid_data_loader) + batch_idx
                image_name, image, mask = data.values()
                image = image.to(self.device)
                mask = mask.to(self.device)

                output = self.model(image)
                loss = self.criterion(output, mask)

                self.writer.set_step(step, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns_seg:
                    self.valid_metrics.update(met.__name__, met(output, mask))

                self.logger.info('Valid Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx, self.valid_data_loader),
                    loss.item()))

        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx, data_loader):
        base = '[{:2d}/{:2d} ({:2.0f}%)]'
        current = batch_idx * data_loader.batch_size
        total = data_loader.n_samples
        return base.format(current, total, 100.0 * current / total)


class MTLTrainer(BaseTrainer):
    def __init__(self, model, optimizer, config, data_loader, valid_data_loader=None, lr_scheduler=None):
        super(MTLTrainer, self).__init__(model, optimizer, config)
        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader)
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler

        self.log_step = 1
        self.criterion_mask = soft_iou_loss
        self.criterion_conn = balanced_ce_loss

        self.metric_ftns_mask = [rIoU]
        self.metric_ftns_conn = [mIoU]

        self.train_metrics_mask = MetricTracker('loss_mask', *[m.__name__ + '_mask' for m in self.metric_ftns_mask],
                                                writer=self.writer)
        self.train_metrics_conn = MetricTracker('loss_conn', *[m.__name__ + '_conn' for m in self.metric_ftns_conn],
                                                writer=self.writer)

        self.valid_metrics_mask = MetricTracker('loss_mask', *[m.__name__ + '_mask' for m in self.metric_ftns_mask],
                                                writer=self.writer)
        self.valid_metrics_conn = MetricTracker('loss_conn', *[m.__name__ + '_conn' for m in self.metric_ftns_conn],
                                                writer=self.writer)

    def _train_epoch(self, epoch):
        self.model.train()
        self.train_metrics_mask.reset()
        self.valid_metrics_conn.reset()

        for batch_idx, data in enumerate(self.data_loader):
            step = (epoch - 1) * self.len_epoch + batch_idx
            image_name, image, mask, conn = data.values()
            image = image.to(self.device)
            mask = mask.to(self.device)
            conn = conn.to(self.device)

            self.optimizer.zero_grad()
            output1, output2 = self.model(image)

            loss_mask = self.criterion_mask(output1, mask)
            loss_conn = self.criterion_conn(output2, conn)
            loss_total = loss_mask + loss_conn
            loss_total.backward()
            self.optimizer.step()

            self.writer.set_step(step)

            self.train_metrics_mask.update('loss_mask', loss_mask.item())
            self.train_metrics_conn.update('loss_conn', loss_conn.item())

            for met in self.metric_ftns_mask:
                self.train_metrics_mask.update(met.__name__ + '_mask', met(output1, mask))
            for met in self.metric_ftns_conn:
                self.train_metrics_conn.update(met.__name__ + '_conn', met(output2, conn))

            if batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} {} Total_Loss: {:.6f} Loss_Mask: {:.6f} '
                                 'Loss_Conn: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx, self.data_loader),
                    loss_total.item(),
                    loss_mask.item(),
                    loss_conn.item()))

        log_mask = self.train_metrics_mask.result()  # average
        log_conn = self.train_metrics_conn.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log_mask.update(**{'val_' + k: v for k, v in val_log.items() if 'mask' in k})
            log_conn.update(**{'val_' + k: v for k, v in val_log.items() if 'conn' in k})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return {**log_mask, **log_conn}

    def _valid_epoch(self, epoch):
        self.model.eval()
        self.valid_metrics_mask.reset()
        self.valid_metrics_conn.reset()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_data_loader):
                step = (epoch - 1) * len(self.valid_data_loader) + batch_idx
                image_name, image, mask, conn = data.values()
                image = image.to(self.device)
                mask = mask.to(self.device)
                conn = conn.to(self.device)

                output1, output2 = self.model(image)

                loss_mask = self.criterion_mask(output1, mask)
                loss_conn = self.criterion_conn(output2, conn)
                loss_total = loss_mask + loss_conn

                self.writer.set_step(step, 'valid')
                self.valid_metrics_mask.update('loss_mask', loss_mask.item())
                self.valid_metrics_conn.update('loss_conn', loss_conn.item())

                for met in self.metric_ftns_mask:
                    self.valid_metrics_mask.update(met.__name__ + '_mask', met(output1, mask))
                for met in self.metric_ftns_conn:
                    self.valid_metrics_conn.update(met.__name__ + '_conn', met(output2, conn))

                self.logger.info('Valid Epoch: {} {} Total_Loss: {:.6f} Loss_Mask: {:.6f} '
                                 'Loss_Conn: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx, self.valid_data_loader),
                    loss_total.item(),
                    loss_mask.item(),
                    loss_conn.item()))

        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        valid_metrics = {**self.valid_metrics_mask.result(), **self.valid_metrics_conn.result()}

        return valid_metrics

    def _progress(self, batch_idx, data_loader):
        base = '[{:2d}/{:2d} ({:2.0f}%)]'
        current = batch_idx * data_loader.batch_size
        total = data_loader.n_samples
        return base.format(current, total, 100.0 * current / total)


class HGSegmentTrainer(BaseTrainer):
    def __init__(self, model, optimizer, config, data_loader, valid_data_loader=None, lr_scheduler=None):
        super(HGSegmentTrainer, self).__init__(model, optimizer, config)
        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader)
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = 1
        self.criterion = soft_iou_loss
        self.metric_ftns_seg = [rIoU]
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns_seg],
                                           writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns_seg],
                                           writer=self.writer)

    def _train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()

        for batch_idx, data in enumerate(self.data_loader):
            step = (epoch - 1) * self.len_epoch + batch_idx
            image_name, image, mask, mask_4 = data.values()
            image = image.to(self.device)
            mask = mask.to(self.device)
            mask_4 = mask_4.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(image)

            loss = self.criterion(outputs[-1], mask)
            for output in outputs[:-1]:
                loss += self.criterion(output, mask_4)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step(step)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns_seg:
                self.train_metrics.update(met.__name__, met(outputs[-1], mask))

            if batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx, self.data_loader),
                    loss.item()))

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_data_loader):
                step = (epoch - 1) * len(self.valid_data_loader) + batch_idx
                image_name, image, mask, mask_4 = data.values()
                image = image.to(self.device)
                mask = mask.to(self.device)
                mask_4 = mask_4.to(self.device)

                outputs = self.model(image)
                loss = self.criterion(outputs[-1], mask)
                for output in outputs[:-1]:
                    loss += self.criterion(output, mask_4)

                self.writer.set_step(step, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns_seg:
                    self.valid_metrics.update(met.__name__, met(outputs[-1], mask))

                self.logger.info('Valid Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx, self.valid_data_loader),
                    loss.item()))

        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx, data_loader):
        base = '[{:2d}/{:2d} ({:2.0f}%)]'
        current = batch_idx * data_loader.batch_size
        total = data_loader.n_samples
        return base.format(current, total, 100.0 * current / total)


class HGMTLTrainer(BaseTrainer):
    def __init__(self, model, optimizer, config, data_loader, valid_data_loader=None, lr_scheduler=None):
        super(HGMTLTrainer, self).__init__(model, optimizer, config)
        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader)
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler

        self.log_step = 1
        self.criterion_mask = soft_iou_loss
        self.criterion_conn = balanced_ce_loss

        self.metric_ftns_mask = [rIoU]
        self.metric_ftns_conn = [mIoU]

        self.train_metrics_mask = MetricTracker('loss_mask', *[m.__name__ + '_mask' for m in self.metric_ftns_mask],
                                                writer=self.writer)
        self.train_metrics_conn = MetricTracker('loss_conn', *[m.__name__ + '_conn' for m in self.metric_ftns_conn],
                                                writer=self.writer)

        self.valid_metrics_mask = MetricTracker('loss_mask', *[m.__name__ + '_mask' for m in self.metric_ftns_mask],
                                                writer=self.writer)
        self.valid_metrics_conn = MetricTracker('loss_conn', *[m.__name__ + '_conn' for m in self.metric_ftns_conn],
                                                writer=self.writer)

    def _train_epoch(self, epoch):
        self.model.train()
        self.train_metrics_mask.reset()
        self.valid_metrics_conn.reset()

        for batch_idx, data in enumerate(self.data_loader):
            step = (epoch - 1) * self.len_epoch + batch_idx
            image_name, image, mask, mask_4, conn, conn_4 = data.values()
            image = image.to(self.device)
            mask = mask.to(self.device)
            mask_4 = mask_4.to(self.device)
            conn = conn.to(self.device)
            conn_4 = conn_4.to(self.device)

            self.optimizer.zero_grad()
            output1, output2 = self.model(image)

            loss_mask = self.criterion_mask(output1[-1], mask)
            for output in output1[:-1]:
                loss_mask += self.criterion_mask(output, mask_4)

            loss_conn = self.criterion_conn(output2[-1], conn)
            for output in output2[:-1]:
                loss_conn += self.criterion_conn(output, conn_4)

            loss_total = loss_mask + loss_conn
            loss_total.backward()
            self.optimizer.step()

            self.writer.set_step(step)

            self.train_metrics_mask.update('loss_mask', loss_mask.item())
            self.train_metrics_conn.update('loss_conn', loss_conn.item())

            for met in self.metric_ftns_mask:
                self.train_metrics_mask.update(met.__name__ + '_mask', met(output1[-1], mask))
            for met in self.metric_ftns_conn:
                self.train_metrics_conn.update(met.__name__ + '_conn', met(output2[-1], conn))

            if batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} {} Total_Loss: {:.6f} Loss_Mask: {:.6f} '
                                 'Loss_Conn: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx, self.data_loader),
                    loss_total.item(),
                    loss_mask.item(),
                    loss_conn.item()))

        log_mask = self.train_metrics_mask.result()  # average
        log_conn = self.train_metrics_conn.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log_mask.update(**{'val_' + k: v for k, v in val_log.items() if 'mask' in k})
            log_conn.update(**{'val_' + k: v for k, v in val_log.items() if 'conn' in k})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return {**log_mask, **log_conn}

    def _valid_epoch(self, epoch):
        self.model.eval()
        self.valid_metrics_mask.reset()
        self.valid_metrics_conn.reset()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_data_loader):
                step = (epoch - 1) * len(self.valid_data_loader) + batch_idx
                image_name, image, mask, mask_4, conn, conn_4 = data.values()
                image = image.to(self.device)
                mask = mask.to(self.device)
                mask_4 = mask_4.to(self.device)
                conn = conn.to(self.device)
                conn_4 = conn_4.to(self.device)

                output1, output2 = self.model(image)

                loss_mask = self.criterion_mask(output1[-1], mask)
                for output in output1[:-1]:
                    loss_mask += self.criterion_mask(output, mask_4)

                loss_conn = self.criterion_conn(output2[-1], conn)
                for output in output2[:-1]:
                    loss_conn += self.criterion_conn(output, conn_4)

                loss_total = loss_mask + loss_conn

                self.writer.set_step(step, 'valid')
                self.valid_metrics_mask.update('loss_mask', loss_mask.item())
                self.valid_metrics_conn.update('loss_conn', loss_conn.item())

                for met in self.metric_ftns_mask:
                    self.valid_metrics_mask.update(met.__name__ + '_mask', met(output1[-1], mask))
                for met in self.metric_ftns_conn:
                    self.valid_metrics_conn.update(met.__name__ + '_conn', met(output2[-1], conn))

                self.logger.info('Valid Epoch: {} {} Total_Loss: {:.6f} Loss_Mask: {:.6f} '
                                 'Loss_Conn: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx, self.valid_data_loader),
                    loss_total.item(),
                    loss_mask.item(),
                    loss_conn.item()))

        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        valid_metrics = {**self.valid_metrics_mask.result(), **self.valid_metrics_conn.result()}

        return valid_metrics

    def _progress(self, batch_idx, data_loader):
        base = '[{:2d}/{:2d} ({:2.0f}%)]'
        current = batch_idx * data_loader.batch_size
        total = data_loader.n_samples
        return base.format(current, total, 100.0 * current / total)


class MTLTrainer3(BaseTrainer):
    def __init__(self, model, optimizer, config, data_loader, valid_data_loader=None, lr_scheduler=None):
        super(MTLTrainer3, self).__init__(model, optimizer, config)
        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader)
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler

        self.log_step = 1
        self.criterion_mask = soft_iou_loss
        self.criterion_line = mse_loss
        self.criterion_point = mse_loss

        self.metric_ftns_mask = [rIoU]
        self.metric_ftns_line = [MSE]
        self.metric_ftns_point = [MSE]

        self.train_metrics_mask = MetricTracker('loss_mask', *[m.__name__ + '_mask' for m in self.metric_ftns_mask],
                                                writer=self.writer)
        self.train_metrics_line = MetricTracker('loss_line', *[m.__name__ + '_line' for m in self.metric_ftns_line],
                                                writer=self.writer)
        self.train_metrics_point = MetricTracker('loss_point', *[m.__name__ + '_point' for m in self.metric_ftns_point],
                                                 writer=self.writer)

        self.valid_metrics_mask = MetricTracker('loss_mask', *[m.__name__ + '_mask' for m in self.metric_ftns_mask],
                                                writer=self.writer)
        self.valid_metrics_line = MetricTracker('loss_line', *[m.__name__ + '_line' for m in self.metric_ftns_line],
                                                writer=self.writer)
        self.valid_metrics_point = MetricTracker('loss_point', *[m.__name__ + '_point' for m in self.metric_ftns_point],
                                                 writer=self.writer)

    def _train_epoch(self, epoch):
        self.model.train()
        self.train_metrics_mask.reset()
        self.train_metrics_line.reset()
        self.train_metrics_point.reset()

        for batch_idx, data in enumerate(self.data_loader):
            step = (epoch - 1) * self.len_epoch + batch_idx
            image_name, image, mask, line, point = data.values()
            image = image.to(self.device)
            mask = mask.to(self.device)
            line = line.to(self.device)
            point = point.to(self.device)

            self.optimizer.zero_grad()
            output1, output2, output3 = self.model(image)

            loss_mask = self.criterion_mask(output1, mask)
            loss_line = self.criterion_line(output2, line)
            loss_point = self.criterion_point(output3, point)
            loss_total = loss_mask + loss_line + loss_point

            loss_total.backward()
            self.optimizer.step()

            self.writer.set_step(step)

            self.train_metrics_mask.update('loss_mask', loss_mask.item())
            self.train_metrics_line.update('loss_line', loss_line.item())
            self.train_metrics_point.update('loss_point', loss_point.item())

            for met in self.metric_ftns_mask:
                self.train_metrics_mask.update(met.__name__ + '_mask', met(output1, mask))
            for met in self.metric_ftns_line:
                self.train_metrics_line.update(met.__name__ + '_line', met(output2, line))
            for met in self.metric_ftns_point:
                self.train_metrics_point.update(met.__name__ + '_point', met(output3, point))

            if batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} {} Total_Loss: {:.6f} Loss_Mask: {:.6f} '
                                 'Loss_Line: {:.6f} Loss_Point: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx, self.data_loader),
                    loss_total.item(),
                    loss_mask.item(),
                    loss_line.item(),
                    loss_point.item()))

        log_mask = self.train_metrics_mask.result()  # average
        log_line = self.train_metrics_line.result()
        log_point = self.train_metrics_point.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log_mask.update(**{'val_' + k: v for k, v in val_log['mask'].items()})
            log_line.update(**{'val_' + k: v for k, v in val_log['line'].items()})
            log_point.update(**{'val_' + k: v for k, v in val_log['point'].items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return {**log_mask, **log_line, **log_point}

    def _valid_epoch(self, epoch):
        self.model.eval()
        self.valid_metrics_mask.reset()
        self.valid_metrics_line.reset()
        self.valid_metrics_point.reset()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_data_loader):
                step = (epoch - 1) * len(self.valid_data_loader) + batch_idx
                image_name, image, mask, line, point = data.values()
                image = image.to(self.device)
                mask = mask.to(self.device)
                line = line.to(self.device)
                point = point.to(self.device)

                output1, output2, output3 = self.model(image)

                loss_mask = self.criterion_mask(output1, mask)
                loss_line = self.criterion_line(output2, line)
                loss_point = self.criterion_point(output3, point)
                loss_total = loss_mask + loss_line + loss_point

                self.writer.set_step(step, 'valid')
                self.valid_metrics_mask.update('loss_mask', loss_mask.item())
                self.valid_metrics_line.update('loss_line', loss_line.item())
                self.valid_metrics_point.update('loss_point', loss_point.item())

                for met in self.metric_ftns_mask:
                    self.valid_metrics_mask.update(met.__name__ + '_mask', met(output1, mask))
                for met in self.metric_ftns_line:
                    self.valid_metrics_line.update(met.__name__ + '_line', met(output2, line))
                for met in self.metric_ftns_point:
                    self.valid_metrics_point.update(met.__name__ + '_point', met(output3, point))

                self.logger.info('Valid Epoch: {} {} Total_Loss: {:.6f} Loss_Mask: {:.6f} '
                                 'Loss_Line: {:.6f} Loss_Point: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx, self.valid_data_loader),
                    loss_total.item(),
                    loss_mask.item(),
                    loss_line.item(),
                    loss_point.item()))

        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        valid_metrics = {"mask": self.valid_metrics_mask.result(),
                         "line": self.valid_metrics_line.result(),
                         "point": self.valid_metrics_point.result()}

        return valid_metrics

    def _progress(self, batch_idx, data_loader):
        base = '[{:2d}/{:2d} ({:2.0f}%)]'
        current = batch_idx * data_loader.batch_size
        total = data_loader.n_samples
        return base.format(current, total, 100.0 * current / total)


class ImproveTrainer(BaseTrainer):
    def __init__(self, model, optimizer, config, data_loader, valid_data_loader=None, lr_scheduler=None):
        super(ImproveTrainer, self).__init__(model, optimizer, config)
        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader)
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler

        self.log_step = 1
        self.criterion_mask = soft_iou_loss
        self.criterion_conn = balanced_ce_loss

        self.metric_ftns_mask = [rIoU]
        self.metric_ftns_conn = [mIoU]

        self.train_metrics_mask = MetricTracker('loss_mask', *[m.__name__ + '_mask' for m in self.metric_ftns_mask],
                                                writer=self.writer)
        self.train_metrics_conn = MetricTracker('loss_conn', *[m.__name__ + '_conn' for m in self.metric_ftns_conn],
                                                writer=self.writer)

        self.valid_metrics_mask = MetricTracker('loss_mask', *[m.__name__ + '_mask' for m in self.metric_ftns_mask],
                                                writer=self.writer)
        self.valid_metrics_conn = MetricTracker('loss_conn', *[m.__name__ + '_conn' for m in self.metric_ftns_conn],
                                                writer=self.writer)

    def _train_epoch(self, epoch):
        self.model.train()
        self.train_metrics_mask.reset()
        self.valid_metrics_conn.reset()

        for batch_idx, data in enumerate(self.data_loader):
            step = (epoch - 1) * self.len_epoch + batch_idx
            image_name, image, mask, mask_2, mask_4, conn, conn_2, conn_4 = data.values()
            image = image.to(self.device)
            mask = mask.to(self.device)
            mask_2 = mask_2.to(self.device)
            mask_4 = mask_4.to(self.device)
            conn = conn.to(self.device)
            conn_2 = conn_2.to(self.device)
            conn_4 = conn_4.to(self.device)

            self.optimizer.zero_grad()
            output1, output2 = self.model(image)

            loss_mask = self.criterion_mask(output1[-1], mask)
            loss_mask += self.criterion_mask(output1[-2], mask_2)
            for output in output1[:-2]:
                loss_mask += self.criterion_mask(output, mask_4)

            loss_conn = self.criterion_conn(output2[-1], conn)
            loss_conn += self.criterion_conn(output2[-2], conn_2)
            for output in output2[:-2]:
                loss_conn += self.criterion_conn(output, conn_4)

            loss_total = loss_mask + loss_conn
            loss_total.backward()
            self.optimizer.step()

            self.writer.set_step(step)

            self.train_metrics_mask.update('loss_mask', loss_mask.item())
            self.train_metrics_conn.update('loss_conn', loss_conn.item())

            for met in self.metric_ftns_mask:
                self.train_metrics_mask.update(met.__name__ + '_mask', met(output1[-1], mask))
            for met in self.metric_ftns_conn:
                self.train_metrics_conn.update(met.__name__ + '_conn', met(output2[-1], conn))

            if batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} {} Total_Loss: {:.6f} Loss_Mask: {:.6f} '
                                 'Loss_Conn: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx, self.data_loader),
                    loss_total.item(),
                    loss_mask.item(),
                    loss_conn.item()))

        log_mask = self.train_metrics_mask.result()  # average
        log_conn = self.train_metrics_conn.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log_mask.update(**{'val_' + k: v for k, v in val_log.items() if 'mask' in k})
            log_conn.update(**{'val_' + k: v for k, v in val_log.items() if 'conn' in k})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return {**log_mask, **log_conn}

    def _valid_epoch(self, epoch):
        self.model.eval()
        self.valid_metrics_mask.reset()
        self.valid_metrics_conn.reset()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_data_loader):
                step = (epoch - 1) * len(self.valid_data_loader) + batch_idx
                image_name, image, mask, mask_2, mask_4, conn, conn_2, conn_4 = data.values()
                image = image.to(self.device)
                mask = mask.to(self.device)
                mask_2 = mask_2.to(self.device)
                mask_4 = mask_4.to(self.device)
                conn = conn.to(self.device)
                conn_2 = conn_2.to(self.device)
                conn_4 = conn_4.to(self.device)

                output1, output2 = self.model(image)

                loss_mask = self.criterion_mask(output1[-1], mask)
                loss_mask += self.criterion_mask(output1[-2], mask_2)
                for output in output1[:-2]:
                    loss_mask += self.criterion_mask(output, mask_4)

                loss_conn = self.criterion_conn(output2[-1], conn)
                loss_conn += self.criterion_conn(output2[-2], conn_2)
                for output in output2[:-2]:
                    loss_conn += self.criterion_conn(output, conn_4)

                loss_total = loss_mask + loss_conn

                self.writer.set_step(step, 'valid')
                self.valid_metrics_mask.update('loss_mask', loss_mask.item())
                self.valid_metrics_conn.update('loss_conn', loss_conn.item())

                for met in self.metric_ftns_mask:
                    self.valid_metrics_mask.update(met.__name__ + '_mask', met(output1[-1], mask))
                for met in self.metric_ftns_conn:
                    self.valid_metrics_conn.update(met.__name__ + '_conn', met(output2[-1], conn))

                self.logger.info('Valid Epoch: {} {} Total_Loss: {:.6f} Loss_Mask: {:.6f} '
                                 'Loss_Conn: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx, self.valid_data_loader),
                    loss_total.item(),
                    loss_mask.item(),
                    loss_conn.item()))

        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        valid_metrics = {**self.valid_metrics_mask.result(), **self.valid_metrics_conn.result()}

        return valid_metrics

    def _progress(self, batch_idx, data_loader):
        base = '[{:2d}/{:2d} ({:2.0f}%)]'
        current = batch_idx * data_loader.batch_size
        total = data_loader.n_samples
        return base.format(current, total, 100.0 * current / total)