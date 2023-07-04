import csv
import sys


class CSVLogger:
    def __init__(self, fieldnames, filename='log.csv'):
        self.filename = filename
        self.csv_file = open(filename, 'w')

        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

        self.csv_file.flush()

    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def generate_loggers():
    # Image Batches
    loss_batch = AverageMeter()
    loss_original_batch = AverageMeter()
    ssim_batch = AverageMeter()
    ssim_original_batch = AverageMeter()
    psnr_batch = AverageMeter()
    psnr_original_batch = AverageMeter()

    batch_loggers = (loss_batch, loss_original_batch, ssim_batch, ssim_original_batch, psnr_batch, psnr_original_batch)

    # Total Training
    loss_meter_train = AverageMeter()
    loss_original_meter_train = AverageMeter()
    ssim_meter_train = AverageMeter()
    ssim_original_meter_train = AverageMeter()
    psnr_meter_train = AverageMeter()
    psnr_original_meter_train = AverageMeter()

    train_loggers = (loss_meter_train, loss_original_meter_train, ssim_meter_train, ssim_original_meter_train,
                     psnr_meter_train, psnr_original_meter_train)

    # Validation
    loss_meter_val = AverageMeter()
    loss_original_meter_val = AverageMeter()
    ssim_meter_val = AverageMeter()
    ssim_original_meter_val = AverageMeter()
    psnr_meter_val = AverageMeter()
    psnr_original_meter_val = AverageMeter()

    val_loggers = (loss_meter_val, loss_original_meter_val, ssim_meter_val, ssim_original_meter_val, psnr_meter_val,
                   psnr_original_meter_val)

    return batch_loggers, train_loggers, val_loggers
