import datetime
import time
from collections import defaultdict, deque
from logging import FileHandler, Formatter

import torch
import torch.distributed as dist

from ..common.constant import def_logger, LOGGING_FORMAT
from ..common.file_util import make_parent_dirs
from ..common.main_util import is_dist_avail_and_initialized

logger = def_logger.getChild(__name__)


def setup_log_file(log_file_path):
    """
    Sets a file handler with ``log_file_path`` to write a log file.

    :param log_file_path: log file path.
    :type log_file_path: str
    """
    make_parent_dirs(log_file_path)
    fh = FileHandler(filename=log_file_path, mode='w')
    fh.setFormatter(Formatter(LOGGING_FORMAT))
    def_logger.addHandler(fh)


class SmoothedValue(object):
    """
    A deque-based value object tracks a series of values and provides access to smoothed values
    over a window or the global series average. The original implementation is https://github.com/pytorch/vision/blob/main/references/classification/utils.py

    :param window_size: window size.
    :type window_size: int
    :param fmt: text format.
    :type fmt: str or None
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        """
        Appends ``value``.

        :param value: value to be added.
        :type value: float or int
        :param n: sample count.
        :type n: int
        """
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Synchronizes between processes.

        .. warning::
            It does not synchronize the deque.
        """
        if not is_dist_avail_and_initialized():
            return

        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value
        )


class MetricLogger(object):
    """
    A metric logger with :class:`SmoothedValue`.
    The original implementation is https://github.com/pytorch/vision/blob/main/references/classification/utils.py

    :param delimiter: delimiter in a log message.
    :type delimiter: str
    """
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        """
        Updates a metric dict whose values are :class:`SmoothedValue`.

        :param kwargs: keys and values.
        :type kwargs: dict
        """
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            assert isinstance(v, (float, int)), f'`{k}` ({v}) should be either float or int'
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        """
        Synchronizes between processes.
        """
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        """
        Add a new metric name and value.

        :param name: metric name.
        :type name: str
        :param meter: smoothed value.
        :type meter: SmoothedValue
        """
        self.meters[name] = meter

    def log_every(self, iterable, log_freq, header=None):
        """
        Add a new metric name and value.

        :param iterable: iterable object (e.g., data loader).
        :type iterable: typing.Iterable
        :param log_freq: log frequency.
        :type log_freq: int
        :param header: log message header.
        :type header: str
        :return: item in ``iterative``.
        :rtype: Any
        """
        i = 0
        if not header:
            header = ''

        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])

        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % log_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    logger.info(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    logger.info(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))

            i += 1
            end = time.time()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info('{} Total time: {}'.format(header, total_time_str))
