import datetime
import logging
import time
from collections import defaultdict, deque
from logging import FileHandler, Formatter

import torch
import torch.distributed as dist

from ..common.constant import def_logger, LOGGING_FORMAT
from ..common.file_util import make_parent_dirs
from ..common.main_util import is_dist_avail_and_initialized, is_main_process

logger = def_logger.getChild(__name__)


def set_basic_log_config():
    """
    Sets a default basic configuration for logging.
    """
    logging.basicConfig(
        format=LOGGING_FORMAT,
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO
    )


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


class TrainingTracker(object):
    """
    A thin wrapper around an experiment tracking library (`trackio` or `wandb`).
    The library is imported lazily so that it stays an optional dependency.

    :param engine: tracking library name ('trackio' or 'wandb').
    :type engine: str
    :param kwargs: keyword arguments passed to the library's ``init`` function
        (e.g., ``project``, ``name``, ``config``). See the external references below for available arguments.
    :type kwargs: dict

    .. code-block:: python
       :caption: An example to instantiate :class:`TrainingTracker` and log metrics with it.

        tracker = TrainingTracker(
            'trackio',
            project='torchdistill-cifar10',
            name='resnet18-kd-run1',
            config={'train': {'num_epochs': 182}, 'student_model': 'resnet18'}
        )
        tracker.log({'train/loss': 0.512, 'train/lr': 0.1}, step=100)
        tracker.log({'val/acc1': 92.3, 'epoch': 0}, step=391)
        tracker.finish()

    .. seealso::
        * `Trackio documentation <https://huggingface.co/docs/trackio/index>`_ (``trackio.init``)
          for ``engine='trackio'``
        * `wandb.init reference <https://docs.wandb.ai/ref/python/init/>`_ for ``engine='wandb'``
    """
    SUPPORTED_ENGINES = ('trackio', 'wandb')

    def __init__(self, engine, **kwargs):
        if engine not in self.SUPPORTED_ENGINES:
            raise ValueError(f'`engine` should be one of {self.SUPPORTED_ENGINES}, but got `{engine}`')

        if engine == 'trackio':
            import trackio
            self.module = trackio
        else:
            import wandb
            self.module = wandb

        self.engine = engine
        self.module.init(**kwargs)

    def log(self, metrics, step=None):
        """
        Logs a metric dict.

        :param metrics: metric names and values.
        :type metrics: dict
        :param step: global step to associate the metrics with.
        :type step: int or None
        """
        self.module.log(metrics, step=step)

    def finish(self):
        """
        Finishes the tracking run.
        """
        self.module.finish()


def setup_tracker(tracker_config, run_config=None):
    """
    Sets up a :class:`TrainingTracker` from ``tracker_config``.

    :param tracker_config: tracker configuration with 'engine' ('trackio' or 'wandb') and
        optional 'kwargs' passed to the library's ``init`` function.
        If None or its 'engine' is None, no tracker is set up.
    :type tracker_config: dict or None
    :param run_config: run configuration (e.g., loaded yaml config) to be logged as the run's config.
    :type run_config: dict or None
    :return: training tracker if configured and this is the main process, None otherwise.
    :rtype: TrainingTracker or None

    .. code-block:: yaml
       :caption: An example (partial) YAML config whose ``tracker`` entry is passed to :func:`setup_tracker`
          as ``tracker_config``. ``kwargs`` is passed as-is to ``trackio.init`` / ``wandb.init``.

        tracker:
          engine: 'trackio'
          kwargs:
            project: 'torchdistill-cifar10'
            name: 'resnet18-kd-run1'

    .. code-block:: python
       :caption: An example to set up a :class:`TrainingTracker` with the YAML config above.

        config = yaml_util.load_yaml_file('/path/to/the/yaml/config/above.yaml')
        tracker = setup_tracker(config.get('tracker', None), run_config=config)

        # Equivalent dict-based setup without a YAML file
        tracker = setup_tracker(
            {'engine': 'trackio', 'kwargs': {'project': 'torchdistill-cifar10', 'name': 'resnet18-kd-run1'}},
            run_config=config
        )

    .. seealso::
        * `Trackio documentation <https://huggingface.co/docs/trackio/index>`_ (``trackio.init``)
          for ``engine: 'trackio'``
        * `wandb.init reference <https://docs.wandb.ai/ref/python/init/>`_ for ``engine: 'wandb'``
    """
    if tracker_config is None or tracker_config.get('engine', None) is None or not is_main_process():
        return None

    kwargs = dict(tracker_config.get('kwargs', None) or dict())
    if run_config is not None:
        kwargs.setdefault('config', run_config)
    return TrainingTracker(tracker_config['engine'], **kwargs)


class TrainingTrackerReader(object):
    """
    A read-side companion to :class:`TrainingTracker` that loads metrics logged with `trackio` or `wandb`
    back as ``pandas.DataFrame``. The libraries are imported lazily so that they stay optional dependencies.

    :param engine: tracking library name ('trackio' or 'wandb').
    :type engine: str
    :param wandb_entity: wandb entity (user or team name). Used only if ``engine`` = 'wandb'.
        If None, the default entity of the wandb API key is used.
    :type wandb_entity: str or None

    .. code-block:: python
       :caption: An example to load the run history logged with the :class:`TrainingTracker` example.

        reader = TrainingTrackerReader('trackio')
        history = reader.load_run_history('torchdistill-cifar10', 'resnet18-kd-run1')
        # `history` is a pandas.DataFrame with 'step', 'relative_time', and metric columns
        # such as 'train/loss' and 'val/acc1'
        print(history[history['val/acc1'].notna()][['epoch', 'val/acc1']])

    .. seealso::
        * `Trackio documentation <https://huggingface.co/docs/trackio/index>`_ for ``engine='trackio'``
        * `wandb public API reference <https://docs.wandb.ai/ref/python/public-api/>`_ for ``engine='wandb'``
    """
    SUPPORTED_ENGINES = ('trackio', 'wandb')

    def __init__(self, engine, wandb_entity=None):
        if engine not in self.SUPPORTED_ENGINES:
            raise ValueError(f'`engine` should be one of {self.SUPPORTED_ENGINES}, but got `{engine}`')

        if engine == 'trackio':
            from trackio.sqlite_storage import SQLiteStorage
            self.storage = SQLiteStorage
            self.api = None
        else:
            import wandb
            self.storage = None
            self.api = wandb.Api()

        self.engine = engine
        self.wandb_entity = wandb_entity

    def _find_wandb_run(self, project, run_name):
        project_path = project if self.wandb_entity is None else f'{self.wandb_entity}/{project}'
        runs = self.api.runs(project_path, filters={'display_name': run_name})
        if len(runs) == 0:
            raise ValueError(f'run `{run_name}` was not found in wandb project `{project_path}`')
        return runs[0]

    @staticmethod
    def _normalize_history(data_frame):
        import pandas as pd
        column_dict = {'_step': 'step', '_runtime': 'relative_time'}
        data_frame = data_frame.rename(columns={k: v for k, v in column_dict.items() if k in data_frame.columns})
        if 'relative_time' not in data_frame.columns and 'timestamp' in data_frame.columns:
            timestamps = pd.to_datetime(data_frame['timestamp'])
            data_frame['relative_time'] = (timestamps - timestamps.min()).dt.total_seconds()
        if 'step' in data_frame.columns:
            data_frame = data_frame.sort_values('step').reset_index(drop=True)
        return data_frame

    def load_run_history(self, project, run_name):
        """
        Loads the metric history of a run as a ``pandas.DataFrame``, one row per logged step.
        Engine-specific columns are normalized so that 'step' and 'relative_time' (seconds since
        the first log) are available for both engines, in addition to the logged metric columns.

        :param project: project name used at logging time.
        :type project: str
        :param run_name: run name used at logging time.
        :type run_name: str
        :return: metric history of the run.
        :rtype: pandas.DataFrame
        """
        import pandas as pd
        if self.engine == 'trackio':
            records = self.storage.get_logs(project, run_name)
            if len(records) == 0:
                raise ValueError(f'run `{run_name}` was not found in trackio project `{project}`')
        else:
            run = self._find_wandb_run(project, run_name)
            records = list(run.scan_history())
        return self._normalize_history(pd.DataFrame(records))


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
    :param tracker: training tracker to log metrics with. If None, no tracking is done.
    :type tracker: TrainingTracker or None
    :param tracker_prefix: prefix prepended to metric names when logging with ``tracker`` (e.g., 'train/').
    :type tracker_prefix: str
    :param tracker_start_step: global step at which this logger's iterations start.
    :type tracker_start_step: int
    """
    def __init__(self, delimiter="\t", tracker=None, tracker_prefix='', tracker_start_step=0):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.tracker = tracker
        self.tracker_prefix = tracker_prefix
        self.tracker_start_step = tracker_start_step

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

                if self.tracker is not None:
                    self.tracker.log(
                        {self.tracker_prefix + name: meter.value for name, meter in self.meters.items()},
                        step=self.tracker_start_step + i
                    )

            i += 1
            end = time.time()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info('{} Total time: {}'.format(header, total_time_str))
