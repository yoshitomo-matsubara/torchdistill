import argparse
import os

import matplotlib.pyplot as plt
import seaborn as sns

from torchdistill.misc.log import TrainingTrackerReader


def get_args():
    parser = argparse.ArgumentParser(description='Log visualizer')
    parser.add_argument(
        '--engine', default='trackio', choices=['trackio', 'wandb', 'text'],
        help='where to read metrics from: an experiment tracker or legacy text log files'
    )
    parser.add_argument('--project', help='tracker project name (for trackio/wandb engines)')
    parser.add_argument('--runs', metavar='N', nargs='+', help='list of run names to visualize')
    parser.add_argument(
        '--x', default='step', help='column used as x-axis e.g., `step`, `epoch`, `relative_time` (sec since first log)'
    )
    parser.add_argument(
        '--y', metavar='N', nargs='+', default=['val/acc1'],
        help='list of metric names to plot e.g., `train/loss`, `val/acc1`'
    )
    parser.add_argument('--labels', metavar='N', nargs='+', help='list of labels used in plots')
    parser.add_argument('--entity', help='wandb entity (user or team name), for wandb engine only')
    parser.add_argument('--out', help='output image file path; if not given, plots are shown in a window')
    parser.add_argument('--logs', metavar='N', nargs='+', help='list of log file paths, for text engine only')
    parser.add_argument(
        '--task', default='classification',
        help='type of tasks defined in the log files, for text engine only'
    )
    return parser.parse_args()


def read_files(file_paths, labels):
    if labels is None or len(labels) != len(file_paths):
        labels = [os.path.basename(file_path) for file_path in file_paths]

    log_dict = dict()
    for file_path, label in zip(file_paths, labels):
        with open(os.path.expanduser(file_path), 'r') as fp:
            log_dict[file_path] = ([line.strip() for line in fp], label)
    return log_dict


def load_run_histories(engine, project, run_names, labels, wandb_entity=None):
    if labels is None or len(labels) != len(run_names):
        labels = run_names

    reader = TrainingTrackerReader(engine, wandb_entity=wandb_entity)
    return {label: reader.load_run_history(project, run_name) for run_name, label in zip(run_names, labels)}


def visualize_metrics(history_dict, x_name, y_names, output_file_path=None):
    sns.set()
    fig, axes = plt.subplots(1, len(y_names), figsize=(6 * len(y_names), 4.5), squeeze=False)
    for ax, y_name in zip(axes[0], y_names):
        for label, history in history_dict.items():
            if y_name not in history.columns:
                print(f'metric `{y_name}` was not found in run `{label}`, skipping')
                continue

            sub_history = history[history[y_name].notna() & history[x_name].notna()]
            ax.plot(sub_history[x_name], sub_history[y_name], '-o', label=r'${}$'.format(label))

        ax.legend()
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)

    fig.tight_layout()
    if output_file_path is None:
        plt.show()
    else:
        fig.savefig(output_file_path)
        print(f'saved plots at `{output_file_path}`')


def extract_train_time(message, keyword='Total time: ', sub_keyword=' day'):
    if not message.startswith('Epoch:') or keyword not in message:
        return None

    time_str = message[message.find(keyword) + len(keyword):]
    hours = 0
    if sub_keyword in time_str:
        start_idx = time_str.find(sub_keyword)
        hours = 24 * int(time_str[:start_idx])
        time_str = time_str.split(' ')[-1]
    h, m, s = map(int, time_str.split(':'))
    return ((hours + h) * 60 + m) * 60 + s


def extract_val_acc(message, acc1_str='Acc@1 '):
    if acc1_str not in message:
        return None

    acc1 = float(message[message.find(acc1_str) + len(acc1_str):])
    return acc1


def extract_val_performance(log_lines):
    train_time_list, val_acc1_list = list(), list()
    for line in log_lines:
        elements = line.split('\t')
        if len(elements) < 3:
            continue

        message = elements[3]
        train_time = extract_train_time(message)
        if isinstance(train_time, int):
            train_time_list.append(train_time)
            continue

        val_acc1 = extract_val_acc(message)
        if isinstance(val_acc1, float):
            val_acc1_list.append(val_acc1)
        if 'Training time' in message:
            break
    return train_time_list, val_acc1_list


def visualize_val_performance(log_dict):
    sns.set()
    val_performance_dict = dict()
    for file_path, (log_lines, label) in log_dict.items():
        train_times, val_acc1s = extract_val_performance(log_lines)
        val_performance_dict[file_path] = (train_times, val_acc1s, label)
        xs = list(range(len(val_acc1s)))
        plt.plot(xs, val_acc1s, label=r'${}$'.format(label))

    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Top-1 Validation Accuracy [%]')
    plt.tight_layout()
    plt.show()

    for file_path, (train_times, val_acc1s, label) in val_performance_dict.items():
        accum_train_times = [sum(train_times[:i + 1]) for i in range(len(train_times))]
        plt.plot(accum_train_times, val_acc1s, '-o', label=r'${}$'.format(label))

    plt.legend()
    plt.xlabel('Training time [sec]')
    plt.ylabel('Top-1 Validation Accuracy [%]')
    plt.tight_layout()
    plt.show()


def main(args):
    if args.engine == 'text':
        assert args.logs is not None, '`--logs` is required for text engine'
        log_dict = read_files(args.logs, args.labels)
        if args.task == 'classification':
            visualize_val_performance(log_dict)
        return

    assert args.project is not None and args.runs is not None, \
        '`--project` and `--runs` are required for trackio/wandb engines'
    history_dict = load_run_histories(args.engine, args.project, args.runs, args.labels, wandb_entity=args.entity)
    visualize_metrics(history_dict, args.x, args.y, args.out)


if __name__ == '__main__':
    main(get_args())
