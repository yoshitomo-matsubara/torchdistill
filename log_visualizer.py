import argparse
import os

import matplotlib.pyplot as plt
import seaborn as sns


def get_argparser():
    parser = argparse.ArgumentParser(description='Log visualizer')
    parser.add_argument('--logs', required=True, metavar='N', nargs='+', help='list of log file paths to visualize')
    parser.add_argument('--task', default='classification', help='type of tasks defined in the log files')
    return parser


def read_files(file_paths):
    log_dict = dict()
    for file_path in file_paths:
        with open(os.path.expanduser(file_path), 'r') as fp:
            log_dict[file_path] = [line.strip() for line in fp]
    return log_dict


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
    for file_path, log_lines in log_dict.items():
        train_times, val_acc1s = extract_val_performance(log_lines)
        val_performance_dict[file_path] = (train_times, val_acc1s)
        xs = list(range(len(val_acc1s)))
        plt.plot(xs, val_acc1s, label=os.path.basename(file_path))

    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Top-1 Validation Accuracy [%]')
    plt.show()

    for file_path, (train_times, val_acc1s) in val_performance_dict.items():
        accum_train_times = [sum(train_times[:i + 1]) for i in range(len(train_times))]
        plt.plot(accum_train_times, val_acc1s, '-o', label=os.path.basename(file_path))

    plt.legend()
    plt.xlabel('Training time [sec]')
    plt.ylabel('Top-1 Validation Accuracy [%]')
    plt.show()


def main(args):
    log_dict = read_files(args.logs)
    task = args.task
    if task == 'classification':
        visualize_val_performance(log_dict)


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
