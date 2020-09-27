import argparse

import torch

from myutils.common.file_util import check_if_exists, make_parent_dirs


def get_argparser():
    parser = argparse.ArgumentParser(description='Extracting parameters from checkpoint')
    parser.add_argument('--src', required=True, help='input ckpt file path')
    parser.add_argument('--keys', required=True, nargs='+', help='keys of parameters to be extracted from ckpt')
    parser.add_argument('-use_dict', action='store_true', help='Save as a dict even if # keys = 1')
    parser.add_argument('--dst', required=True, help='output file path')
    return parser


def save_obj(obj, output_file_path):
    make_parent_dirs(output_file_path)
    torch.save(obj, output_file_path)


def main(args):
    in_ckpt_file_path = args.src
    if not check_if_exists(in_ckpt_file_path):
        print('ckpt file is not found at `{}`'.format(in_ckpt_file_path))
        return

    src_ckpt = torch.load(in_ckpt_file_path, map_location='cpu')
    out_ckpt_file_path = args.dst
    if len(args.keys) == 1 and not args.use_dict:
        key = args.keys[0]
        if key in src_ckpt:
            save_obj(src_ckpt[key], out_ckpt_file_path)
        else:
            print('Parameter key `{}` was not found'.format(key))
        return

    dst_ckpt = dict()
    for key in args.keys:
        if key in src_ckpt:
            dst_ckpt[key] = src_ckpt[key]
        else:
            print('Parameter key `{}` was not found'.format(key))

    if len(dst_ckpt) > 0:
        save_obj(dst_ckpt, out_ckpt_file_path)


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
