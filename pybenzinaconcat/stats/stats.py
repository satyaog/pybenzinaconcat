import argparse
import glob
import os
import subprocess

from jug import CachedFunction, TaskGenerator, mapreduce

from pybenzinaconcat.utils import argsutils


@TaskGenerator
def ffmpeg_data_format_batch(batch):
    batch_format_info = []
    for image_path in batch:
        out = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-i",
                             image_path], stderr=subprocess.PIPE) \
              .stderr.decode('utf-8')
        parse_out = {}
        for i, line in enumerate(out.split('\n')):
            if i == 0:
                _, parse_out["filename"] = line.split("from '")
                # trim `':`
                parse_out["filename"] = parse_out["filename"][:-2]
            elif i == 2:
                _, line = line.split("Video: ")
                line_split = line.split(", ")
                # ex.: mjpeg (Baseline)
                parse_out["format"] = line_split.pop(0)
                # ex.: yuvj420p(pc, bt470bg/unknown/unknown)
                pixformat_all = [line_split.pop(0)]
                while line_split:
                    split = line_split.pop(0)
                    pixformat_all.append(split)
                    if ')' in split:
                        break
                pixformat_all = ', '.join(pixformat_all)
                parse_out["pixformat"], pixformat_extra = \
                    pixformat_all.split('(')
                parse_out["pixformat_extra"] = pixformat_extra.rstrip(')') \
                                               .split(', ')
                # ex.: 2816x2112 [SAR 72:72 DAR 4:3]
                resolution_split = line_split.pop(0).split(' ')
                parse_out["resolution"] = \
                    tuple(int(d) for d in resolution_split.pop(0).split('x'))
                parse_out["resolution_extra"] = ' '.join(resolution_split)
                parse_out["extra"] = line_split
        batch_format_info.append(parse_out)
    return batch_format_info


@TaskGenerator
def stats_reducer(last, batch):
    if isinstance(last, dict):
        stats = last
    else:
        batch = last + batch
        stats = {}
    for element in batch:
        for field in element.keys():
            stats.setdefault(field, {})
            value = element[field]
            if isinstance(value, list):
                value = tuple(value)
            stats[field].setdefault(value, 0)
            stats[field][value] += 1
    return stats


def data_format(src, start=0, size=None, batch_size=1024):
    if isinstance(src, str):
        if '*' not in src:
            src = os.path.join(src, '*')
        files = CachedFunction(glob.glob, src)
        files.sort()
        if size is None:
            size = len(files)
        files = files[start:start + size]
        batches = [[f for f in files[s:s + batch_size]]
                   for s in range(0, len(files), batch_size)]
    else:
        batches = src

    return [ffmpeg_data_format_batch(b) for b in batches]


def stats_aggregate(batches):
    if len(batches) > 1:
        return mapreduce.reduce(stats_reducer, batches, reduce_step=1)
    else:
        return stats_reducer([], next(iter(batches), []))


def build_base_parser():
    parser = argparse.ArgumentParser(description="Benzina stats",
                                     add_help=False,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("_action", metavar="action",
                        choices=list(ACTIONS.keys()), help="action to execute")

    return parser


def build_data_format_parser():
    parser = argparse.ArgumentParser(description="Benzina stats "
                                                 "action: data-format",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("_action", metavar="data-format",
                        help="action to execute")
    parser.add_argument("src", metavar="PATH",
                        help="Glob to use when searching for images to print "
                             "pixel formats")
    parser.add_argument("--start", metavar="IDX", default=0, type=int,
                        help="the start element index to validate in source")
    parser.add_argument("--size", default=None, metavar="NUM", type=int,
                        help="the number of elements to validate")
    parser.add_argument("--batch-size", default=1024, metavar="NUM", type=int,
                        help="the batch size for a single job")

    return parser


def parse_args(argv=None):
    return argsutils.parse_args(ACTIONS_PARSER, argv)


def main(args=None, _=None):
    if args is None:
        args, _ = parse_args()
    else:
        try:
            args, _ = args
        except TypeError:
            pass

    return argsutils.run_action(ACTIONS, **vars(args))


ACTIONS = {"data-format": data_format}
ACTIONS_PARSER = {"data-format": build_data_format_parser(),
                  "_base": build_base_parser()}
