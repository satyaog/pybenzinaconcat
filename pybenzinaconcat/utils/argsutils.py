import argparse
import sys
from collections import namedtuple

FileDesc = namedtuple("FileDesc", ["name", "mode"])


class ChainAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        action = option_string.lstrip('-')
        setattr(namespace, self.dest, [action] + values)


class DatasetAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        dataset_id, ar_format = values.split(':')
        dest = self.dest.lstrip('_')
        setattr(namespace, dest + "_id", dataset_id)
        setattr(namespace, dest + "_format", ar_format)
        delattr(namespace, self.dest) 


class CheckFileType(argparse.FileType):
    def __call__(self, string):
        f = super(CheckFileType, self).__call__(string)
        f.close()
        return FileDesc(f.name, f.mode)


def parse_args(actions_parser, argv=None):
    argv = sys.argv[1:] if argv is None else argv

    if len(argv) == 1 and ("-h" in argv or "--help" in argv):
        actions = list(actions_parser.keys())
        actions.sort()
        for action in actions:
            if action.startswith("_"):
                continue
            actions_parser[action].print_help()
        exit(1)

    base_args, _ = actions_parser["_base"].parse_known_args(argv)
    args = actions_parser.get(base_args._action).parse_args(argv)
    try:
        argv = args._chain or tuple()
        del args._chain
    except AttributeError:
        argv = tuple()
    return args, argv


def run_action(actions, _action=None, **kwargs):
    return actions.get(_action)(**kwargs)
