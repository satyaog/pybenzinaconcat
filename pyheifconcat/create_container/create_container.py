import argparse
import sys

from pybzparse import boxes as bx_def
from pybzparse.headers import BoxHeader


def create_container(args):
    ftyp = bx_def.FTYP(BoxHeader())
    ftyp.header.type = b"ftyp"
    ftyp.major_brand = 1769172845          # b"isom"
    ftyp.minor_version = 0
    ftyp.compatible_brands = [1652190817,  # b"bzna"
                              1769172845]  # b"isom"
    ftyp.refresh_box_size()

    mdat = bx_def.MDAT(BoxHeader())
    mdat.header.type = b"mdat"
    mdat.data = b''
    mdat.refresh_box_size()

    args.container.write(bytes(ftyp) + bytes(mdat))


def build_parser():
    parser = argparse.ArgumentParser(description="Benzina Create Container",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("container", type=argparse.FileType('xwb'),
                        help="the container file to be created")
    return parser


def parse_args(raw_arguments=None):
    argv = sys.argv[1:] if raw_arguments is None else raw_arguments
    args = build_parser().parse_args(argv)
    return args
