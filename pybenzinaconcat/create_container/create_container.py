import argparse
import sys

from pybenzinaparse import boxes as bx_def
from pybenzinaparse.headers import BoxHeader


def create_container(container):
    ftyp = bx_def.FTYP(BoxHeader())
    ftyp.header.type = b"ftyp"
    ftyp.major_brand = 1769172845           # b"isom"
    ftyp.minor_version = 0
    ftyp.compatible_brands = [1652190817,   # b"bzna"
                              1769172845]   # b"isom"
    ftyp.refresh_box_size()

    mdat = bx_def.MDAT(BoxHeader())
    mdat.header.type = b"mdat"
    mdat.data = b''

    # Force the usage of box_ext_size in the computation of the box size to
    # prevent having to shift data if it ends up being bigger than the limit of
    # box_size
    mdat.header.box_ext_size = 0
    mdat.refresh_box_size()

    if isinstance(container, str):
        container = open(container, "xb")
    container.write(bytes(ftyp) + bytes(mdat))
    container.close()


def build_parser():
    parser = argparse.ArgumentParser(description="Benzina Create Container",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("container", type=argparse.FileType('xb'),
                        help="the container file to be created")
    return parser


def parse_args(raw_arguments=None):
    argv = sys.argv[1:] if raw_arguments is None else raw_arguments
    args = build_parser().parse_args(argv)
    return args


def main(args=None):
    if args is None:
        args = parse_args()
    create_container(**vars(args))
