import os
import shutil

from bitstring import ConstBitStream

from pybzparse import Parser, boxes as bx_def

from pyheifconcat.create_container import create_container, parse_args


PWD = "tests_tmp"

if PWD and not os.path.exists(PWD):
    os.makedirs(PWD)

os.chdir(PWD)


def test_create_container():
    container_name = "container_name.bzna"

    try:
        args = parse_args([container_name])

        create_container(args)

        assert os.path.exists(container_name)

        bstr = ConstBitStream(filename=container_name)
        boxes = list(Parser.parse(bstr))
        for box in boxes:
            box.load(bstr)

        assert len(boxes) == 2

        ftyp = boxes.pop(0)
        isinstance(ftyp, bx_def.FTYP)
        assert ftyp.header.type == b"ftyp"
        assert ftyp.major_brand == 1769172845           # b"isom"
        assert ftyp.minor_version == 0
        assert ftyp.compatible_brands == [1652190817,   # b"bzna"
                                          1769172845]   # b"isom"

        mdat = boxes.pop(0)
        isinstance(mdat, bx_def.MDAT)
        assert mdat.header.type == b"mdat"
        # mdat should be using the box extended size field to prevent having to
        # shift data if it is bigger than the limit of the regular box size field
        assert mdat.header.box_ext_size is not None
        assert mdat.header.box_ext_size > 0
        assert mdat.data == b''

    finally:
        shutil.rmtree(".", ignore_errors=True)


def test_parse_args():
    raw_arguments = ["container_name.bzna"]

    try:
        args = parse_args(raw_arguments)
        assert args.container.name == "container_name.bzna"

    finally:
        shutil.rmtree(".", ignore_errors=True)


