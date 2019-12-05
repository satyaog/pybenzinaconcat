import hashlib
import os
import shutil

from bitstring import ConstBitStream

from pybzparse import Parser
from pybzparse.utils import get_trak_sample_bytes, find_boxes

from pyheifconcat.index_metadata import index_metadata, parse_args

DATA_DIR = os.path.abspath("test_datasets")

PWD = "tests_tmp"

if PWD and not os.path.exists(PWD):
    os.makedirs(PWD)

os.chdir(PWD)


def _md5(filename):
    with open(filename, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def test_index_metadata():
    container_filename = "concat.bzna"
    shutil.copyfile(os.path.join(DATA_DIR, "mini_dataset_to_concat/concat.bzna"),
                    container_filename, follow_symlinks=True)

    try:
        args = parse_args([container_filename])

        index_metadata(args)

        assert not os.path.exists(container_filename + ".moov")

        bstr = ConstBitStream(filename=container_filename)
        boxes = list(Parser.parse(bstr))

        # mdat should be using the box extended size field to prevent having to
        # shift data if it is bigger than the limit of the regular box size field
        mdat = next(find_boxes(boxes, b"mdat"))
        assert mdat.header.box_ext_size is not None

        moov = next(find_boxes(boxes, b"moov"))
        moov.load(bstr)

        explicit_targets = [b'\x01\x00\x00\x00\x00\x00\x00\x00',
                            b'\x01\x00\x00\x00\x00\x00\x00\x00',
                            b'\x01\x00\x00\x00\x00\x00\x00\x00',
                            b'\x02\x00\x00\x00\x00\x00\x00\x00',
                            b'\x02\x00\x00\x00\x00\x00\x00\x00',
                            b'\x02\x00\x00\x00\x00\x00\x00\x00',
                            b'\x00\x00\x00\x00\x00\x00\x00\x00',
                            b'\x00\x00\x00\x00\x00\x00\x00\x00',
                            b'\x00\x00\x00\x00\x00\x00\x00\x00',
                            None]

        explicit_filenames = [b'n02100735_7054.JPEG',
                              b'n02100735_7553.JPEG',
                              b'n02100735_8211.JPEG',
                              b'n02110185_2014.JPEG',
                              b'n02110185_679.JPEG',
                              b'n02110185_7939.JPEG',
                              b'n02119789_11296.JPEG',
                              b'n02119789_4903.JPEG',
                              b'n02119789_6970.JPEG',
                              b'n02100735_8211_fake_no_target.JPEG']

        samples = [get_trak_sample_bytes(bstr, moov.boxes, b"bzna_input\0", i) for i in range(10)]
        targets = [get_trak_sample_bytes(bstr, moov.boxes, b"bzna_target\0", i) for i in range(10)]
        filenames = [get_trak_sample_bytes(bstr, moov.boxes, b"bzna_fname\0", i) for i in range(10)]
        thumbs = [get_trak_sample_bytes(bstr, moov.boxes, b"bzna_thumb\0", i) for i in range(10)]

        for i, (sample, target, filename, thumb) in enumerate(zip(samples, targets, filenames, thumbs)):
            sample_bstr = ConstBitStream(bytes=sample)
            sample_moov = next(find_boxes(Parser.parse(sample_bstr), b"moov"))
            sample_moov.load(sample_bstr)
            sample_mp4_filename = os.path.splitext(filename.decode("utf-8"))[0] + ".mp4"
            assert hashlib.md5(sample).hexdigest() == \
                   _md5(os.path.join(DATA_DIR, "mini_dataset_to_transcode",
                                     sample_mp4_filename))
            assert target == explicit_targets[i]
            assert target == get_trak_sample_bytes(sample_bstr, sample_moov.boxes, b"bzna_target\0", 0)
            assert filename == explicit_filenames[i]
            assert filename == get_trak_sample_bytes(sample_bstr, sample_moov.boxes, b"bzna_fname\0", 0)
            assert thumb == get_trak_sample_bytes(sample_bstr, sample_moov.boxes, b"bzna_thumb\0", 0)

    finally:
        shutil.rmtree(".", ignore_errors=True)


def test_index_metadata_second_pass():
    container_filename = "concat_indexed.bzna"
    shutil.copyfile(os.path.join(DATA_DIR, "mini_dataset_to_concat/concat_indexed.bzna"),
                    container_filename, follow_symlinks=True)

    try:
        args = parse_args([container_filename])

        md5_before = _md5(container_filename)

        index_metadata(args)

        assert not os.path.exists(container_filename + ".moov")

        md5_after = _md5(container_filename)

        assert md5_before == md5_after

    finally:
        shutil.rmtree(".", ignore_errors=True)


def test_parse_args():
    raw_arguments = ["container_name.bzna"]

    try:
        with open(raw_arguments[0], "wb"):
            pass
        
        args = parse_args(raw_arguments)
        assert args.container.name == "container_name.bzna"

    finally:
        shutil.rmtree(".", ignore_errors=True)
