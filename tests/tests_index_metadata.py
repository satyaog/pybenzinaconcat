import hashlib
import glob
import os
import shutil
import subprocess

from bitstring import ConstBitStream
import jug
from jug.task import recursive_dependencies
from jug.tests.task_reset import task_reset_at_exit, task_reset
# task_reset is a bit of a hack and needs task_reset_at_exit to be imported
# the line below is only to prevent any complaints from code analysers
task_reset_at_exit=task_reset_at_exit

from pybenzinaparse import Parser
from pybenzinaparse.utils import get_trak_sample_bytes, find_boxes

from pybenzinaconcat.index_metadata import index_metadata, parse_args

TESTS_WORKING_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(TESTS_WORKING_DIR, "test_datasets")

os.environ["PATH"] = ':'.join([os.environ["PATH"],
                               os.path.join(TESTS_WORKING_DIR, "mocks")])

PWD = os.path.join(TESTS_WORKING_DIR, "tests_tmp")

if PWD and not os.path.exists(PWD):
    os.makedirs(PWD)

os.chdir(PWD)


def _run_tasks(tasks) -> list:
    for task in tasks:
        _run_tasks(recursive_dependencies(task))
        if not task.can_load() and task.can_run():
            task.run()
    return jug.value(tasks)


def _md5(filename):
    with open(filename, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def _create_container():
    src = os.path.join(DATA_DIR, "dev_im_net/dev_im_net.tar")
    extract_dest = "output/dir/extract/"

    transcode_dest = "output/dir/transcode"
    upload_dir = os.path.join(transcode_dest, "upload")
    queue_dir = os.path.join(transcode_dest, "queue")
    transcode_tmp = "tmp/"

    concat_file = "output/dir/concat.bzna"

    if upload_dir and not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    if queue_dir and not os.path.exists(queue_dir):
        os.makedirs(queue_dir)

    extract_args = ["extract", src, extract_dest, "imagenet:tar",
                    "--indices", "5", "--size", "10", "--batch-size", "5"]
    transcode_args = ["--transcode", transcode_dest, "--mp4",
                      "--tmp", transcode_tmp]

    subprocess.run(
        ["python3", "../../pybenzinaconcat/create_container", "--",
         concat_file], check=True)

    processes = [subprocess.Popen(
        ["python3", "../../pybenzinaconcat", "--"] + extract_args)
        for _ in range(2)]

    try:
        for p in processes:
            p.wait(30)
            assert p.returncode == 0
    finally:
        for p in processes:
            p.kill()

    # Fake a target-less input
    extracted_files = glob.glob(os.path.join(extract_dest, "*.JPEG"))
    extracted_files.sort()
    try:
        os.remove(extracted_files[-1] + ".target")
    except FileNotFoundError:
        pass

    processes = [subprocess.Popen(
        ["python3", "../../pybenzinaconcat", "--"] + extract_args +
        transcode_args) for _ in range(3)]

    try:
        for p in processes:
            p.wait(30)
            assert p.returncode == 0
    finally:
        for p in processes:
            p.kill()

    # backup transcoded files to compare later
    for fn in glob.glob(os.path.join(queue_dir, "*")):
        shutil.copy(fn, transcode_tmp)
    transcoded_files = glob.glob(os.path.join(transcode_tmp, "*"))
    transcoded_files.sort()

    subprocess.run(
        ["python3", "../../pybenzinaconcat", "--"] + extract_args +
        transcode_args + ["--concat", concat_file], check=True)

    return concat_file, transcoded_files


@task_reset
def test_index_metadata():
    try:
        container_filename, transcoded_files = _create_container()

        _run_tasks([index_metadata(container_filename)])

        assert not os.path.exists(container_filename + ".moov")

        bstr = ConstBitStream(filename=container_filename)
        boxes = list(Parser.parse(bstr))

        # mdat should be using the box extended size field to prevent having to
        # shift data if it is bigger than the limit of the regular box size field
        mdat = next(find_boxes(boxes, b"mdat"))
        assert mdat.header.box_ext_size is not None

        moov = next(find_boxes(boxes, b"moov"))
        moov.load(bstr)

        explicit_targets = [b'\x00\x00\x00\x00\x00\x00\x00\x00',
                            b'\x00\x00\x00\x00\x00\x00\x00\x00',
                            b'\x00\x00\x00\x00\x00\x00\x00\x00',
                            b'\x00\x00\x00\x00\x00\x00\x00\x00',
                            b'\x00\x00\x00\x00\x00\x00\x00\x00',
                            b'\x01\x00\x00\x00\x00\x00\x00\x00',
                            b'\x01\x00\x00\x00\x00\x00\x00\x00',
                            b'\x01\x00\x00\x00\x00\x00\x00\x00',
                            b'\x01\x00\x00\x00\x00\x00\x00\x00',
                            None]

        explicit_filenames = [b'n01440764_11155.JPEG\x00',
                              b'n01440764_7719.JPEG\x00',
                              b'n01440764_7304.JPEG\x00',
                              b'n01440764_8469.JPEG\x00',
                              b'n01440764_6432.JPEG\x00',
                              b'n01443537_2772.JPEG\x00',
                              b'n01443537_1029.JPEG\x00',
                              b'n01443537_1955.JPEG\x00',
                              b'n01443537_962.JPEG\x00',
                              b'n01443537_2563.JPEG\x00']

        samples = [get_trak_sample_bytes(bstr, moov.boxes, b"bzna_input\0", i) for i in range(10)]
        targets = [get_trak_sample_bytes(bstr, moov.boxes, b"bzna_target\0", i) for i in range(10)]
        filenames = [get_trak_sample_bytes(bstr, moov.boxes, b"bzna_fname\0", i) for i in range(10)]
        thumbs = [get_trak_sample_bytes(bstr, moov.boxes, b"bzna_thumb\0", i) for i in range(10)]

        for i, (sample, target, filename, thumb, transcoded_file) in \
                enumerate(zip(samples, targets, filenames, thumbs,
                              transcoded_files)):
            sample_bstr = ConstBitStream(bytes=sample)
            sample_moov = next(find_boxes(Parser.parse(sample_bstr), b"moov"))
            sample_moov.load(sample_bstr)
            sample_transcoded_filename = filename.decode("utf-8")[:-1] + \
                                         ".transcoded"
            assert transcoded_file.endswith(sample_transcoded_filename)
            assert hashlib.md5(sample).hexdigest() == \
                   _md5(transcoded_file)
            assert target == explicit_targets[i]
            assert target == get_trak_sample_bytes(sample_bstr, sample_moov.boxes, b"bzna_target\0", 0)
            assert filename == explicit_filenames[i]
            assert filename == get_trak_sample_bytes(sample_bstr, sample_moov.boxes, b"bzna_fname\0", 0)
            assert thumb == get_trak_sample_bytes(sample_bstr, sample_moov.boxes, b"bzna_thumb\0", 0)

    finally:
        shutil.rmtree(".", ignore_errors=True)


def test_index_metadata_second_pass():
    try:
        container_filename, transcoded_files = _create_container()

        subprocess.run(
            ["python3", "../../pybenzinaconcat/index_metadata", "--",
             container_filename], check=True)

        md5_before = _md5(container_filename)

        subprocess.run(
            ["python3", "../../pybenzinaconcat/index_metadata", "--",
             container_filename], check=True)

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
