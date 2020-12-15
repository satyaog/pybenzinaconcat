import copy
import os
import shutil
import subprocess

import jug
from jug.task import recursive_dependencies
from jug.tests.task_reset import task_reset_at_exit, task_reset
# task_reset is a bit of a hack and needs task_reset_at_exit to be imported
# the line below is only to prevent any complaints from code analysers
task_reset_at_exit=task_reset_at_exit

from pybenzinaconcat import extract
from pybenzinaconcat.stats import data_format, stats_aggregate
from pybenzinaconcat.stats.stats import ffmpeg_data_format, stats_reducer

TESTS_WORKING_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(TESTS_WORKING_DIR, "test_datasets")

PWD = os.path.join(TESTS_WORKING_DIR, "tests_tmp")

if PWD and not os.path.exists(PWD):
    os.makedirs(PWD)

os.chdir(PWD)


def _run_tasks(tasks) -> list:
    for task in tasks:
        _run_tasks(recursive_dependencies(task))
        if not task.can_load() and task.can_run():
            task.run()
    return tasks


def test_ffmpeg_data_format():
    outs = [
        "Input #0, image2, from './extract/000000000000.n01631663_3978.JPEG':\n" +
        "  Duration: 00:00:00.04, start: 0.000000, bitrate: 18301 kb/s\n" +
        "    Stream #0:0: Video: mjpeg, yuvj422p(pc, bt470bg/unknown/unknown), 499x339 [SAR 72:72 DAR 499:339], 25 tbr, 25 tbn, 25 tbc\n" +
        "At least one output file must be specified\n"
        "",
        "Input #0, image2, from './extract/000000000023.n02795169_169.JPEG':\n" +
        "  Duration: 00:00:00.04, start: 0.000000, bitrate: 22033 kb/s\n" +
        "    Stream #0:0: Video: mjpeg, gray(bt470bg/unknown/unknown), 500x334 [SAR 300:300 DAR 250:167], 25 tbr, 25 tbn, 25 tbc\n" +
        "At least one output file must be specified\n"
        "",
        "[mjpeg @ 0x556a3e2e5e80] Invalid TIFF tag type 0 found for GPSVersionID with size 1330403661\n"
        "Input #0, image2, from './extract/000000000048.n03089624_6573.JPEG':\n"
        "  Duration: 00:00:00.04, start: 0.000000, bitrate: 13362 kb/s\n"
        "    Stream #0:0: Video: mjpeg, yuvj420p(pc, bt470bg/unknown/unknown), 480x360 [SAR 314:314 DAR 4:3], 25 tbr, 25 tbn, 25 tbc\n"
        "At least one output file must be specified\n"
        ""
    ]
    expects = [
        {'filename': './extract/000000000000.n01631663_3978.JPEG',
         'format': 'mjpeg',
         'pixformat': 'yuvj422p',
         'pixformat_extra': ['pc', 'bt470bg/unknown/unknown'],
         'resolution': (499, 339),
         'resolution_extra': '[SAR 72:72 DAR 499:339]',
         'extra': ['25 tbr', '25 tbn', '25 tbc']},
        {'filename': './extract/000000000023.n02795169_169.JPEG',
         'format': 'mjpeg',
         'pixformat': 'gray',
         'pixformat_extra': ['bt470bg/unknown/unknown'],
         'resolution': (500, 334),
         'resolution_extra': '[SAR 300:300 DAR 250:167]',
         'extra': ['25 tbr', '25 tbn', '25 tbc']},
        {'filename': './extract/000000000048.n03089624_6573.JPEG',
         'format': 'mjpeg',
         'pixformat': 'yuvj420p',
         'pixformat_extra': ['pc', 'bt470bg/unknown/unknown'],
         'resolution': (480, 360),
         'resolution_extra': '[SAR 314:314 DAR 4:3]',
         'extra': ['25 tbr', '25 tbn', '25 tbc']}
    ]

    for out, expect in zip(outs, expects):
        assert ffmpeg_data_format(out) == expect


@task_reset
def test_data_format():
    src = os.path.join(DATA_DIR, "dev_im_net/dev_im_net.tar")
    dest = "output/dir/extract/"

    try:
        extracted_filepaths = extract(src, dest, "imagenet", "tar", indices=10,
                                      size=15)
        batches = _run_tasks(data_format(extracted_filepaths))
        for batch in jug.value(batches):
            assert len(batch) == 15
            for im_format in batch:
                assert os.path.isfile(im_format["filename"])
                assert im_format["format"]
                assert im_format["pixformat"]
                assert len(im_format["resolution"]) == 2

        assert jug.value(_run_tasks(data_format(dest + "*.JPEG", start=0,
                                                size=15))) == \
            jug.value(batches)

        extracted_filepaths = extract(src, dest, "imagenet", "tar", indices=10,
                                      size=15, batch_size=5)
        batches = _run_tasks(data_format(extracted_filepaths))
        for batch in jug.value(batches):
            assert len(batch) == 5
            for im_format in batch:
                assert os.path.isfile(im_format["filename"])
                assert im_format["format"]
                assert im_format["pixformat"]
                assert len(im_format["resolution"]) == 2

    finally:
        shutil.rmtree(".", ignore_errors=True)


def test_python_data_format():
    src = os.path.join(DATA_DIR, "dev_im_net/dev_im_net.tar")
    dest = "output/dir/extract/"

    extract_args = ["--jugdir", "jugdir", "--",
                    "extract", src, dest, "imagenet:tar", "--indices", "10",
                    "--size", "17", "--batch-size", "5"]

    data_format_args = ["--jugdir", "jugdir", "--",
                        "data-format", dest + "*.JPEG", "--start", "0",
                        "--size", "17", "--batch-size", "5"]

    try:         
        processes = [subprocess.Popen(["python3", "../../pybenzinaconcat"] +
                                      extract_args) for _ in range(4)]

        for p in processes:
            p.wait(30)
            assert p.returncode == 0

        processes = [
            subprocess.Popen(["python3", "../../pybenzinaconcat/stats"] +
                             data_format_args) for _ in range(4)]

        for p in processes:
            p.wait(30)
            assert p.returncode == 0

        jug.set_jugdir("jugdir")
        batches = data_format(dest + "*.JPEG", start=0, size=17, batch_size=5)
        for batch in jug.value(batches):
            assert len(batch) in {2, 5}
            for im_format in batch:
                assert os.path.isfile(im_format["filename"])
                assert im_format["format"]
                assert im_format["pixformat"]
                assert len(im_format["resolution"]) == 2

    finally:
        shutil.rmtree(".", ignore_errors=True)


@task_reset
def test_stats_aggregate():
    src = os.path.join(DATA_DIR, "dev_im_net/dev_im_net.tar")
    dest = "output/dir/extract/"

    try:
        extracted_filepaths = extract(src, dest, "imagenet", "tar", indices=10,
                                      size=15)
        batches = data_format(extracted_filepaths)
        stats = _run_tasks([stats_aggregate(batches)])[0]
        stats = jug.value(stats)

        assert len(stats["format"]) <= 15
        assert len(stats["pixformat"]) <= 15
        assert len(stats["resolution"]) <= 15

        for field in stats.keys():
            assert sum(v for v in stats[field].values()) == 15

        extracted_filepaths = extract(src, dest, "imagenet", "tar", indices=10,
                                      size=15, batch_size=5)
        batches = data_format(extracted_filepaths)
        assert jug.value(
            _run_tasks([stats_aggregate(batches, reduce_step=5)])[0]) == stats

    finally:
        shutil.rmtree(".", ignore_errors=True)


@task_reset
def test_stats_aggregate_combine():
    src = os.path.join(DATA_DIR, "dev_im_net/dev_im_net.tar")
    dest = "output/dir/extract/"

    try:
        extracted_filepaths = extract(src, dest, "imagenet", "tar", indices=10,
                                      size=15)
        batches = data_format(extracted_filepaths)
        stats = _run_tasks([stats_aggregate(batches)])[0]
        stats = jug.value(stats)
        stats_combined = _run_tasks([stats_reducer(stats,
                                                   copy.deepcopy(stats))])[0]
        stats_combined = jug.value(stats_combined)

        for field, value in stats_combined.items():
            for combined_count, count in zip(value.values(),
                                             stats[field].values()):
                assert combined_count == 2 * count

    finally:
        shutil.rmtree(".", ignore_errors=True)
