import os
import shutil
import subprocess

import jug
from jug.task import recursive_dependencies
from jug.tests.task_reset import task_reset

from pybenzinaconcat import extract
from pybenzinaconcat.stats import data_format, stats_aggregate

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


@task_reset
def test_data_format():
    src = os.path.join(DATA_DIR, "dev_im_net/dev_im_net.tar")
    dest = "output/dir/extract/"

    try:
        extracted_filepaths = extract(src, dest, "imagenet", "tar", start=10,
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

        extracted_filepaths = extract(src, dest, "imagenet", "tar", start=10,
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
                    "extract", src, dest, "imagenet:tar", "--start", "10",
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
        extracted_filepaths = extract(src, dest, "imagenet", "tar", start=10,
                                      size=15)
        batches = data_format(extracted_filepaths)
        stats = _run_tasks([stats_aggregate(batches)])
        stats = jug.value(stats[0])

        assert len(stats["filename"]) == 15
        assert len(stats["format"]) <= 15
        assert len(stats["pixformat"]) <= 15
        assert len(stats["resolution"]) <= 15

        for field in stats.keys():
            assert sum(v for v in stats[field].values()) == 15

        extracted_filepaths = extract(src, dest, "imagenet", "tar", start=10,
                                      size=15, batch_size=5)
        batches = data_format(extracted_filepaths)
        assert jug.value(_run_tasks([stats_aggregate(batches)])[0]) == stats

    finally:
        shutil.rmtree(".", ignore_errors=True)
