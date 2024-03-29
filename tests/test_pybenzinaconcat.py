import glob
import os
import shutil
import subprocess

import jug
from jug.task import recursive_dependencies
from jug.tests.task_reset import task_reset_at_exit, task_reset
# task_reset is a bit of a hack and needs task_reset_at_exit to be imported
# the line below is only to prevent any complaints from code analysers
task_reset_at_exit=task_reset_at_exit

from pybenzinaconcat.utils.fnutils import FILENAME_TEMPLATE, _get_file_index, \
    get_clean_filepath, _is_transcoded, _make_index_filepath, \
    _make_transcoded_filepath
from pybenzinaconcat.benzinaconcat import concat, extract, transcode, \
    parse_args, main

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


def _prepare_concat_data(to_concat_filepaths, queue_dir, dest_dir):
    if queue_dir and not os.path.exists(queue_dir):
        os.makedirs(queue_dir)
    if dest_dir and not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    processes = []
    for i, to_concat_filepath in enumerate(to_concat_filepaths):
        processes.append(
            subprocess.Popen(["dd", "if=/dev/urandom",
                              "of={}".format(to_concat_filepath),
                              "bs=1000", "count=5000"]))

    files_bytes = []
    for i, process in enumerate(processes):
        process.wait()
        assert process.returncode == 0
        with open(to_concat_filepaths[i], "rb") as file:
            files_bytes.append(file.read())
            assert len(files_bytes[i]) == 5000 * 1000

    return files_bytes


def _test_concat(to_concat_filepaths, concat_filepaths, files_bytes, dest):
    with open(dest, "rb") as file:
        assert file.read() == b''.join(files_bytes)

    nb_files_to_skip = len(files_bytes) - len(concat_filepaths)
    assert list(concat_filepaths) == to_concat_filepaths[nb_files_to_skip:]


@task_reset
def test_concat():
    """Test that files are concatenated sequentially"""
    src = "input/dir/"
    dest = "output/dir/concat.bzna"
    src_dir = os.path.dirname(src)
    dest_dir = os.path.dirname(dest)
    queue_dir = os.path.join(src_dir, "queue")

    to_concat_filepaths = []
    for i in range(10):
        to_concat_filepaths.append(os.path.join(queue_dir,
                                                "file_{}_5mb.img.transcoded"
                                                .format(i)))

    args, _ = parse_args(["concat", src, dest])
    del args._action

    try:
        files_bytes = _prepare_concat_data(to_concat_filepaths, queue_dir,
                                           dest_dir)
        concat_filepaths, _ = _run_tasks(concat(**vars(args)))[0]
        _test_concat(to_concat_filepaths, concat_filepaths, files_bytes,
                     args.dest)

    finally:
        shutil.rmtree(".", ignore_errors=True)


@task_reset
def test_concat_index_completed_3():
    """Test that files are concatenated only once"""
    src = "input/dir/"
    dest = "output/dir/concat.bzna"
    src_dir = os.path.dirname(src)
    dest_dir = os.path.dirname(dest)
    queue_dir = os.path.join(src_dir, "queue")

    to_concat_filepaths = []
    for i in range(10):
        filepath = os.path.join(queue_dir, "file_{}_5mb.img.transcoded"
                                           .format(i))
        filepath = _make_index_filepath(filepath, i)
        to_concat_filepaths.append(filepath)

    args, _ = parse_args(["concat", src, dest])
    del args._action

    try:
        files_bytes = _prepare_concat_data(to_concat_filepaths[:3], queue_dir,
                                           dest_dir)
        concat_filepaths, _ = _run_tasks(concat(**vars(args)))[0]
        _test_concat(to_concat_filepaths[:3], concat_filepaths, files_bytes,
                     args.dest)

        files_bytes += _prepare_concat_data(to_concat_filepaths, queue_dir,
                                            dest_dir)[3:]
        concat_filepaths, remain_filepaths = \
            _run_tasks(concat(**vars(args)))[0]
        assert remain_filepaths == to_concat_filepaths[:3]
        _test_concat(to_concat_filepaths, remain_filepaths + concat_filepaths,
                     files_bytes, args.dest)

    finally:
        shutil.rmtree(".", ignore_errors=True)


@task_reset
def test_concat_no_queue():
    """Test that directories hierarchy is created even when there are no files
    to concatenate
    """
    src = "input/dir/"
    dest = "output/dir/concat.bzna"
    src_dir = os.path.dirname(src)

    args, _ = parse_args(["concat", src, dest])
    del args._action

    try:
        assert _run_tasks(concat(**vars(args)))[0] == ([], [])

        assert os.path.exists(os.path.join(src_dir, "upload/"))
        assert os.path.exists(os.path.join(src_dir, "queue/"))

        assert not os.path.exists(args.dest)

    finally:
        shutil.rmtree(".", ignore_errors=True)


@task_reset
def test_pybenzinaconcat_concat():
    """Test that files are concatenated sequentially using the main
    entry point
    """
    src = "input/dir/"
    dest = "output/dir/concat.bzna"
    src_dir = os.path.dirname(src)
    dest_dir = os.path.dirname(dest)
    queue_dir = os.path.join(src_dir, "queue")

    to_concat_filepaths = []
    for i in range(10):
        to_concat_filepaths.append(os.path.join(queue_dir,
                                                "file_{}_5mb.img.transcoded"
                                                .format(i)))

    args, argv = parse_args(["concat", src, dest])

    try:
        files_bytes = _prepare_concat_data(to_concat_filepaths, queue_dir,
                                           dest_dir)
        concat_filepaths, _ = _run_tasks(main(args, argv))[0]
        _test_concat(to_concat_filepaths, concat_filepaths, files_bytes, dest)

    finally:
        shutil.rmtree(".", ignore_errors=True)


def _prepare_transcode_data(tmp_filepaths, tmp_dir, dest_dir):
    upload_dir = os.path.join(dest_dir, "upload")
    queue_dir = os.path.join(dest_dir, "queue")

    if tmp_dir and not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    if upload_dir and not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    if queue_dir and not os.path.exists(queue_dir):
        os.makedirs(queue_dir)

    processes = []
    for i, tmp_filepath in enumerate(tmp_filepaths):
        processes.append(
            subprocess.Popen(["dd", "if=/dev/urandom",
                              "of={}".format(tmp_filepath),
                              "bs=1000", "count=5000"]))

    files_bytes = []
    for i, process in enumerate(processes):
        process.wait()
        assert process.returncode == 0
        with open(tmp_filepaths[i], "rb") as file:
            files_bytes.append(file.read())
            assert len(files_bytes[i]) == 5000 * 1000

    return files_bytes


def _prepare_transcode_target_data(tmp_filepaths):
    tragets_bytes = []
    for i, tmp_filepath in enumerate(tmp_filepaths):
        tragets_bytes.append(i.to_bytes(8, byteorder="little"))
        try:
            with open(tmp_filepath + ".target", "xb") as file:
                file.write(tragets_bytes[-1])
        except FileExistsError:
            continue

    return tragets_bytes


def _test_trancode(tmp_filepaths, dest_dir, files_bytes, targets_bytes):
    upload_dir = os.path.join(dest_dir, "upload")
    queue_dir = os.path.join(dest_dir, "queue")

    assert len(glob.glob(os.path.join(upload_dir, '*'))) == 0

    queued_list = glob.glob(os.path.join(queue_dir, '*'))
    queued_list.sort()
    assert len(queued_list) == len(tmp_filepaths)

    for i, filepath in enumerate(queued_list):
        with open(filepath, "rb") as file:
            file_bytes = file.read()
        assert file_bytes == files_bytes[i] + targets_bytes[i]


@task_reset
def test_trancode():
    """Test that files are transcoded to the correct location"""
    dest = "output/dir/"
    dest_dir = os.path.dirname(dest)
    tmp_dir = "tmp/"

    tmp_filepaths = []
    for i in range(10):
        tmp_filepaths.append(os.path.join(tmp_dir,
                                          "file_{}_5mb.img".format(i)))

    args, _ = parse_args(["transcode", ','.join(tmp_filepaths), dest])
    del args._action

    try:
        files_bytes = _prepare_transcode_data(tmp_filepaths, tmp_dir, dest_dir)
        targets_bytes = [b'' for _ in range(len(tmp_filepaths))]
        transcode_filepaths = _run_tasks([transcode(**vars(args))])[0]
        assert len(transcode_filepaths) == len(tmp_filepaths)
        _test_trancode(tmp_filepaths, dest_dir, files_bytes, targets_bytes)

    finally:
        shutil.rmtree(".", ignore_errors=True)


@task_reset
def test_trancode_index_completed_3():
    """Test that files are transcoded only once"""
    dest = "output/dir/"
    dest_dir = os.path.dirname(dest)
    tmp_dir = "tmp/"

    tmp_filepaths = []
    for i in range(10):
        filepath = os.path.join(tmp_dir, "file_{}_5mb.img"
                                         .format(i))
        filepath = _make_index_filepath(filepath, i)
        tmp_filepaths.append(filepath)

    try:
        files_bytes = _prepare_transcode_data(tmp_filepaths[:3], tmp_dir,
                                              dest_dir)
        targets_bytes = [b'' for _ in range(len(tmp_filepaths))]
        args, _ = parse_args(["transcode", ','.join(tmp_filepaths[:3]), dest])
        del args._action
        transcode_filepaths = _run_tasks([transcode(**vars(args))])[0]
        assert len(transcode_filepaths) == len(tmp_filepaths[:3])
        _test_trancode(tmp_filepaths[:3], dest_dir, files_bytes,
                       targets_bytes[:3])

        files_bytes += _prepare_transcode_data(tmp_filepaths, tmp_dir,
                                               dest_dir)[3:]
        tasks = []
        for i in range(0, len(tmp_filepaths), 3):
            args, _ = parse_args(["transcode", ','.join(tmp_filepaths[i:i+3]), dest])
            del args._action
            tasks.append(transcode(**vars(args)))
        transcode_filepaths = _run_tasks(tasks)
        transcode_filepaths = [path for l in transcode_filepaths for path in l]
        assert len(transcode_filepaths) == len(tmp_filepaths)
        _test_trancode(tmp_filepaths, dest_dir, files_bytes, targets_bytes)

    finally:
        shutil.rmtree(".", ignore_errors=True)


@task_reset
def test_trancode_target_data():
    """Test that files are transcoded with their target datum"""
    dest = "output/dir/"
    dest_dir = os.path.dirname(dest)
    tmp_dir = "tmp/"

    tmp_filepaths = []
    for i in range(10):
        tmp_filepaths.append(os.path.join(tmp_dir,
                                          "file_{}_5mb.img".format(i)))

    args, _ = parse_args(["transcode", ','.join(tmp_filepaths), dest])
    del args._action

    try:
        files_bytes = _prepare_transcode_data(tmp_filepaths, tmp_dir, dest_dir)
        targets_bytes = _prepare_transcode_target_data(tmp_filepaths)
        transcode_filepaths = _run_tasks([transcode(**vars(args))])[0]
        assert len(transcode_filepaths) == len(tmp_filepaths)
        _test_trancode(tmp_filepaths, dest_dir, files_bytes, targets_bytes)

    finally:
        shutil.rmtree(".", ignore_errors=True)


@task_reset
def test_trancode_target_data_completed_3():
    """Test that files are transcoded only once with their target datum"""
    dest = "output/dir/"
    dest_dir = os.path.dirname(dest)
    tmp_dir = "tmp/"

    tmp_filepaths = []
    for i in range(10):
        tmp_filepaths.append(os.path.join(tmp_dir,
                                          "file_{}_5mb.img".format(i)))

    try:
        files_bytes = _prepare_transcode_data(tmp_filepaths[:3], tmp_dir,
                                              dest_dir)
        targets_bytes = _prepare_transcode_target_data(tmp_filepaths[:3])
        args, _ = parse_args(["transcode", ','.join(tmp_filepaths[:3]), dest])
        del args._action
        transcode_filepaths = _run_tasks([transcode(**vars(args))])[0]
        assert len(transcode_filepaths) == len(tmp_filepaths[:3])
        _test_trancode(tmp_filepaths[:3], dest_dir, files_bytes,
                       targets_bytes[:3])

        files_bytes += _prepare_transcode_data(tmp_filepaths, tmp_dir,
                                               dest_dir)[3:]
        targets_bytes = _prepare_transcode_target_data(tmp_filepaths)
        tasks = []
        for i in range(0, len(tmp_filepaths), 3):
            args, _ = parse_args(["transcode", ','.join(tmp_filepaths[i:i+3]), dest])
            del args._action
            tasks.append(transcode(**vars(args)))
        transcode_filepaths = _run_tasks(tasks)
        transcode_filepaths = [path for l in transcode_filepaths for path in l]
        assert len(transcode_filepaths) == len(tmp_filepaths)
        _test_trancode(tmp_filepaths, dest_dir, files_bytes, targets_bytes)

    finally:
        shutil.rmtree(".", ignore_errors=True)


@task_reset
def test_trancode_files_list():
    """Test that files are transcoded using a list file"""
    dest = "output/dir/"
    dest_dir = os.path.dirname(dest)
    tmp_dir = "tmp/"

    tmp_filepaths = []
    for i in range(10):
        tmp_filepaths.append(os.path.join(tmp_dir,
                                          "file_{}_5mb.img".format(i)))

    args, _ = parse_args(["transcode", "list", dest])
    del args._action

    try:
        with open("list", "w") as files_list:
            files_list.write('\n'.join(tmp_filepaths[3:]))

        files_bytes = _prepare_transcode_data(tmp_filepaths, tmp_dir, dest_dir)
        targets_bytes = _prepare_transcode_target_data(tmp_filepaths)
        transcode_filepaths = _run_tasks([transcode(**vars(args))])[0]
        assert len(transcode_filepaths) == len(tmp_filepaths[3:])
        _test_trancode(tmp_filepaths[3:], dest_dir, files_bytes[3:],
                       targets_bytes[3:])

    finally:
        shutil.rmtree(".", ignore_errors=True)


@task_reset
def test_trancode_excludes():
    """Test that files are excluded from transcoding using an excludes file"""
    dest = "output/dir/"
    dest_dir = os.path.dirname(dest)
    tmp_dir = "tmp/"

    tmp_filepaths = []
    for i in range(10):
        tmp_filepaths.append(os.path.join(tmp_dir,
                                          "file_{}_5mb.img".format(i)))

    try:
        with open("list", "w") as files_list:
            files_list.write('\n'.join(tmp_filepaths))
        with open("excludes", "w") as excludes:
            excludes.write('\n'.join(tmp_filepaths[:3]))

        args, _ = parse_args(["transcode", "list", dest,
                              "--excludes", "excludes"])
        del args._action

        files_bytes = _prepare_transcode_data(tmp_filepaths, tmp_dir, dest_dir)
        targets_bytes = _prepare_transcode_target_data(tmp_filepaths)
        transcode_filepaths = _run_tasks([transcode(**vars(args))])[0]
        assert len(transcode_filepaths) == len(tmp_filepaths[3:])
        _test_trancode(tmp_filepaths[3:], dest_dir, files_bytes[3:],
                       targets_bytes[3:])

    finally:
        shutil.rmtree(".", ignore_errors=True)


@task_reset
def test_trancode_crf():
    """Test that files are transcoded to the correct location"""
    dest = "output/dir/"
    dest_dir = os.path.dirname(dest)
    tmp_dir = "tmp/"

    tmp_filepaths = []
    for i in range(10):
        tmp_filepaths.append(os.path.join(tmp_dir,
                                          "file_{}_5mb.img".format(i)))

    args, _ = parse_args(["transcode", ','.join(tmp_filepaths), dest,
                          "--crf", "5"])
    del args._action

    try:
        files_bytes = _prepare_transcode_data(tmp_filepaths, tmp_dir, dest_dir)
        targets_bytes = [b'' for _ in range(len(tmp_filepaths))]
        transcode_filepaths = _run_tasks([transcode(**vars(args))])[0]
        assert len(transcode_filepaths) == len(tmp_filepaths)
        _test_trancode(tmp_filepaths, dest_dir, files_bytes, targets_bytes)

    finally:
        shutil.rmtree(".", ignore_errors=True)


@task_reset
def test_pybenzinaconcat_trancode():
    """Test that files are transcoded using the main entry point"""
    dest = "output/dir/"
    dest_dir = os.path.dirname(dest)
    tmp_dir = "tmp/"

    tmp_filepaths = []
    for i in range(10):
        tmp_filepaths.append(os.path.join(tmp_dir,
                                          "file_{}_5mb.img".format(i)))

    args, argv = parse_args(["transcode", ','.join(tmp_filepaths), dest])

    try:
        files_bytes = _prepare_transcode_data(tmp_filepaths, tmp_dir, dest_dir)
        targets_bytes = [b'' for _ in range(len(tmp_filepaths))]
        transcode_filepaths = _run_tasks(main(args, argv))[0]
        assert len(transcode_filepaths) == len(tmp_filepaths)
        _test_trancode(tmp_filepaths, dest_dir, files_bytes, targets_bytes)

    finally:
        shutil.rmtree(".", ignore_errors=True)


@task_reset
def test_pybenzinaconcat_trancode_chain_concat():
    """Test that files are transcoded then concatenated sequentially using the
    main entry point
    """
    dest = "output/dir/"
    dest_dir = os.path.dirname(dest)
    tmp_dir = "tmp/"

    concat_file = "output/dir/concat.bzna"

    tmp_filepaths = []
    for i in range(10):
        tmp_filepaths.append(os.path.join(tmp_dir,
                                          "file_{}_5mb.img".format(i)))

    args, argv = parse_args(["transcode", ','.join(tmp_filepaths), dest,
                             "--concat", concat_file])

    try:
        files_bytes = _prepare_transcode_data(tmp_filepaths, tmp_dir, dest_dir)
        concat_filepaths, _ = _run_tasks(main(args, argv)[0])[0]
        assert len(concat_filepaths) == len(tmp_filepaths)
        with open(concat_file, "rb") as f:
            assert f.read() == b''.join(files_bytes)

    finally:
        shutil.rmtree(".", ignore_errors=True)  


def test_jug_trancode():
    """Test that files are transcoded using jug"""
    dest = "output/dir/"
    dest_dir = os.path.dirname(dest)
    tmp_dir = "tmp/"

    tmp_filepaths = []
    for i in range(10):
        tmp_filepaths.append(os.path.join(tmp_dir,
                                          "file_{}_5mb.img".format(i)))

    args = ["transcode", ','.join(tmp_filepaths), dest]

    try:
        files_bytes = _prepare_transcode_data(tmp_filepaths, tmp_dir, dest_dir)
        targets_bytes = [b'' for _ in range(len(tmp_filepaths))]

        assert subprocess.run(
            ["jug", "status", "--jugdir", "pybenzinaconcat.jugdir/", "--",
             "../../pybenzinaconcat/__main__.py"] + args).returncode == 0

        processes = [subprocess.Popen(
            ["jug", "execute", "--jugdir", "pybenzinaconcat.jugdir/", "--",
             "../../pybenzinaconcat/__main__.py"] + args)
            for _ in range(3)]

        for p in processes:
            p.wait()
            assert p.returncode == 0

        assert subprocess.run(
            ["jug", "status", "--jugdir", "pybenzinaconcat.jugdir/", "--",
             "../../pybenzinaconcat/__main__.py"] + args).returncode > 0

        _test_trancode(tmp_filepaths, dest_dir, files_bytes, targets_bytes)

    finally:
        shutil.rmtree(".", ignore_errors=True)


def test_python_trancode():
    """Test that files are transcoded using python"""
    dest = "output/dir/"
    dest_dir = os.path.dirname(dest)
    tmp_dir = "tmp/"

    tmp_filepaths = []
    for i in range(10):
        tmp_filepaths.append(os.path.join(tmp_dir,
                                          "file_{}_5mb.img".format(i)))

    args = ["transcode", ','.join(tmp_filepaths), dest]

    try:
        files_bytes = _prepare_transcode_data(tmp_filepaths, tmp_dir, dest_dir)
        targets_bytes = [b'' for _ in range(len(tmp_filepaths))]

        subprocess.run(["python3", "../../pybenzinaconcat"] + args, check=True)

        _test_trancode(tmp_filepaths, dest_dir, files_bytes, targets_bytes)

    finally:
        shutil.rmtree(".", ignore_errors=True)


def test_python_trancode_jugdir():
    """Test that files are transcoded using python"""
    dest = "output/dir/"
    dest_dir = os.path.dirname(dest)
    tmp_dir = "tmp/"

    tmp_filepaths = []
    for i in range(10):
        tmp_filepaths.append(os.path.join(tmp_dir,
                                          "file_{}_5mb.img".format(i)))

    args = ["--", "transcode", ','.join(tmp_filepaths), dest]

    try:
        _ = _prepare_transcode_data(tmp_filepaths, tmp_dir, dest_dir)
        subprocess.run(["python3", "../../pybenzinaconcat"] + args, check=True)
        assert os.path.exists("pybenzinaconcat.jugdir/")
        shutil.rmtree("pybenzinaconcat.jugdir/", ignore_errors=True)

        subprocess.run(["python3", "../../pybenzinaconcat",
                        "--jugdir", "pybenzinaconcat.jugdir/"] + args,
                       check=True)
        assert os.path.exists("pybenzinaconcat.jugdir/")
        shutil.rmtree("pybenzinaconcat.jugdir/", ignore_errors=True)

        subprocess.run(["python3", "../../pybenzinaconcat",
                        "--jugdir", "pybenzinaconcat__.jugdir/"] + args,
                       check=True)
        assert not os.path.exists("pybenzinaconcat.jugdir/")
        assert os.path.exists("pybenzinaconcat__.jugdir/")
    finally:
        shutil.rmtree(".", ignore_errors=True)


@task_reset
def test_extract():
    """Test that files are extracted"""
    src = os.path.join(DATA_DIR, "dev_im_net/dev_im_net.tar")
    dest = "output/dir/extract/"
    dest_dir = os.path.dirname(dest)

    args, _ = parse_args(["extract", src, dest, "imagenet:tar"])
    del args._action

    try:
        extracted_filepaths = _run_tasks(extract(**vars(args)))[0]

        queued_list = glob.glob(os.path.join(dest_dir, '*'))
        queued_list.sort()
        assert extracted_filepaths == \
               list(filter(lambda fn: not fn.endswith(".target"), queued_list))
        assert queued_list == \
               ['output/dir/extract/000000000000.n01440764_2708.JPEG',
                'output/dir/extract/000000000000.n01440764_2708.JPEG.target',
                'output/dir/extract/000000000001.n01440764_7173.JPEG',
                'output/dir/extract/000000000001.n01440764_7173.JPEG.target',
                'output/dir/extract/000000000002.n01440764_6388.JPEG',
                'output/dir/extract/000000000002.n01440764_6388.JPEG.target',
                'output/dir/extract/000000000003.n01440764_3198.JPEG',
                'output/dir/extract/000000000003.n01440764_3198.JPEG.target',
                'output/dir/extract/000000000004.n01440764_3724.JPEG',
                'output/dir/extract/000000000004.n01440764_3724.JPEG.target',
                'output/dir/extract/000000000005.n01440764_11155.JPEG',
                'output/dir/extract/000000000005.n01440764_11155.JPEG.target',
                'output/dir/extract/000000000006.n01440764_7719.JPEG',
                'output/dir/extract/000000000006.n01440764_7719.JPEG.target',
                'output/dir/extract/000000000007.n01440764_7304.JPEG',
                'output/dir/extract/000000000007.n01440764_7304.JPEG.target',
                'output/dir/extract/000000000008.n01440764_8469.JPEG',
                'output/dir/extract/000000000008.n01440764_8469.JPEG.target',
                'output/dir/extract/000000000009.n01440764_6432.JPEG',
                'output/dir/extract/000000000009.n01440764_6432.JPEG.target',
                'output/dir/extract/000000000010.n01443537_2772.JPEG',
                'output/dir/extract/000000000010.n01443537_2772.JPEG.target',
                'output/dir/extract/000000000011.n01443537_1029.JPEG',
                'output/dir/extract/000000000011.n01443537_1029.JPEG.target',
                'output/dir/extract/000000000012.n01443537_1955.JPEG',
                'output/dir/extract/000000000012.n01443537_1955.JPEG.target',
                'output/dir/extract/000000000013.n01443537_962.JPEG',
                'output/dir/extract/000000000013.n01443537_962.JPEG.target',
                'output/dir/extract/000000000014.n01443537_2563.JPEG',
                'output/dir/extract/000000000014.n01443537_2563.JPEG.target',
                'output/dir/extract/000000000015.n01443537_3344.JPEG',
                'output/dir/extract/000000000015.n01443537_3344.JPEG.target',
                'output/dir/extract/000000000016.n01443537_3601.JPEG',
                'output/dir/extract/000000000016.n01443537_3601.JPEG.target',
                'output/dir/extract/000000000017.n01443537_2333.JPEG',
                'output/dir/extract/000000000017.n01443537_2333.JPEG.target',
                'output/dir/extract/000000000018.n01443537_801.JPEG',
                'output/dir/extract/000000000018.n01443537_801.JPEG.target',
                'output/dir/extract/000000000019.n01443537_2228.JPEG',
                'output/dir/extract/000000000019.n01443537_2228.JPEG.target',
                'output/dir/extract/000000000020.n01484850_4496.JPEG',
                'output/dir/extract/000000000020.n01484850_4496.JPEG.target',
                'output/dir/extract/000000000021.n01484850_2506.JPEG',
                'output/dir/extract/000000000021.n01484850_2506.JPEG.target',
                'output/dir/extract/000000000022.n01484850_17864.JPEG',
                'output/dir/extract/000000000022.n01484850_17864.JPEG.target',
                'output/dir/extract/000000000023.n01484850_4645.JPEG',
                'output/dir/extract/000000000023.n01484850_4645.JPEG.target',
                'output/dir/extract/000000000024.n01484850_22221.JPEG',
                'output/dir/extract/000000000024.n01484850_22221.JPEG.target',
                'output/dir/extract/000000000025.n01484850_2301.JPEG',
                'output/dir/extract/000000000025.n01484850_2301.JPEG.target',
                'output/dir/extract/000000000026.n01484850_2030.JPEG',
                'output/dir/extract/000000000026.n01484850_2030.JPEG.target',
                'output/dir/extract/000000000027.n01484850_3955.JPEG',
                'output/dir/extract/000000000027.n01484850_3955.JPEG.target',
                'output/dir/extract/000000000028.n01484850_7557.JPEG',
                'output/dir/extract/000000000028.n01484850_7557.JPEG.target',
                'output/dir/extract/000000000029.n01484850_8744.JPEG',
                'output/dir/extract/000000000029.n01484850_8744.JPEG.target',
                'output/dir/extract/000000000030.n01491361_8657.JPEG',
                'output/dir/extract/000000000030.n01491361_8657.JPEG.target',
                'output/dir/extract/000000000031.n01491361_1378.JPEG',
                'output/dir/extract/000000000031.n01491361_1378.JPEG.target',
                'output/dir/extract/000000000032.n01491361_733.JPEG',
                'output/dir/extract/000000000032.n01491361_733.JPEG.target',
                'output/dir/extract/000000000033.n01491361_11165.JPEG',
                'output/dir/extract/000000000033.n01491361_11165.JPEG.target',
                'output/dir/extract/000000000034.n01491361_2371.JPEG',
                'output/dir/extract/000000000034.n01491361_2371.JPEG.target',
                'output/dir/extract/000000000035.n01491361_3244.JPEG',
                'output/dir/extract/000000000035.n01491361_3244.JPEG.target',
                'output/dir/extract/000000000036.n01491361_392.JPEG',
                'output/dir/extract/000000000036.n01491361_392.JPEG.target',
                'output/dir/extract/000000000037.n01491361_5778.JPEG',
                'output/dir/extract/000000000037.n01491361_5778.JPEG.target',
                'output/dir/extract/000000000038.n01491361_10052.JPEG',
                'output/dir/extract/000000000038.n01491361_10052.JPEG.target',
                'output/dir/extract/000000000039.n01491361_8285.JPEG',
                'output/dir/extract/000000000039.n01491361_8285.JPEG.target',
                'output/dir/extract/000000000040.n01494475_5106.JPEG',
                'output/dir/extract/000000000040.n01494475_5106.JPEG.target',
                'output/dir/extract/000000000041.n01494475_6425.JPEG',
                'output/dir/extract/000000000041.n01494475_6425.JPEG.target',
                'output/dir/extract/000000000042.n01494475_4099.JPEG',
                'output/dir/extract/000000000042.n01494475_4099.JPEG.target',
                'output/dir/extract/000000000043.n01494475_3676.JPEG',
                'output/dir/extract/000000000043.n01494475_3676.JPEG.target',
                'output/dir/extract/000000000044.n01494475_3233.JPEG',
                'output/dir/extract/000000000044.n01494475_3233.JPEG.target',
                'output/dir/extract/000000000045.n01494475_1096.JPEG',
                'output/dir/extract/000000000045.n01494475_1096.JPEG.target',
                'output/dir/extract/000000000046.n01494475_7121.JPEG',
                'output/dir/extract/000000000046.n01494475_7121.JPEG.target',
                'output/dir/extract/000000000047.n01494475_1963.JPEG',
                'output/dir/extract/000000000047.n01494475_1963.JPEG.target',
                'output/dir/extract/000000000048.n01494475_6324.JPEG',
                'output/dir/extract/000000000048.n01494475_6324.JPEG.target',
                'output/dir/extract/000000000049.n01494475_16001.JPEG',
                'output/dir/extract/000000000049.n01494475_16001.JPEG.target',
                'output/dir/extract/000000000050.n01496331_1147.JPEG',
                'output/dir/extract/000000000050.n01496331_1147.JPEG.target',
                'output/dir/extract/000000000051.n01496331_8641.JPEG',
                'output/dir/extract/000000000051.n01496331_8641.JPEG.target',
                'output/dir/extract/000000000052.n01496331_7517.JPEG',
                'output/dir/extract/000000000052.n01496331_7517.JPEG.target',
                'output/dir/extract/000000000053.n01496331_11655.JPEG',
                'output/dir/extract/000000000053.n01496331_11655.JPEG.target',
                'output/dir/extract/000000000054.n01496331_11015.JPEG',
                'output/dir/extract/000000000054.n01496331_11015.JPEG.target',
                'output/dir/extract/000000000055.n01496331_13372.JPEG',
                'output/dir/extract/000000000055.n01496331_13372.JPEG.target',
                'output/dir/extract/000000000056.n01496331_1010.JPEG',
                'output/dir/extract/000000000056.n01496331_1010.JPEG.target',
                'output/dir/extract/000000000057.n01496331_11391.JPEG',
                'output/dir/extract/000000000057.n01496331_11391.JPEG.target',
                'output/dir/extract/000000000058.n01496331_3375.JPEG',
                'output/dir/extract/000000000058.n01496331_3375.JPEG.target',
                'output/dir/extract/000000000059.n01496331_11989.JPEG',
                'output/dir/extract/000000000059.n01496331_11989.JPEG.target',
                'output/dir/extract/000000000060.n01498041_928.JPEG',
                'output/dir/extract/000000000060.n01498041_928.JPEG.target',
                'output/dir/extract/000000000061.n01498041_1228.JPEG',
                'output/dir/extract/000000000061.n01498041_1228.JPEG.target',
                'output/dir/extract/000000000062.n01498041_11710.JPEG',
                'output/dir/extract/000000000062.n01498041_11710.JPEG.target',
                'output/dir/extract/000000000063.n01498041_10701.JPEG',
                'output/dir/extract/000000000063.n01498041_10701.JPEG.target',
                'output/dir/extract/000000000064.n01498041_11273.JPEG',
                'output/dir/extract/000000000064.n01498041_11273.JPEG.target',
                'output/dir/extract/000000000065.n01498041_11075.JPEG',
                'output/dir/extract/000000000065.n01498041_11075.JPEG.target',
                'output/dir/extract/000000000066.n01498041_3515.JPEG',
                'output/dir/extract/000000000066.n01498041_3515.JPEG.target',
                'output/dir/extract/000000000067.n01498041_13381.JPEG',
                'output/dir/extract/000000000067.n01498041_13381.JPEG.target',
                'output/dir/extract/000000000068.n01498041_11359.JPEG',
                'output/dir/extract/000000000068.n01498041_11359.JPEG.target',
                'output/dir/extract/000000000069.n01498041_11378.JPEG',
                'output/dir/extract/000000000069.n01498041_11378.JPEG.target',
                'output/dir/extract/000000000070.n01514668_17075.JPEG',
                'output/dir/extract/000000000070.n01514668_17075.JPEG.target',
                'output/dir/extract/000000000071.n01514668_14627.JPEG',
                'output/dir/extract/000000000071.n01514668_14627.JPEG.target',
                'output/dir/extract/000000000072.n01514668_13092.JPEG',
                'output/dir/extract/000000000072.n01514668_13092.JPEG.target',
                'output/dir/extract/000000000073.n01514668_19593.JPEG',
                'output/dir/extract/000000000073.n01514668_19593.JPEG.target',
                'output/dir/extract/000000000074.n01514668_10921.JPEG',
                'output/dir/extract/000000000074.n01514668_10921.JPEG.target',
                'output/dir/extract/000000000075.n01514668_13383.JPEG',
                'output/dir/extract/000000000075.n01514668_13383.JPEG.target',
                'output/dir/extract/000000000076.n01514668_19416.JPEG',
                'output/dir/extract/000000000076.n01514668_19416.JPEG.target',
                'output/dir/extract/000000000077.n01514668_16059.JPEG',
                'output/dir/extract/000000000077.n01514668_16059.JPEG.target',
                'output/dir/extract/000000000078.n01514668_13265.JPEG',
                'output/dir/extract/000000000078.n01514668_13265.JPEG.target',
                'output/dir/extract/000000000079.n01514668_10979.JPEG',
                'output/dir/extract/000000000079.n01514668_10979.JPEG.target',
                'output/dir/extract/000000000080.n01514859_12602.JPEG',
                'output/dir/extract/000000000080.n01514859_12602.JPEG.target',
                'output/dir/extract/000000000081.n01514859_12244.JPEG',
                'output/dir/extract/000000000081.n01514859_12244.JPEG.target',
                'output/dir/extract/000000000082.n01514859_11592.JPEG',
                'output/dir/extract/000000000082.n01514859_11592.JPEG.target',
                'output/dir/extract/000000000083.n01514859_11406.JPEG',
                'output/dir/extract/000000000083.n01514859_11406.JPEG.target',
                'output/dir/extract/000000000084.n01514859_3495.JPEG',
                'output/dir/extract/000000000084.n01514859_3495.JPEG.target',
                'output/dir/extract/000000000085.n01514859_11542.JPEG',
                'output/dir/extract/000000000085.n01514859_11542.JPEG.target',
                'output/dir/extract/000000000086.n01514859_6849.JPEG',
                'output/dir/extract/000000000086.n01514859_6849.JPEG.target',
                'output/dir/extract/000000000087.n01514859_12645.JPEG',
                'output/dir/extract/000000000087.n01514859_12645.JPEG.target',
                'output/dir/extract/000000000088.n01514859_5217.JPEG',
                'output/dir/extract/000000000088.n01514859_5217.JPEG.target',
                'output/dir/extract/000000000089.n01514859_2287.JPEG',
                'output/dir/extract/000000000089.n01514859_2287.JPEG.target',
                'output/dir/extract/000000000090.n01518878_832.JPEG',
                'output/dir/extract/000000000090.n01518878_832.JPEG.target',
                'output/dir/extract/000000000091.n01518878_587.JPEG',
                'output/dir/extract/000000000091.n01518878_587.JPEG.target',
                'output/dir/extract/000000000092.n01518878_3943.JPEG',
                'output/dir/extract/000000000092.n01518878_3943.JPEG.target',
                'output/dir/extract/000000000093.n01518878_12010.JPEG',
                'output/dir/extract/000000000093.n01518878_12010.JPEG.target',
                'output/dir/extract/000000000094.n01518878_2507.JPEG',
                'output/dir/extract/000000000094.n01518878_2507.JPEG.target',
                'output/dir/extract/000000000095.n01518878_10939.JPEG',
                'output/dir/extract/000000000095.n01518878_10939.JPEG.target',
                'output/dir/extract/000000000096.n01518878_681.JPEG',
                'output/dir/extract/000000000096.n01518878_681.JPEG.target',
                'output/dir/extract/000000000097.n01518878_3924.JPEG',
                'output/dir/extract/000000000097.n01518878_3924.JPEG.target',
                'output/dir/extract/000000000098.n01518878_5201.JPEG',
                'output/dir/extract/000000000098.n01518878_5201.JPEG.target',
                'output/dir/extract/000000000099.n01518878_10581.JPEG',
                'output/dir/extract/000000000099.n01518878_10581.JPEG.target',
                'output/dir/extract/000000000100.n01530575_10086.JPEG',
                'output/dir/extract/000000000100.n01530575_10086.JPEG.target',
                'output/dir/extract/000000000101.n01530575_1894.JPEG',
                'output/dir/extract/000000000101.n01530575_1894.JPEG.target',
                'output/dir/extract/000000000102.n01530575_10208.JPEG',
                'output/dir/extract/000000000102.n01530575_10208.JPEG.target',
                'output/dir/extract/000000000103.n01530575_10595.JPEG',
                'output/dir/extract/000000000103.n01530575_10595.JPEG.target',
                'output/dir/extract/000000000104.n01530575_78.JPEG',
                'output/dir/extract/000000000104.n01530575_78.JPEG.target',
                'output/dir/extract/000000000105.n01530575_10463.JPEG',
                'output/dir/extract/000000000105.n01530575_10463.JPEG.target',
                'output/dir/extract/000000000106.n01530575_10487.JPEG',
                'output/dir/extract/000000000106.n01530575_10487.JPEG.target',
                'output/dir/extract/000000000107.n01530575_10581.JPEG',
                'output/dir/extract/000000000107.n01530575_10581.JPEG.target',
                'output/dir/extract/000000000108.n01530575_10021.JPEG',
                'output/dir/extract/000000000108.n01530575_10021.JPEG.target',
                'output/dir/extract/000000000109.n01530575_9806.JPEG',
                'output/dir/extract/000000000109.n01530575_9806.JPEG.target',
                'output/dir/extract/000000000110.n01531178_3907.JPEG',
                'output/dir/extract/000000000110.n01531178_3907.JPEG.target',
                'output/dir/extract/000000000111.n01531178_21208.JPEG',
                'output/dir/extract/000000000111.n01531178_21208.JPEG.target',
                'output/dir/extract/000000000112.n01531178_18788.JPEG',
                'output/dir/extract/000000000112.n01531178_18788.JPEG.target',
                'output/dir/extract/000000000113.n01531178_17669.JPEG',
                'output/dir/extract/000000000113.n01531178_17669.JPEG.target',
                'output/dir/extract/000000000114.n01531178_20318.JPEG',
                'output/dir/extract/000000000114.n01531178_20318.JPEG.target',
                'output/dir/extract/000000000115.n01531178_15737.JPEG',
                'output/dir/extract/000000000115.n01531178_15737.JPEG.target',
                'output/dir/extract/000000000116.n01531178_5049.JPEG',
                'output/dir/extract/000000000116.n01531178_5049.JPEG.target',
                'output/dir/extract/000000000117.n01531178_1996.JPEG',
                'output/dir/extract/000000000117.n01531178_1996.JPEG.target',
                'output/dir/extract/000000000118.n01531178_5894.JPEG',
                'output/dir/extract/000000000118.n01531178_5894.JPEG.target',
                'output/dir/extract/000000000119.n01531178_16393.JPEG',
                'output/dir/extract/000000000119.n01531178_16393.JPEG.target']

    finally:
        shutil.rmtree(".", ignore_errors=True)


@task_reset
def test_extract_indices():
    """Test that the correct number of files are extracted"""
    src = os.path.join(DATA_DIR, "dev_im_net/dev_im_net.tar")
    dest = "output/dir/extract/"
    dest_dir = os.path.dirname(dest)

    args, _ = parse_args(["extract", src, dest, "imagenet:tar",
                          "--indices", ",".join([str(v)
                                                 for v in range(10, 25)])])
    del args._action

    try:
        extracted_filepaths = _run_tasks(extract(**vars(args)))[0]

        queued_list = glob.glob(os.path.join(dest_dir, '*'))
        queued_list.sort()
        assert extracted_filepaths == \
               list(filter(lambda fn: not fn.endswith(".target"), queued_list))
        assert queued_list == \
               ['output/dir/extract/000000000010.n01443537_2772.JPEG',
                'output/dir/extract/000000000010.n01443537_2772.JPEG.target',
                'output/dir/extract/000000000011.n01443537_1029.JPEG',
                'output/dir/extract/000000000011.n01443537_1029.JPEG.target',
                'output/dir/extract/000000000012.n01443537_1955.JPEG',
                'output/dir/extract/000000000012.n01443537_1955.JPEG.target',
                'output/dir/extract/000000000013.n01443537_962.JPEG',
                'output/dir/extract/000000000013.n01443537_962.JPEG.target',
                'output/dir/extract/000000000014.n01443537_2563.JPEG',
                'output/dir/extract/000000000014.n01443537_2563.JPEG.target',
                'output/dir/extract/000000000015.n01443537_3344.JPEG',
                'output/dir/extract/000000000015.n01443537_3344.JPEG.target',
                'output/dir/extract/000000000016.n01443537_3601.JPEG',
                'output/dir/extract/000000000016.n01443537_3601.JPEG.target',
                'output/dir/extract/000000000017.n01443537_2333.JPEG',
                'output/dir/extract/000000000017.n01443537_2333.JPEG.target',
                'output/dir/extract/000000000018.n01443537_801.JPEG',
                'output/dir/extract/000000000018.n01443537_801.JPEG.target',
                'output/dir/extract/000000000019.n01443537_2228.JPEG',
                'output/dir/extract/000000000019.n01443537_2228.JPEG.target',
                'output/dir/extract/000000000020.n01484850_4496.JPEG',
                'output/dir/extract/000000000020.n01484850_4496.JPEG.target',
                'output/dir/extract/000000000021.n01484850_2506.JPEG',
                'output/dir/extract/000000000021.n01484850_2506.JPEG.target',
                'output/dir/extract/000000000022.n01484850_17864.JPEG',
                'output/dir/extract/000000000022.n01484850_17864.JPEG.target',
                'output/dir/extract/000000000023.n01484850_4645.JPEG',
                'output/dir/extract/000000000023.n01484850_4645.JPEG.target',
                'output/dir/extract/000000000024.n01484850_22221.JPEG',
                'output/dir/extract/000000000024.n01484850_22221.JPEG.target']
        args.size = 5
        assert _run_tasks(extract(**vars(args)))[0] == extracted_filepaths

    finally:
        shutil.rmtree(".", ignore_errors=True)


@task_reset
def test_extract_int_indices_size():
    """Test that the correct number of files are extracted"""
    src = os.path.join(DATA_DIR, "dev_im_net/dev_im_net.tar")
    dest = "output/dir/extract/"
    dest_dir = os.path.dirname(dest)

    args, _ = parse_args(["extract", src, dest, "imagenet:tar",
                          "--indices", "10", "--size", "15"])
    del args._action

    try:
        extracted_filepaths = _run_tasks(extract(**vars(args)))[0]

        queued_list = glob.glob(os.path.join(dest_dir, '*'))
        queued_list.sort()
        assert extracted_filepaths == \
               list(filter(lambda fn: not fn.endswith(".target"), queued_list))
        assert queued_list == \
               ['output/dir/extract/000000000010.n01443537_2772.JPEG',
                'output/dir/extract/000000000010.n01443537_2772.JPEG.target',
                'output/dir/extract/000000000011.n01443537_1029.JPEG',
                'output/dir/extract/000000000011.n01443537_1029.JPEG.target',
                'output/dir/extract/000000000012.n01443537_1955.JPEG',
                'output/dir/extract/000000000012.n01443537_1955.JPEG.target',
                'output/dir/extract/000000000013.n01443537_962.JPEG',
                'output/dir/extract/000000000013.n01443537_962.JPEG.target',
                'output/dir/extract/000000000014.n01443537_2563.JPEG',
                'output/dir/extract/000000000014.n01443537_2563.JPEG.target',
                'output/dir/extract/000000000015.n01443537_3344.JPEG',
                'output/dir/extract/000000000015.n01443537_3344.JPEG.target',
                'output/dir/extract/000000000016.n01443537_3601.JPEG',
                'output/dir/extract/000000000016.n01443537_3601.JPEG.target',
                'output/dir/extract/000000000017.n01443537_2333.JPEG',
                'output/dir/extract/000000000017.n01443537_2333.JPEG.target',
                'output/dir/extract/000000000018.n01443537_801.JPEG',
                'output/dir/extract/000000000018.n01443537_801.JPEG.target',
                'output/dir/extract/000000000019.n01443537_2228.JPEG',
                'output/dir/extract/000000000019.n01443537_2228.JPEG.target',
                'output/dir/extract/000000000020.n01484850_4496.JPEG',
                'output/dir/extract/000000000020.n01484850_4496.JPEG.target',
                'output/dir/extract/000000000021.n01484850_2506.JPEG',
                'output/dir/extract/000000000021.n01484850_2506.JPEG.target',
                'output/dir/extract/000000000022.n01484850_17864.JPEG',
                'output/dir/extract/000000000022.n01484850_17864.JPEG.target',
                'output/dir/extract/000000000023.n01484850_4645.JPEG',
                'output/dir/extract/000000000023.n01484850_4645.JPEG.target',
                'output/dir/extract/000000000024.n01484850_22221.JPEG',
                'output/dir/extract/000000000024.n01484850_22221.JPEG.target']

    finally:
        shutil.rmtree(".", ignore_errors=True)


@task_reset
def test_extract_int_indices_batch_size():
    """Test that the correct number of files are extracted"""
    src = os.path.join(DATA_DIR, "dev_im_net/dev_im_net.tar")
    dest = "output/dir/extract/"
    dest_dir = os.path.dirname(dest)

    args, _ = parse_args(["extract", src, dest, "imagenet:tar",
                          "--indices", "10", "--size", "15",
                          "--batch-size", "5"])
    del args._action

    try:
        extracted_filepaths = _run_tasks(extract(**vars(args)))
        assert len(extracted_filepaths) == 3

        for l in extracted_filepaths[1:]:
            assert len(l) == 5
            extracted_filepaths[0].extend(l)

        extracted_filepaths = extracted_filepaths[0]

        queued_list = glob.glob(os.path.join(dest_dir, '*'))
        queued_list.sort()
        assert extracted_filepaths == \
               list(filter(lambda fn: not fn.endswith(".target"), queued_list))
        assert queued_list == \
               ['output/dir/extract/000000000010.n01443537_2772.JPEG',
                'output/dir/extract/000000000010.n01443537_2772.JPEG.target',
                'output/dir/extract/000000000011.n01443537_1029.JPEG',
                'output/dir/extract/000000000011.n01443537_1029.JPEG.target',
                'output/dir/extract/000000000012.n01443537_1955.JPEG',
                'output/dir/extract/000000000012.n01443537_1955.JPEG.target',
                'output/dir/extract/000000000013.n01443537_962.JPEG',
                'output/dir/extract/000000000013.n01443537_962.JPEG.target',
                'output/dir/extract/000000000014.n01443537_2563.JPEG',
                'output/dir/extract/000000000014.n01443537_2563.JPEG.target',
                'output/dir/extract/000000000015.n01443537_3344.JPEG',
                'output/dir/extract/000000000015.n01443537_3344.JPEG.target',
                'output/dir/extract/000000000016.n01443537_3601.JPEG',
                'output/dir/extract/000000000016.n01443537_3601.JPEG.target',
                'output/dir/extract/000000000017.n01443537_2333.JPEG',
                'output/dir/extract/000000000017.n01443537_2333.JPEG.target',
                'output/dir/extract/000000000018.n01443537_801.JPEG',
                'output/dir/extract/000000000018.n01443537_801.JPEG.target',
                'output/dir/extract/000000000019.n01443537_2228.JPEG',
                'output/dir/extract/000000000019.n01443537_2228.JPEG.target',
                'output/dir/extract/000000000020.n01484850_4496.JPEG',
                'output/dir/extract/000000000020.n01484850_4496.JPEG.target',
                'output/dir/extract/000000000021.n01484850_2506.JPEG',
                'output/dir/extract/000000000021.n01484850_2506.JPEG.target',
                'output/dir/extract/000000000022.n01484850_17864.JPEG',
                'output/dir/extract/000000000022.n01484850_17864.JPEG.target',
                'output/dir/extract/000000000023.n01484850_4645.JPEG',
                'output/dir/extract/000000000023.n01484850_4645.JPEG.target',
                'output/dir/extract/000000000024.n01484850_22221.JPEG',
                'output/dir/extract/000000000024.n01484850_22221.JPEG.target']

    finally:
        shutil.rmtree(".", ignore_errors=True)


@task_reset
def test_pybenzinaconcat_extract_chain_transcode():
    """Test that files are extracted then transcoded using the main
    entry point
    """
    src = os.path.join(DATA_DIR, "dev_im_net/dev_im_net.tar")
    dest = "output/dir/extract/"
    dest_dir = os.path.dirname(dest)

    transcode_dest = "output/dir/"
    upload_dir = os.path.join(transcode_dest, "upload")
    queue_dir = os.path.join(transcode_dest, "queue")
    transcode_tmp = "tmp/"

    args, argv = parse_args(["extract", src, dest, "imagenet:tar",
                             "--indices", "10", "--size", "15",
                             "--transcode", transcode_dest,
                             "--tmp", transcode_tmp])

    try:
        if upload_dir and not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        if queue_dir and not os.path.exists(queue_dir):
            os.makedirs(queue_dir)

        transcode_filepaths = _run_tasks(main(args, argv))[0]

        queued_list = glob.glob(os.path.join(queue_dir, '*'))
        queued_list.sort()
        assert transcode_filepaths == \
               list(filter(lambda fn: not fn.endswith(".target"), queued_list))
        assert queued_list == \
               ['output/dir/queue/000000000010.n01443537_2772.JPEG.transcoded',
                'output/dir/queue/000000000011.n01443537_1029.JPEG.transcoded',
                'output/dir/queue/000000000012.n01443537_1955.JPEG.transcoded',
                'output/dir/queue/000000000013.n01443537_962.JPEG.transcoded',
                'output/dir/queue/000000000014.n01443537_2563.JPEG.transcoded',
                'output/dir/queue/000000000015.n01443537_3344.JPEG.transcoded',
                'output/dir/queue/000000000016.n01443537_3601.JPEG.transcoded',
                'output/dir/queue/000000000017.n01443537_2333.JPEG.transcoded',
                'output/dir/queue/000000000018.n01443537_801.JPEG.transcoded',
                'output/dir/queue/000000000019.n01443537_2228.JPEG.transcoded',
                'output/dir/queue/000000000020.n01484850_4496.JPEG.transcoded',
                'output/dir/queue/000000000021.n01484850_2506.JPEG.transcoded',
                'output/dir/queue/000000000022.n01484850_17864.JPEG.transcoded',
                'output/dir/queue/000000000023.n01484850_4645.JPEG.transcoded',
                'output/dir/queue/000000000024.n01484850_22221.JPEG.transcoded']

        assert len(glob.glob(os.path.join(dest_dir, '*'))) == 30
        assert len(glob.glob(os.path.join(transcode_tmp, '*'))) == 0
        assert len(glob.glob(os.path.join(upload_dir, '*'))) == 0

    finally:
        shutil.rmtree(".", ignore_errors=True)


@task_reset
def test_pybenzinaconcat_extract_chain_transcode_mp4():
    """Test that files are extracted then transcoded to mop4 using the
    main entry point
    """
    src = os.path.join(DATA_DIR, "dev_im_net/dev_im_net.tar")
    dest = "output/dir/extract/"
    dest_dir = os.path.dirname(dest)

    transcode_dest = "output/dir/"
    upload_dir = os.path.join(transcode_dest, "upload")
    queue_dir = os.path.join(transcode_dest, "queue")
    transcode_tmp = "tmp/"

    args, argv = parse_args(["extract", src, dest, "imagenet:tar",
                             "--indices", "10", "--size", "15",
                             "--transcode", transcode_dest, "--mp4",
                             "--tmp", transcode_tmp])

    try:
        if upload_dir and not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        if queue_dir and not os.path.exists(queue_dir):
            os.makedirs(queue_dir)

        transcode_filepaths = _run_tasks(main(args, argv))[0]

        extract_list = glob.glob(os.path.join(dest_dir, '*'))
        extract_list.sort()
        assert extract_list == \
               ['output/dir/extract/000000000010.n01443537_2772.JPEG',
                'output/dir/extract/000000000010.n01443537_2772.JPEG.target',
                'output/dir/extract/000000000011.n01443537_1029.JPEG',
                'output/dir/extract/000000000011.n01443537_1029.JPEG.target',
                'output/dir/extract/000000000012.n01443537_1955.JPEG',
                'output/dir/extract/000000000012.n01443537_1955.JPEG.target',
                'output/dir/extract/000000000013.n01443537_962.JPEG',
                'output/dir/extract/000000000013.n01443537_962.JPEG.target',
                'output/dir/extract/000000000014.n01443537_2563.JPEG',
                'output/dir/extract/000000000014.n01443537_2563.JPEG.target',
                'output/dir/extract/000000000015.n01443537_3344.JPEG',
                'output/dir/extract/000000000015.n01443537_3344.JPEG.target',
                'output/dir/extract/000000000016.n01443537_3601.JPEG',
                'output/dir/extract/000000000016.n01443537_3601.JPEG.target',
                'output/dir/extract/000000000017.n01443537_2333.JPEG',
                'output/dir/extract/000000000017.n01443537_2333.JPEG.target',
                'output/dir/extract/000000000018.n01443537_801.JPEG',
                'output/dir/extract/000000000018.n01443537_801.JPEG.target',
                'output/dir/extract/000000000019.n01443537_2228.JPEG',
                'output/dir/extract/000000000019.n01443537_2228.JPEG.target',
                'output/dir/extract/000000000020.n01484850_4496.JPEG',
                'output/dir/extract/000000000020.n01484850_4496.JPEG.target',
                'output/dir/extract/000000000021.n01484850_2506.JPEG',
                'output/dir/extract/000000000021.n01484850_2506.JPEG.target',
                'output/dir/extract/000000000022.n01484850_17864.JPEG',
                'output/dir/extract/000000000022.n01484850_17864.JPEG.target',
                'output/dir/extract/000000000023.n01484850_4645.JPEG',
                'output/dir/extract/000000000023.n01484850_4645.JPEG.target',
                'output/dir/extract/000000000024.n01484850_22221.JPEG',
                'output/dir/extract/000000000024.n01484850_22221.JPEG.target']

        queued_list = glob.glob(os.path.join(queue_dir, '*'))
        queued_list.sort()
        assert transcode_filepaths == \
               list(filter(lambda fn: not fn.endswith(".target"), queued_list))
        assert queued_list == \
               ['output/dir/queue/000000000010.n01443537_2772.JPEG.transcoded',
                'output/dir/queue/000000000011.n01443537_1029.JPEG.transcoded',
                'output/dir/queue/000000000012.n01443537_1955.JPEG.transcoded',
                'output/dir/queue/000000000013.n01443537_962.JPEG.transcoded',
                'output/dir/queue/000000000014.n01443537_2563.JPEG.transcoded',
                'output/dir/queue/000000000015.n01443537_3344.JPEG.transcoded',
                'output/dir/queue/000000000016.n01443537_3601.JPEG.transcoded',
                'output/dir/queue/000000000017.n01443537_2333.JPEG.transcoded',
                'output/dir/queue/000000000018.n01443537_801.JPEG.transcoded',
                'output/dir/queue/000000000019.n01443537_2228.JPEG.transcoded',
                'output/dir/queue/000000000020.n01484850_4496.JPEG.transcoded',
                'output/dir/queue/000000000021.n01484850_2506.JPEG.transcoded',
                'output/dir/queue/000000000022.n01484850_17864.JPEG.transcoded',
                'output/dir/queue/000000000023.n01484850_4645.JPEG.transcoded',
                'output/dir/queue/000000000024.n01484850_22221.JPEG.transcoded']

        assert len(glob.glob(os.path.join(transcode_tmp, '*'))) == 0
        assert len(glob.glob(os.path.join(upload_dir, '*'))) == 0

    finally:
        shutil.rmtree(".", ignore_errors=True)


def test_python_extract_batch_size_chain_transcode_mp4():
    """Test that files are extracted then transcoded to mop4 using python"""
    src = os.path.join(DATA_DIR, "dev_im_net/dev_im_net.tar")
    dest = "output/dir/extract/"
    dest_dir = os.path.dirname(dest)

    transcode_dest = "output/dir/"
    upload_dir = os.path.join(transcode_dest, "upload")
    queue_dir = os.path.join(transcode_dest, "queue")
    transcode_tmp = "tmp/"

    args = ["--", "extract", src, dest, "imagenet:tar", "--indices", "10",
            "--size", "15",
            "--transcode", transcode_dest, "--mp4", "--tmp", transcode_tmp]

    try:
        if upload_dir and not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        if queue_dir and not os.path.exists(queue_dir):
            os.makedirs(queue_dir)

        processes = [subprocess.Popen(["python3", "../../pybenzinaconcat"] +
                                      args) for _ in range(4)]

        for p in processes:
            p.wait()
            assert p.returncode == 0

        extract_list = glob.glob(os.path.join(dest_dir, '*'))
        extract_list.sort()
        assert extract_list == \
               ['output/dir/extract/000000000010.n01443537_2772.JPEG',
                'output/dir/extract/000000000010.n01443537_2772.JPEG.target',
                'output/dir/extract/000000000011.n01443537_1029.JPEG',
                'output/dir/extract/000000000011.n01443537_1029.JPEG.target',
                'output/dir/extract/000000000012.n01443537_1955.JPEG',
                'output/dir/extract/000000000012.n01443537_1955.JPEG.target',
                'output/dir/extract/000000000013.n01443537_962.JPEG',
                'output/dir/extract/000000000013.n01443537_962.JPEG.target',
                'output/dir/extract/000000000014.n01443537_2563.JPEG',
                'output/dir/extract/000000000014.n01443537_2563.JPEG.target',
                'output/dir/extract/000000000015.n01443537_3344.JPEG',
                'output/dir/extract/000000000015.n01443537_3344.JPEG.target',
                'output/dir/extract/000000000016.n01443537_3601.JPEG',
                'output/dir/extract/000000000016.n01443537_3601.JPEG.target',
                'output/dir/extract/000000000017.n01443537_2333.JPEG',
                'output/dir/extract/000000000017.n01443537_2333.JPEG.target',
                'output/dir/extract/000000000018.n01443537_801.JPEG',
                'output/dir/extract/000000000018.n01443537_801.JPEG.target',
                'output/dir/extract/000000000019.n01443537_2228.JPEG',
                'output/dir/extract/000000000019.n01443537_2228.JPEG.target',
                'output/dir/extract/000000000020.n01484850_4496.JPEG',
                'output/dir/extract/000000000020.n01484850_4496.JPEG.target',
                'output/dir/extract/000000000021.n01484850_2506.JPEG',
                'output/dir/extract/000000000021.n01484850_2506.JPEG.target',
                'output/dir/extract/000000000022.n01484850_17864.JPEG',
                'output/dir/extract/000000000022.n01484850_17864.JPEG.target',
                'output/dir/extract/000000000023.n01484850_4645.JPEG',
                'output/dir/extract/000000000023.n01484850_4645.JPEG.target',
                'output/dir/extract/000000000024.n01484850_22221.JPEG',
                'output/dir/extract/000000000024.n01484850_22221.JPEG.target']

        queued_list = glob.glob(os.path.join(queue_dir, '*'))
        queued_list.sort()
        assert queued_list == \
               ['output/dir/queue/000000000010.n01443537_2772.JPEG.transcoded',
                'output/dir/queue/000000000011.n01443537_1029.JPEG.transcoded',
                'output/dir/queue/000000000012.n01443537_1955.JPEG.transcoded',
                'output/dir/queue/000000000013.n01443537_962.JPEG.transcoded',
                'output/dir/queue/000000000014.n01443537_2563.JPEG.transcoded',
                'output/dir/queue/000000000015.n01443537_3344.JPEG.transcoded',
                'output/dir/queue/000000000016.n01443537_3601.JPEG.transcoded',
                'output/dir/queue/000000000017.n01443537_2333.JPEG.transcoded',
                'output/dir/queue/000000000018.n01443537_801.JPEG.transcoded',
                'output/dir/queue/000000000019.n01443537_2228.JPEG.transcoded',
                'output/dir/queue/000000000020.n01484850_4496.JPEG.transcoded',
                'output/dir/queue/000000000021.n01484850_2506.JPEG.transcoded',
                'output/dir/queue/000000000022.n01484850_17864.JPEG.transcoded',
                'output/dir/queue/000000000023.n01484850_4645.JPEG.transcoded',
                'output/dir/queue/000000000024.n01484850_22221.JPEG.transcoded']

        assert len(glob.glob(os.path.join(transcode_tmp, '*'))) == 0
        assert len(glob.glob(os.path.join(upload_dir, '*'))) == 0

    finally:
        shutil.rmtree(".", ignore_errors=True)


def test_python_extract_batch_size_chain_transcode_mp4_force_bmp():
    """Test that files are extracted then transcoded to mop4 using python"""
    src = os.path.join(DATA_DIR, "dev_im_net/dev_im_net.tar")
    dest = "output/dir/extract/"

    transcode_dest = "output/dir/"
    upload_dir = os.path.join(transcode_dest, "upload")
    queue_dir = os.path.join(transcode_dest, "queue")
    transcode_tmp = "tmp/"

    args = ["--", "extract", src, dest, "imagenet:tar", "--indices", "10",
            "--size", "15",
            "--transcode", transcode_dest, "--mp4", "--force-bmp",
            "--tmp", transcode_tmp]

    try:
        if upload_dir and not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        if queue_dir and not os.path.exists(queue_dir):
            os.makedirs(queue_dir)
        if transcode_tmp and not os.path.exists(transcode_tmp):
            os.makedirs(transcode_tmp)

        processes = [subprocess.Popen(["python3", "../../pybenzinaconcat"] +
                                      args) for _ in range(4)]

        for p in processes:
            p.wait()
            assert p.returncode == 0

        queued_list = glob.glob(os.path.join(queue_dir, '*'))
        queued_list.sort()
        assert queued_list == \
               ['output/dir/queue/000000000010.n01443537_2772.BMP.transcoded',
                'output/dir/queue/000000000011.n01443537_1029.BMP.transcoded',
                'output/dir/queue/000000000012.n01443537_1955.BMP.transcoded',
                'output/dir/queue/000000000013.n01443537_962.BMP.transcoded',
                'output/dir/queue/000000000014.n01443537_2563.BMP.transcoded',
                'output/dir/queue/000000000015.n01443537_3344.BMP.transcoded',
                'output/dir/queue/000000000016.n01443537_3601.BMP.transcoded',
                'output/dir/queue/000000000017.n01443537_2333.BMP.transcoded',
                'output/dir/queue/000000000018.n01443537_801.BMP.transcoded',
                'output/dir/queue/000000000019.n01443537_2228.BMP.transcoded',
                'output/dir/queue/000000000020.n01484850_4496.BMP.transcoded',
                'output/dir/queue/000000000021.n01484850_2506.BMP.transcoded',
                'output/dir/queue/000000000022.n01484850_17864.BMP.transcoded',
                'output/dir/queue/000000000023.n01484850_4645.BMP.transcoded',
                'output/dir/queue/000000000024.n01484850_22221.BMP.transcoded']

        bmp_list = glob.glob(os.path.join(transcode_tmp, '*'))
        bmp_list.sort()
        assert bmp_list == \
               ['tmp/000000000010.n01443537_2772.BMP',
                'tmp/000000000010.n01443537_2772.BMP.target',
                'tmp/000000000011.n01443537_1029.BMP',
                'tmp/000000000011.n01443537_1029.BMP.target',
                'tmp/000000000012.n01443537_1955.BMP',
                'tmp/000000000012.n01443537_1955.BMP.target',
                'tmp/000000000013.n01443537_962.BMP',
                'tmp/000000000013.n01443537_962.BMP.target',
                'tmp/000000000014.n01443537_2563.BMP',
                'tmp/000000000014.n01443537_2563.BMP.target',
                'tmp/000000000015.n01443537_3344.BMP',
                'tmp/000000000015.n01443537_3344.BMP.target',
                'tmp/000000000016.n01443537_3601.BMP',
                'tmp/000000000016.n01443537_3601.BMP.target',
                'tmp/000000000017.n01443537_2333.BMP',
                'tmp/000000000017.n01443537_2333.BMP.target',
                'tmp/000000000018.n01443537_801.BMP',
                'tmp/000000000018.n01443537_801.BMP.target',
                'tmp/000000000019.n01443537_2228.BMP',
                'tmp/000000000019.n01443537_2228.BMP.target',
                'tmp/000000000020.n01484850_4496.BMP',
                'tmp/000000000020.n01484850_4496.BMP.target',
                'tmp/000000000021.n01484850_2506.BMP',
                'tmp/000000000021.n01484850_2506.BMP.target',
                'tmp/000000000022.n01484850_17864.BMP',
                'tmp/000000000022.n01484850_17864.BMP.target',
                'tmp/000000000023.n01484850_4645.BMP',
                'tmp/000000000023.n01484850_4645.BMP.target',
                'tmp/000000000024.n01484850_22221.BMP',
                'tmp/000000000024.n01484850_22221.BMP.target']

        assert len(glob.glob(os.path.join(upload_dir, '*'))) == 0

    finally:
        shutil.rmtree(".", ignore_errors=True)


def test_python_extract_batch_size_chain_transcode_chain_concat():
    """Test that files are extracted then transcoded and finally concatenated
    sequentially using python"""
    src = os.path.join(DATA_DIR, "dev_im_net/dev_im_net.tar")
    dest = "output/dir/extract/"
    dest_dir = os.path.dirname(dest)

    transcode_dest = "output/dir/"
    upload_dir = os.path.join(transcode_dest, "upload")
    queue_dir = os.path.join(transcode_dest, "queue")
    transcode_tmp = "tmp/"

    concat_file = "output/dir/concat.bzna"

    args = ["--", "extract", src, dest, "imagenet:tar", "--indices", "10",
            "--size", "15", "--batch-size", "5",
            "--transcode", transcode_dest, "--tmp", transcode_tmp,
            "--concat", concat_file]

    try:
        if upload_dir and not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        if queue_dir and not os.path.exists(queue_dir):
            os.makedirs(queue_dir)

        processes = [subprocess.Popen(["python3", "../../pybenzinaconcat"] +
                                      args) for _ in range(6)]

        for p in processes:
            p.wait(30)
            assert p.returncode == 0

        extract_list = glob.glob(os.path.join(dest_dir, '*'))
        extract_list.sort()
        assert extract_list == \
               ['output/dir/extract/000000000010.n01443537_2772.JPEG',
                'output/dir/extract/000000000010.n01443537_2772.JPEG.target',
                'output/dir/extract/000000000011.n01443537_1029.JPEG',
                'output/dir/extract/000000000011.n01443537_1029.JPEG.target',
                'output/dir/extract/000000000012.n01443537_1955.JPEG',
                'output/dir/extract/000000000012.n01443537_1955.JPEG.target',
                'output/dir/extract/000000000013.n01443537_962.JPEG',
                'output/dir/extract/000000000013.n01443537_962.JPEG.target',
                'output/dir/extract/000000000014.n01443537_2563.JPEG',
                'output/dir/extract/000000000014.n01443537_2563.JPEG.target',
                'output/dir/extract/000000000015.n01443537_3344.JPEG',
                'output/dir/extract/000000000015.n01443537_3344.JPEG.target',
                'output/dir/extract/000000000016.n01443537_3601.JPEG',
                'output/dir/extract/000000000016.n01443537_3601.JPEG.target',
                'output/dir/extract/000000000017.n01443537_2333.JPEG',
                'output/dir/extract/000000000017.n01443537_2333.JPEG.target',
                'output/dir/extract/000000000018.n01443537_801.JPEG',
                'output/dir/extract/000000000018.n01443537_801.JPEG.target',
                'output/dir/extract/000000000019.n01443537_2228.JPEG',
                'output/dir/extract/000000000019.n01443537_2228.JPEG.target',
                'output/dir/extract/000000000020.n01484850_4496.JPEG',
                'output/dir/extract/000000000020.n01484850_4496.JPEG.target',
                'output/dir/extract/000000000021.n01484850_2506.JPEG',
                'output/dir/extract/000000000021.n01484850_2506.JPEG.target',
                'output/dir/extract/000000000022.n01484850_17864.JPEG',
                'output/dir/extract/000000000022.n01484850_17864.JPEG.target',
                'output/dir/extract/000000000023.n01484850_4645.JPEG',
                'output/dir/extract/000000000023.n01484850_4645.JPEG.target',
                'output/dir/extract/000000000024.n01484850_22221.JPEG',
                'output/dir/extract/000000000024.n01484850_22221.JPEG.target']
        files_bytes = []
        for fn in extract_list:
            with open(fn, "rb") as f:
                files_bytes.append(f.read())

        with open(concat_file, "rb") as f:
            assert f.read() == b''.join(files_bytes)

        assert len(glob.glob(os.path.join(transcode_tmp, '*'))) == 0
        assert len(glob.glob(os.path.join(queue_dir, '*'))) == 0
        assert len(glob.glob(os.path.join(upload_dir, '*'))) == 0

    finally:
        shutil.rmtree(".", ignore_errors=True)


def test__is_transcoded():
    filename = FILENAME_TEMPLATE.format(filename="some_filename.extension",
                                        index=12)
    splitted_filename = filename.split('.')

    assert len(splitted_filename) == 4
    assert _is_transcoded(filename)
    assert not _is_transcoded('.'.join(splitted_filename[:-1]))


def test__get_clean_filepath():
    filename = FILENAME_TEMPLATE.format(filename="some_filename.extension",
                                        index=12)
    splitted_filename = filename.split('.')

    assert get_clean_filepath(filename) == "some_filename.extension"
    assert get_clean_filepath('.'.join(splitted_filename[:-1])) == \
           "some_filename.extension"
    assert get_clean_filepath('.'.join(splitted_filename[1:])) == \
           "some_filename.extension"
    assert get_clean_filepath('.'.join(splitted_filename[1:-1])) == \
           "some_filename.extension"
    assert get_clean_filepath("dir/dir/" + filename, basename=True) == \
           "some_filename.extension"
    assert get_clean_filepath("dir/dir/" + filename) == \
           "dir/dir/some_filename.extension"


def test__get_file_index():
    filename = FILENAME_TEMPLATE.format(filename="some_filename.extension",
                                        index=12)
    splitted_filename = filename.split('.')

    assert _get_file_index(filename) == 12
    assert _get_file_index('.'.join(splitted_filename[:-1])) == 12
    assert _get_file_index('.'.join(splitted_filename[1:])) is None
    assert _get_file_index("dir/dir/" + filename) == 12


def test__make_index_filepath():
    assert _make_index_filepath("some_filename.extension", 12) == \
           "000000000012.some_filename.extension"
    assert _make_index_filepath("dir/dir/some_filename.extension", 12) == \
           "dir/dir/000000000012.some_filename.extension"


def test__make_transcoded_filepath():
    assert _make_transcoded_filepath("some_filename.extension") == \
           "some_filename.extension.transcoded"
    assert _make_transcoded_filepath("dir/dir/some_filename.extension") == \
           "dir/dir/some_filename.extension.transcoded"


def test_action_redirection():
    raw_concat_argv = ["concat", "src_concat", "dest_concat"]

    raw_transcode_argv = ["transcode",
                          "src_transcode",
                          "dest_transcode",
                          "--ssh-remote", "remote_transcode",
                          "--tmp", "tmp_transcode"]

    raw_transcode_w_extra_argv = ["transcode",
                                  "src_transcode",
                                  "dest_transcode",
                                  "--ssh-remote", "remote_transcode",
                                  "--tmp", "tmp_transcode",
                                  "--concat",
                                  "src_concat",
                                  "dest_concat"]

    raw_extract_hdf5_argv = ["extract",
                             "src_extract_hdf5",
                             "dest_extract_hdf5",
                             "imagenet:hdf5",
                             "--indices", "10",
                             "--size", "15",
                             "--batch-size", "5"]

    raw_extract_tar_argv = ["extract",
                            "src_extract_tar",
                            "dest_extract_tar",
                            "imagenet:tar",
                            "--indices", "10",
                            "--size", "15",
                            "--batch-size", "5"]

    raw_extract_tinyimzip_argv = ["extract",
                                  "src_extract_zip",
                                  "dest_extract_zip",
                                  "tinyimagenet:zip",
                                  "--indices", "10",
                                  "--size", "15",
                                  "--batch-size", "5"]

    raw_extract_tar_w_extra_argv = ["extract",
                                    "src_extract_tar",
                                    "dest_extract_tar",
                                    "imagenet:tar",
                                    "--indices", "10",
                                    "--size", "15",
                                    "--batch-size", "5",
                                    "--transcode",
                                    "src_transcode",
                                    "dest_transcode",
                                    "--ssh-remote", "remote_transcode",
                                    "--tmp", "tmp_transcode"]

    args, argv = parse_args(raw_concat_argv)

    assert args._action == "concat"
    assert args.src == "src_concat"
    assert args.dest == "dest_concat"
    assert len(argv) == 0

    args, argv = parse_args(raw_transcode_argv)

    assert args._action == "transcode"
    assert args.src == "src_transcode"
    assert args.dest == "dest_transcode"
    assert args.ssh_remote == "remote_transcode"
    assert args.tmp == "tmp_transcode"
    assert len(argv) == 0

    args, argv = parse_args(raw_transcode_w_extra_argv)
    concat_args, argv = parse_args(argv)

    assert args._action == "transcode"
    assert args.src == "src_transcode"
    assert args.dest == "dest_transcode"
    assert args.ssh_remote == "remote_transcode"
    assert args.tmp == "tmp_transcode"
    assert concat_args._action == "concat"
    assert concat_args.src == "src_concat"
    assert concat_args.dest == "dest_concat"
    assert len(argv) == 0

    args, argv = parse_args(raw_extract_hdf5_argv)

    assert args._action == "extract"
    assert args.dataset_id == "imagenet"
    assert args.dataset_format == "hdf5"
    assert args.src == "src_extract_hdf5"
    assert args.dest == "dest_extract_hdf5"
    assert args.indices == 10
    assert args.size == 15
    assert args.batch_size == 5
    assert len(argv) == 0

    args, argv = parse_args(raw_extract_tar_argv)

    assert args._action == "extract"
    assert args.dataset_id == "imagenet"
    assert args.dataset_format == "tar"
    assert args.src == "src_extract_tar"
    assert args.dest == "dest_extract_tar"
    assert args.indices == 10
    assert args.size == 15
    assert args.batch_size == 5
    assert len(argv) == 0

    args, argv = parse_args(raw_extract_tinyimzip_argv)

    assert args._action == "extract"
    assert args.dataset_id == "tinyimagenet"
    assert args.dataset_format == "zip"
    assert args.src == "src_extract_zip"
    assert args.dest == "dest_extract_zip"
    assert args.indices == 10
    assert args.size == 15
    assert args.batch_size == 5
    assert len(argv) == 0

    args, argv = parse_args(raw_extract_tar_w_extra_argv)
    transcode_args, argv = parse_args(argv)

    assert args._action == "extract"
    assert args.dataset_id == "imagenet"
    assert args.dataset_format == "tar"
    assert args.src == "src_extract_tar"
    assert args.dest == "dest_extract_tar"
    assert args.indices == 10
    assert args.size == 15
    assert args.batch_size == 5
    assert transcode_args._action == "transcode"
    assert transcode_args.src == "src_transcode"
    assert transcode_args.dest == "dest_transcode"
    assert transcode_args.ssh_remote == "remote_transcode"
    assert transcode_args.tmp == "tmp_transcode"
    assert len(argv) == 0
