import glob, os, shutil, subprocess

from pyheifconcat import FILENAME_TEMPLATE, _get_file_index, \
    _get_clean_filepath, _is_transcoded, \
    _make_index_filepath, _make_transcoded_filepath, \
    concat, transcode, parse_args

TESTS_WORKING_DIR = os.path.abspath('.')

os.environ["PATH"] = ':'.join([os.environ["PATH"],
                               os.path.join(TESTS_WORKING_DIR, "mocks")])

PWD = "tests_tmp"

if PWD and not os.path.exists(PWD):
    os.makedirs(PWD)

os.chdir(PWD)


def _prepare_concat_data(to_concat_filepaths, nb_files_to_skip,
                         completed_list_filepath, queue_dir, dest_dir):
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
        with open(to_concat_filepaths[i], "rb") as file:
            files_bytes.append(file.read())
            assert len(files_bytes[i]) == 5000 * 1000

    with open(completed_list_filepath, "w") as completed_list_file:
        for to_concat_filepath in to_concat_filepaths[:nb_files_to_skip]:
            completed_list_file.write(to_concat_filepath)
            completed_list_file.write('\n')

    return files_bytes


def _test_concat(to_concat_filepaths, nb_files_to_skip,
                 completed_list_filepath, files_bytes, args):
    with open(args.dest, "rb") as file:
        assert file.read() == b''.join(files_bytes[nb_files_to_skip:])

    with open(completed_list_filepath, "r") \
            as completed_list:
        assert list(filter(None, completed_list.read().split('\n'))) == \
               to_concat_filepaths


def test_concat():
    src = "input/dir/"
    dest = "output/dir/concat.bza"
    src_dir = os.path.dirname(src)
    dest_dir = os.path.dirname(dest)
    queue_dir = os.path.join(src_dir, "queue")
    completed_list_filepath = os.path.join(src_dir, "completed_list")

    to_concat_filepaths = []
    for i in range(10):
        to_concat_filepaths.append(os.path.join(queue_dir,
                                                "file_{}_5mb.img.transcoded"
                                                .format(i)))

    args = parse_args(["concat", src, dest])

    try:
        files_bytes = \
            _prepare_concat_data(to_concat_filepaths, 0,
                                 completed_list_filepath, queue_dir, dest_dir)
        concat(args)
        _test_concat(to_concat_filepaths, 0, completed_list_filepath,
                     files_bytes, args)

    finally:
        shutil.rmtree(".", ignore_errors=True)


def test_concat_completed_3():
    src = "input/dir/"
    dest = "output/dir/concat.bza"
    src_dir = os.path.dirname(src)
    dest_dir = os.path.dirname(dest)
    queue_dir = os.path.join(src_dir, "queue")
    completed_list_filepath = os.path.join(src_dir, "completed_list")

    to_concat_filepaths = []
    for i in range(10):
        to_concat_filepaths.append(os.path.join(queue_dir,
                                                "file_{}_5mb.img.transcoded"
                                                .format(i)))

    args = parse_args(["concat", src, dest])

    try:
        files_bytes = \
            _prepare_concat_data(to_concat_filepaths, 3,
                                 completed_list_filepath, queue_dir, dest_dir)
        concat(args)
        _test_concat(to_concat_filepaths, 3, completed_list_filepath,
                     files_bytes, args)

    finally:
        shutil.rmtree(".", ignore_errors=True)


def test_concat_index_completed_3():
    src = "input/dir/"
    dest = "output/dir/concat.bza"
    src_dir = os.path.dirname(src)
    dest_dir = os.path.dirname(dest)
    queue_dir = os.path.join(src_dir, "queue")
    completed_list_filepath = os.path.join(src_dir, "completed_list")

    to_concat_filepaths = []
    for i in range(10):
        filepath = os.path.join(queue_dir, "file_{}_5mb.img.transcoded"
                                           .format(i))
        filepath = _make_index_filepath(filepath, i)
        to_concat_filepaths.append(filepath)

    args = parse_args(["concat", src, dest])

    try:
        files_bytes = \
            _prepare_concat_data(to_concat_filepaths, 3,
                                 completed_list_filepath, queue_dir, dest_dir)
        concat(args)
        _test_concat(to_concat_filepaths, 3, completed_list_filepath,
                     files_bytes, args)

    finally:
        shutil.rmtree(".", ignore_errors=True)


def test_concat_no_queue():
    src = "input/dir/"
    dest = "output/dir/concat.bza"
    src_dir = os.path.dirname(src)
    completed_list_filepath = os.path.join(src_dir, "completed_list")

    to_concat_filepaths = []

    args = parse_args(["concat", src, dest])

    try:
        concat_bytes = b''
        concat(args)

        assert os.path.exists(os.path.join(src_dir, "upload/"))
        assert os.path.exists(os.path.join(src_dir, "queue/"))

        with open(args.dest, "rb") as file:
            assert file.read() == concat_bytes

        with open(completed_list_filepath, "r") \
             as completed_list:
            assert list(filter(None, completed_list.read().split('\n'))) == \
                   to_concat_filepaths

    finally:
        shutil.rmtree(".", ignore_errors=True)


def _prepare_transcode_data(tmp_filepaths, nb_files_to_skip, tmp_dir, dest_dir):
    upload_dir = os.path.join(dest_dir, "upload")
    queue_dir = os.path.join(dest_dir, "queue")
    completed_list_filepath = os.path.join(dest_dir, "completed_list")

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
        with open(tmp_filepaths[i], "rb") as file:
            files_bytes.append(file.read())
            assert len(files_bytes[i]) == 5000 * 1000

    with open(completed_list_filepath, "w") as completed_list_file:
        for tmp_filepath in tmp_filepaths[:nb_files_to_skip]:
            tmp_filename = os.path.basename(tmp_filepath)
            transcoded_filename = _make_transcoded_filepath(tmp_filename)
            completed_list_file.write(os.path.join(queue_dir, transcoded_filename))
            completed_list_file.write('\n')

    return files_bytes


def _prepare_transcode_target_data(tmp_filepaths):
    tragets_bytes = []
    for i, tmp_filepath in enumerate(tmp_filepaths):
        tragets_bytes.append(i.to_bytes(8, byteorder="little"))
        with open(tmp_filepath + ".target", "xb") as file:
            file.write(tragets_bytes[-1])

    return tragets_bytes


def _test_trancode(tmp_filepaths, nb_files_to_skip, dest_dir,
                   files_bytes, targets_bytes):
    upload_dir = os.path.join(dest_dir, "upload")
    queue_dir = os.path.join(dest_dir, "queue")

    assert len(glob.glob(os.path.join(upload_dir, '*'))) == 0

    queued_list = glob.glob(os.path.join(queue_dir, '*'))
    queued_list.sort()
    assert len(queued_list) == len(tmp_filepaths) - nb_files_to_skip

    for i, filepath in enumerate(queued_list):
        with open(filepath, "rb") as file:
            file_bytes = file.read()
        assert file_bytes == files_bytes[i + nb_files_to_skip] + \
                             targets_bytes[i + nb_files_to_skip]


def test_trancode():
    dest = "output/dir/"
    dest_dir = os.path.dirname(dest)
    tmp_dir = "tmp"

    tmp_filepaths = []
    for i in range(10):
        tmp_filepaths.append(os.path.join(tmp_dir, "file_{}_5mb.img".format(i)))

    args = parse_args(["transcode", ','.join(tmp_filepaths), dest])

    try:
        files_bytes = _prepare_transcode_data(tmp_filepaths, 0, tmp_dir, dest_dir)
        targets_bytes = [b'' for _ in range(len(tmp_filepaths))]
        transcode(args)
        _test_trancode(tmp_filepaths, 0, dest_dir, files_bytes, targets_bytes)

    finally:
        shutil.rmtree(".", ignore_errors=True)


def test_trancode_completed_3():
    dest = "output/dir/"
    dest_dir = os.path.dirname(dest)
    tmp_dir = "tmp"

    tmp_filepaths = []
    for i in range(10):
        tmp_filepaths.append(os.path.join(tmp_dir, "file_{}_5mb.img".format(i)))

    args = parse_args(["transcode", ','.join(tmp_filepaths), dest])

    try:
        files_bytes = _prepare_transcode_data(tmp_filepaths, 3, tmp_dir, dest_dir)
        targets_bytes = [b'' for _ in range(len(tmp_filepaths))]
        transcode(args)
        _test_trancode(tmp_filepaths, 3, dest_dir, files_bytes, targets_bytes)

    finally:
        shutil.rmtree(".", ignore_errors=True)


def test_trancode_index_completed_3():
    dest = "output/dir/"
    dest_dir = os.path.dirname(dest)
    tmp_dir = "tmp"

    tmp_filepaths = []
    for i in range(10):
        filepath = os.path.join(tmp_dir, "file_{}_5mb.img"
                                         .format(i))
        filepath = _make_index_filepath(filepath, i)
        tmp_filepaths.append(filepath)

    args = parse_args(["transcode", ','.join(tmp_filepaths), dest])

    try:
        files_bytes = _prepare_transcode_data(tmp_filepaths, 3, tmp_dir, dest_dir)
        targets_bytes = [b'' for _ in range(len(tmp_filepaths))]
        transcode(args)
        _test_trancode(tmp_filepaths, 3, dest_dir, files_bytes, targets_bytes)

    finally:
        shutil.rmtree(".", ignore_errors=True)


def test_trancode_target_data():
    dest = "output/dir/"
    dest_dir = os.path.dirname(dest)
    tmp_dir = "tmp"

    tmp_filepaths = []
    for i in range(10):
        tmp_filepaths.append(os.path.join(tmp_dir, "file_{}_5mb.img".format(i)))

    args = parse_args(["transcode", ','.join(tmp_filepaths), dest])

    try:
        files_bytes = _prepare_transcode_data(tmp_filepaths, 0, tmp_dir, dest_dir)
        targets_bytes = _prepare_transcode_target_data(tmp_filepaths)
        transcode(args)
        _test_trancode(tmp_filepaths, 0, dest_dir, files_bytes, targets_bytes)

    finally:
        shutil.rmtree(".", ignore_errors=True)


def test_trancode_target_data_completed_3():
    dest = "output/dir/"
    dest_dir = os.path.dirname(dest)
    tmp_dir = "tmp"

    tmp_filepaths = []
    for i in range(10):
        tmp_filepaths.append(os.path.join(tmp_dir, "file_{}_5mb.img".format(i)))

    args = parse_args(["transcode", ','.join(tmp_filepaths), dest])

    try:
        files_bytes = _prepare_transcode_data(tmp_filepaths, 3, tmp_dir, dest_dir)
        targets_bytes = _prepare_transcode_target_data(tmp_filepaths)
        transcode(args)
        _test_trancode(tmp_filepaths, 3, dest_dir, files_bytes, targets_bytes)

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

    assert _get_clean_filepath(filename) == "some_filename.extension"
    assert _get_clean_filepath('.'.join(splitted_filename[:-1])) == \
           "some_filename.extension"
    assert _get_clean_filepath('.'.join(splitted_filename[1:])) == \
           "some_filename.extension"
    assert _get_clean_filepath('.'.join(splitted_filename[1:-1])) == \
           "some_filename.extension"
    assert _get_clean_filepath("dir/dir/" + filename, basename=True) == \
           "some_filename.extension"
    assert _get_clean_filepath("dir/dir/" + filename) == \
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
    raw_concat_arguments = ["concat", "src_concat", "dest_concat"]

    raw_transcode_arguments = ["transcode",
                               "src_transcode",
                               "dest_transcode",
                               "--ssh-remote", "remote_transcode",
                               "--tmp", "tmp_transcode"]

    raw_extract_hdf5_arguments = ["extract_hdf5",
                                  "src_extract_hdf5",
                                  "dest_extract_hdf5",
                                  "--start", "10",
                                  "--number", "15",
                                  "--transcode",
                                  "--ssh-remote", "remote_extract_hdf5",
                                  "--tmp", "tmp_extract_hdf5"]

    args = parse_args(raw_concat_arguments)

    assert args.action == "concat"
    assert args.src == "src_concat"
    assert args.dest == "dest_concat"

    args = parse_args(raw_transcode_arguments)

    assert args.action == "transcode"
    assert args.src == "src_transcode"
    assert args.dest == "dest_transcode"
    assert args.ssh_remote == "remote_transcode"
    assert args.tmp == "tmp_transcode"

    args = parse_args(raw_extract_hdf5_arguments)

    assert args.action == "extract_hdf5"
    assert args.src == "src_extract_hdf5"
    assert args.dest == "dest_extract_hdf5"
    assert args.start == 10
    assert args.number == 15
    assert args.ssh_remote == "remote_extract_hdf5"
    assert args.tmp == "tmp_extract_hdf5"
