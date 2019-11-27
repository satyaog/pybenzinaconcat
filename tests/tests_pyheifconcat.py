import glob, os, shutil, subprocess

from pyheifconcat.pyheifconcat import FILENAME_TEMPLATE, _get_file_index, \
    _get_clean_filepath, _is_transcoded, \
    _make_index_filepath, _make_transcoded_filepath, \
    concat, extract_archive, transcode, parse_args

TESTS_WORKING_DIR = os.path.abspath('.')
DATA_DIR = os.path.abspath("test_datasets")

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


def test_trancode_files_list():
    dest = "output/dir/"
    dest_dir = os.path.dirname(dest)
    tmp_dir = "tmp"

    tmp_filepaths = []
    for i in range(10):
        tmp_filepaths.append(os.path.join(tmp_dir, "file_{}_5mb.img".format(i)))

    args = parse_args(["transcode", "list", dest])

    try:
        with open("list", "w") as files_list:
            files_list.write('\n'.join(tmp_filepaths[3:]))

        files_bytes = _prepare_transcode_data(tmp_filepaths, 0, tmp_dir, dest_dir)
        targets_bytes = _prepare_transcode_target_data(tmp_filepaths)
        transcode(args)
        _test_trancode(tmp_filepaths, 3, dest_dir, files_bytes, targets_bytes)

    finally:
        shutil.rmtree(".", ignore_errors=True)


def test_trancode_excludes():
    dest = "output/dir/"
    dest_dir = os.path.dirname(dest)
    tmp_dir = "tmp"

    tmp_filepaths = []
    for i in range(10):
        tmp_filepaths.append(os.path.join(tmp_dir, "file_{}_5mb.img".format(i)))

    try:
        with open("list", "w") as files_list:
            files_list.write('\n'.join(tmp_filepaths))
        with open("excludes", "w") as excludes:
            excludes.write('\n'.join(tmp_filepaths[:3]))

        args = parse_args(["transcode", "list", dest, "--excludes=excludes"])

        files_bytes = _prepare_transcode_data(tmp_filepaths, 0, tmp_dir, dest_dir)
        targets_bytes = _prepare_transcode_target_data(tmp_filepaths)
        transcode(args)
        _test_trancode(tmp_filepaths, 3, dest_dir, files_bytes, targets_bytes)

    finally:
        shutil.rmtree(".", ignore_errors=True)


def test_extract_tar():
    src = os.path.join(DATA_DIR, "dev_im_net/dev_im_net.tar")
    dest = "output/dir/extract/"
    dest_dir = os.path.dirname(dest)

    args = parse_args(["extract_archive", "tar", src, dest])

    try:
        extract_archive(args)

        queued_list = glob.glob(os.path.join(dest_dir, '*'))
        queued_list.sort()
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


def test_extract_tar_start_number():
    src = os.path.join(DATA_DIR, "dev_im_net/dev_im_net.tar")
    dest = "output/dir/extract/"
    dest_dir = os.path.dirname(dest)

    args = parse_args(["extract_archive", "tar", src, dest,
                       "--start", "10", "--number", "15"])

    try:
        extract_archive(args)

        queued_list = glob.glob(os.path.join(dest_dir, '*'))
        queued_list.sort()
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


def test_extract_tar_start_number_transcode():
    src = os.path.join(DATA_DIR, "dev_im_net/dev_im_net.tar")
    dest = "output/dir/"
    tmp = "tmp/dir/"
    dest_dir = os.path.dirname(dest)
    tmp_dir = os.path.dirname(tmp)
    upload_dir = os.path.join(dest_dir, "upload")
    queue_dir = os.path.join(dest_dir, "queue")

    args = parse_args(["extract_archive", "tar", src, dest,
                       "--start", "10", "--number", "15",
                       "--transcode", "--tmp", tmp])

    try:
        if upload_dir and not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        if queue_dir and not os.path.exists(queue_dir):
            os.makedirs(queue_dir)

        extract_archive(args)

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

        assert len(glob.glob(os.path.join(tmp_dir, '*'))) == 30
        assert len(glob.glob(os.path.join(upload_dir, '*'))) == 0

    finally:
        shutil.rmtree(".", ignore_errors=True)


def test_extract_tar_start_number_transcode_mp4():
    src = os.path.join(DATA_DIR, "dev_im_net/dev_im_net.tar")
    dest = "./"
    tmp = "extract/"
    dest_dir = os.path.dirname(dest)
    tmp_dir = os.path.dirname(tmp)
    upload_dir = os.path.join(dest_dir, "upload")
    queue_dir = os.path.join(dest_dir, "queue")

    args = parse_args(["extract_archive", "tar", src, dest,
                       "--start", "10", "--number", "15",
                       "--transcode", "--mp4", "--tmp", tmp])

    try:
        if upload_dir and not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        if queue_dir and not os.path.exists(queue_dir):
            os.makedirs(queue_dir)

        extract_archive(args)

        extract_list = glob.glob(os.path.join(tmp_dir, '*'))
        extract_list.sort()
        assert extract_list == \
               ['extract/000000000010.n01443537_2772.JPEG',
                'extract/000000000010.n01443537_2772.JPEG.target',
                'extract/000000000011.n01443537_1029.JPEG',
                'extract/000000000011.n01443537_1029.JPEG.target',
                'extract/000000000012.n01443537_1955.JPEG',
                'extract/000000000012.n01443537_1955.JPEG.target',
                'extract/000000000013.n01443537_962.JPEG',
                'extract/000000000013.n01443537_962.JPEG.target',
                'extract/000000000014.n01443537_2563.JPEG',
                'extract/000000000014.n01443537_2563.JPEG.target',
                'extract/000000000015.n01443537_3344.JPEG',
                'extract/000000000015.n01443537_3344.JPEG.target',
                'extract/000000000016.n01443537_3601.JPEG',
                'extract/000000000016.n01443537_3601.JPEG.target',
                'extract/000000000017.n01443537_2333.JPEG',
                'extract/000000000017.n01443537_2333.JPEG.target',
                'extract/000000000018.n01443537_801.JPEG',
                'extract/000000000018.n01443537_801.JPEG.target',
                'extract/000000000019.n01443537_2228.JPEG',
                'extract/000000000019.n01443537_2228.JPEG.target',
                'extract/000000000020.n01484850_4496.JPEG',
                'extract/000000000020.n01484850_4496.JPEG.target',
                'extract/000000000021.n01484850_2506.JPEG',
                'extract/000000000021.n01484850_2506.JPEG.target',
                'extract/000000000022.n01484850_17864.JPEG',
                'extract/000000000022.n01484850_17864.JPEG.target',
                'extract/000000000023.n01484850_4645.JPEG',
                'extract/000000000023.n01484850_4645.JPEG.target',
                'extract/000000000024.n01484850_22221.JPEG',
                'extract/000000000024.n01484850_22221.JPEG.target']

        queued_list = glob.glob(os.path.join(queue_dir, '*'))
        queued_list.sort()
        assert queued_list == \
               ['./queue/000000000010.n01443537_2772.JPEG.transcoded',
                './queue/000000000011.n01443537_1029.JPEG.transcoded',
                './queue/000000000012.n01443537_1955.JPEG.transcoded',
                './queue/000000000013.n01443537_962.JPEG.transcoded',
                './queue/000000000014.n01443537_2563.JPEG.transcoded',
                './queue/000000000015.n01443537_3344.JPEG.transcoded',
                './queue/000000000016.n01443537_3601.JPEG.transcoded',
                './queue/000000000017.n01443537_2333.JPEG.transcoded',
                './queue/000000000018.n01443537_801.JPEG.transcoded',
                './queue/000000000019.n01443537_2228.JPEG.transcoded',
                './queue/000000000020.n01484850_4496.JPEG.transcoded',
                './queue/000000000021.n01484850_2506.JPEG.transcoded',
                './queue/000000000022.n01484850_17864.JPEG.transcoded',
                './queue/000000000023.n01484850_4645.JPEG.transcoded',
                './queue/000000000024.n01484850_22221.JPEG.transcoded']

        assert len(glob.glob(os.path.join(upload_dir, '*'))) == 0

    finally:
        shutil.rmtree(".", ignore_errors=True)


def test_extract_tar_start_number_transcode_excludes():
    src = os.path.join(DATA_DIR, "dev_im_net/dev_im_net.tar")
    dest = "./"
    tmp = "extract/"
    dest_dir = os.path.dirname(dest)
    tmp_dir = os.path.dirname(tmp)
    upload_dir = os.path.join(dest_dir, "upload")
    queue_dir = os.path.join(dest_dir, "queue")
    transcode_excludes = ['extract/000000000010.n01443537_2772.JPEG.transcoded',
                          'extract/000000000011.n01443537_1029.JPEG.transcoded',
                          'extract/000000000012.n01443537_1955.JPEG.transcoded']

    args = parse_args(["extract_archive", "tar", src, dest,
                       "--start", "10", "--number", "15",
                       "--transcode", "--excludes", "excludes",
                       "--tmp", tmp])

    try:
        if upload_dir and not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        if queue_dir and not os.path.exists(queue_dir):
            os.makedirs(queue_dir)
        with open("excludes", "w") as excludes:
            excludes.write('\n'.join(transcode_excludes[:3]))

        extract_archive(args)

        extract_list = glob.glob(os.path.join(tmp_dir, '*'))
        extract_list.sort()
        assert extract_list == \
               ['extract/000000000010.n01443537_2772.JPEG',
                'extract/000000000010.n01443537_2772.JPEG.target',
                'extract/000000000011.n01443537_1029.JPEG',
                'extract/000000000011.n01443537_1029.JPEG.target',
                'extract/000000000012.n01443537_1955.JPEG',
                'extract/000000000012.n01443537_1955.JPEG.target',
                'extract/000000000013.n01443537_962.JPEG',
                'extract/000000000013.n01443537_962.JPEG.target',
                'extract/000000000014.n01443537_2563.JPEG',
                'extract/000000000014.n01443537_2563.JPEG.target',
                'extract/000000000015.n01443537_3344.JPEG',
                'extract/000000000015.n01443537_3344.JPEG.target',
                'extract/000000000016.n01443537_3601.JPEG',
                'extract/000000000016.n01443537_3601.JPEG.target',
                'extract/000000000017.n01443537_2333.JPEG',
                'extract/000000000017.n01443537_2333.JPEG.target',
                'extract/000000000018.n01443537_801.JPEG',
                'extract/000000000018.n01443537_801.JPEG.target',
                'extract/000000000019.n01443537_2228.JPEG',
                'extract/000000000019.n01443537_2228.JPEG.target',
                'extract/000000000020.n01484850_4496.JPEG',
                'extract/000000000020.n01484850_4496.JPEG.target',
                'extract/000000000021.n01484850_2506.JPEG',
                'extract/000000000021.n01484850_2506.JPEG.target',
                'extract/000000000022.n01484850_17864.JPEG',
                'extract/000000000022.n01484850_17864.JPEG.target',
                'extract/000000000023.n01484850_4645.JPEG',
                'extract/000000000023.n01484850_4645.JPEG.target',
                'extract/000000000024.n01484850_22221.JPEG',
                'extract/000000000024.n01484850_22221.JPEG.target']

        queued_list = glob.glob(os.path.join(queue_dir, '*'))
        queued_list.sort()
        assert queued_list == \
               ['./queue/000000000013.n01443537_962.JPEG.transcoded',
                './queue/000000000014.n01443537_2563.JPEG.transcoded',
                './queue/000000000015.n01443537_3344.JPEG.transcoded',
                './queue/000000000016.n01443537_3601.JPEG.transcoded',
                './queue/000000000017.n01443537_2333.JPEG.transcoded',
                './queue/000000000018.n01443537_801.JPEG.transcoded',
                './queue/000000000019.n01443537_2228.JPEG.transcoded',
                './queue/000000000020.n01484850_4496.JPEG.transcoded',
                './queue/000000000021.n01484850_2506.JPEG.transcoded',
                './queue/000000000022.n01484850_17864.JPEG.transcoded',
                './queue/000000000023.n01484850_4645.JPEG.transcoded',
                './queue/000000000024.n01484850_22221.JPEG.transcoded']

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

    raw_extract_hdf5_arguments = ["extract_archive",
                                  "hdf5",
                                  "src_extract_hdf5",
                                  "dest_extract_hdf5",
                                  "--start", "10",
                                  "--number", "15",
                                  "--jobs", "2",
                                  "--transcode",
                                  "--ssh-remote", "remote_extract_hdf5",
                                  "--tmp", "tmp_extract_hdf5"]

    raw_extract_tar_arguments = ["extract_archive",
                                 "tar",
                                 "src_extract_tar",
                                 "dest_extract_tar",
                                 "--start", "10",
                                 "--number", "15",
                                 "--jobs", "2",
                                 "--transcode",
                                 "--ssh-remote", "remote_extract_tar",
                                 "--tmp", "tmp_extract_tar"]

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

    assert args.action == "extract_archive"
    assert args.type == "hdf5"
    assert args.src == "src_extract_hdf5"
    assert args.dest == "dest_extract_hdf5"
    assert args.start == 10
    assert args.number == 15
    assert args.jobs == 2
    assert args.ssh_remote == "remote_extract_hdf5"
    assert args.tmp == "tmp_extract_hdf5"

    args = parse_args(raw_extract_tar_arguments)

    assert args.action == "extract_archive"
    assert args.type == "tar"
    assert args.src == "src_extract_tar"
    assert args.dest == "dest_extract_tar"
    assert args.start == 10
    assert args.number == 15
    assert args.jobs == 2
    assert args.ssh_remote == "remote_extract_tar"
    assert args.tmp == "tmp_extract_tar"
