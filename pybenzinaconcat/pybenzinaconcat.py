import argparse
import copy
import ctypes
import glob
import importlib.util
import logging
import os
import subprocess
import sys
import tarfile
from collections import namedtuple
from time import sleep

import jug
from jug import TaskGenerator
from jug.utils import identity

h5py_spec = importlib.util.find_spec("h5py")
is_h5py_installed = h5py_spec is not None
if is_h5py_installed:
    import h5py
    import numpy as np

LOGGER = logging.getLogger(os.path.basename(__file__))
LOGGER.setLevel(logging.INFO)

STREAM_HANDLER = logging.StreamHandler()
STREAM_HANDLER.setLevel(logging.INFO)
FORMATTER = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
STREAM_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(STREAM_HANDLER)

ID_FILENAME_TEMPLATE = "{index:012d}.{filename}"
FILENAME_TEMPLATE = ID_FILENAME_TEMPLATE + ".transcoded"


def _is_transcoded(filename):
    return filename.split('.')[-1] == "transcoded"


def _get_file_index(filepath):
    if isinstance(filepath, str):
        splitted_filename = os.path.basename(filepath).split('.')
    else:
        splitted_filename = filepath
    if len(splitted_filename[0]) == 12 and splitted_filename[0].isdigit():
        return int(splitted_filename[0])
    return None


def _get_clean_filepath(filepath, basename=False):
    if isinstance(filepath, str):
        dirname = os.path.dirname(filepath)
        splitted_filename = os.path.basename(filepath).split('.')
    else:
        dirname = None
        splitted_filename = filepath
    if splitted_filename[-1] == "transcoded":
        splitted_filename.pop()
    if _get_file_index(splitted_filename) is not None:
        splitted_filename.pop(0)
    return '.'.join(splitted_filename) if basename else \
           os.path.join(dirname, '.'.join(splitted_filename))


def _make_index_filepath(filepath, index):
    dirname = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    return os.path.join(dirname, ID_FILENAME_TEMPLATE.format(filename=filename,
                                                             index=index))


def _make_target_filepath(filepath):
    dirname = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    return os.path.join(dirname, filename + ".target")


def _make_transcoded_filepath(filepath):
    dirname = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    return os.path.join(dirname, filename + ".transcoded")


def _get_remote_path(ssh_remote, path):
    if ssh_remote:
        return ':'.join([ssh_remote, path])
    else:
        return path


def _get_archive_size(archive, archive_type):
    if archive_type == "hdf5":
        with h5py.File(archive, "r") as file_h5:
            return len(file_h5["encoded_images"])
    else:
        return 0


def _get_dir_hierarchy(root):
    return os.path.join(root, "upload/"), os.path.join(root, "queue/")


def _make_concat_dirs(root, ssh_remote=None):
    upload_dir, queue_dir = _get_dir_hierarchy(root)

    if ssh_remote:
        u_process = subprocess.Popen(["ssh", ssh_remote,
                                      "mkdir -p {}".format(upload_dir)])
        q_process = subprocess.Popen(["ssh", ssh_remote,
                                      "mkdir -p {}".format(queue_dir)])
        if u_process.wait() != 0:
            LOGGER.error("Could not make upload dir [{}] on remote [{}]"
                         .format(upload_dir, ssh_remote))
        if q_process.wait() != 0:
            LOGGER.error("Could not make queue dir [{}] on remote [{}]"
                         .format(queue_dir, ssh_remote))
        assert u_process.wait() != 0 and q_process.wait() != 0
    else:
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(queue_dir, exist_ok=True)

    return upload_dir, queue_dir


@TaskGenerator
def _concat_files(filepaths, dest):
    concatenated_files = []

    with open(dest, "ab") as concat_file:
        for filepath in filepaths:
            # Concat files sequentially
            filepath = identity(filepath)
            # Another thread should not own the lock
            if not filepath.lock():
                break

            try:
                # If filepath task has already executed, then the file has
                # already been concatenated
                if filepath.can_load():
                    LOGGER.warning("Ignoring [{}] since it has already been "
                                   "concatenated".format(jug.value(filepath)))
                    continue

                filepath.run()
                with open(jug.value(filepath), "rb") as f:
                    concat_file.write(f.read())
            except Exception:
                filepath.invalidate()
                raise
            finally:
                filepath.unlock()
            os.remove(jug.value(filepath))
            concatenated_files.append(jug.value(filepath))

    _set = set(concatenated_files)
    
    print("\n".join(concatenated_files))
    return concatenated_files, \
           [f for f in filepaths if f not in _set]


def concat(src, dest, _action=None):
    """ Take a source directory containing files and append/concatenate them
    into a single destination file

    Files contained in the subdirectory 'queue' of the source directory will be
    concatenated

    If they don't exist, the subdirectories 'upload' and 'queue' of the source
    directory will be created
    """
    del _action

    if isinstance(src, str):
        src_dir = src
        _, queue_dir = _make_concat_dirs(src_dir)
        queued_files = glob.glob(os.path.join(queue_dir, '*'))
        queued_files.sort()
        batches = [identity(queued_files)]
    else:
        batches = src

    dest_dir = os.path.dirname(dest)

    if dest_dir and not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    result = []
    for batch in batches:
        # Concat files sequentially. If a task is not ready, wait for its
        # completion
        if jug.is_jug_running():
            while not batch.can_load() and not batch.is_failed():
                sleep(60)
            if not batch.can_load():
                break
        result.append(_concat_files(batch, dest))
    return result


@TaskGenerator
def transcode_img(input_path, dest_dir, mp4, ssh_remote=None, tmp=None):
    upload_dir, queue_dir = _get_dir_hierarchy(dest_dir)
    tmp_dir = tmp if tmp is not None else \
              os.path.dirname(input_path)
    filename = os.path.basename(input_path)
    clean_filename = _get_clean_filepath(input_path, basename=True)
    target_path = _make_target_filepath(input_path)

    if tmp_dir and not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    output_path = _make_transcoded_filepath(os.path.join(tmp_dir, filename))
    command = ["python", "-m", "pybenzinaconcat.image2mp4"] \
              if mp4 else ["image2heif"]
    cmd_arguments = " --codec=h265 --tile=512:512:yuv420 --crf=10 " \
                    "--output={dest} " \
                    "--primary --thumb --name={name} " \
                    "--item=path={src}" \
                    .format(name=clean_filename,
                            src=input_path, dest=output_path)
    if os.path.exists(target_path):
        cmd_arguments += " --hidden --name=target " \
                         "--mime=application/octet-stream " \
                         "--item=type=mime,path={target}" \
                         .format(target=target_path)
    else:
        target_path = None

    try:
        subprocess.run(command +
                       ["--" + arg for arg in cmd_arguments.split(" --")[1:]],
                       check=True)
    except subprocess.CalledProcessError:
        LOGGER.error("Could not transcode file [{}] with target [{}] to [{}]"
                     .format(input_path, target_path, output_path))
        return

    uploaded_path = os.path.join(upload_dir, os.path.basename(output_path))

    try:
        subprocess.run(["rsync", "-v", "--remove-source-files", output_path,
                        _get_remote_path(ssh_remote, upload_dir)],
                       check=True)
    except subprocess.CalledProcessError:
        LOGGER.error("Could not move file [{}] to upload dir [{}]"
                     .format(output_path, upload_dir))
        return

    queued_path = os.path.join(queue_dir, os.path.basename(output_path))
    if ssh_remote:
        try:
            subprocess.run(["ssh", ssh_remote,
                            "mv -v {} {}".format(uploaded_path, queued_path)],
                           check=True)
        except subprocess.CalledProcessError:
            LOGGER.error("Could not move file [{}] to queue dir [{}]"
                         .format(uploaded_path, queued_path))
            return
    else:
        os.rename(uploaded_path, queued_path)

    return queued_path


@TaskGenerator
def transcode(src, dest, excludes=None, mp4=True, ssh_remote=None, tmp=None,
              _action=None):
    """ Take a list of images and transcode them into a destination directory

    The suffix ".transcoded" will be appended to the file's base name

    Subdirectories 'upload' and 'queue' will be created in destination
    directory where 'upload' contains the files that are being uploaded and
    'queue' contains the files which are ready to be concatenated
    """
    del _action

    dest_dir = dest

    if isinstance(src, str):
        source = src.split(',')
        if len(source) == 1 and os.path.basename(source[0]) == "list":
            with open(source[0], 'r') as files_list:
                source = files_list.read().split('\n')
    else:
        source = src

    if excludes is not None:
        with open(excludes.name, excludes.mode) as f:
            excluded_files = f.read().split('\n')

        for i, exclude in enumerate(excluded_files):
            excluded_files[i] = _get_clean_filepath(exclude, basename=True)
    else:
        excluded_files = []

    transcoded_imgs = []
    for input_path in source:
        clean_basename = _get_clean_filepath(input_path, basename=True)
        if clean_basename in excluded_files:
            LOGGER.info("Ignoring [{}] since [{}] is in [{}]"
                        .format(input_path, clean_basename, excludes.name))
            continue
        # Transcode batch sequentially
        transcoded_img = transcode_img(input_path, dest_dir, mp4, ssh_remote,
                                       tmp)
        # Another thread should not own the lock
        if not transcoded_img.lock():
            break

        try:
            # If transcoded_img task has already executed, then the file has
            # already been transcoded and queued
            if transcoded_img.can_load():
                LOGGER.warning("Ignoring [{}] since it has already been "
                               "transcoded and moved to {}:{}"
                               .format(input_path, ssh_remote, dest_dir))
                continue
            transcoded_img.run()
        except Exception:
            transcoded_img.invalidate()
            raise
        finally:
            transcoded_img.unlock()

        if jug.value(transcoded_img) is None:
            transcoded_img.invalidate()
        else:
            transcoded_imgs.append(jug.value(transcoded_img))

    return transcoded_imgs


@TaskGenerator
def extract_hdf5(src, dest, start, size, tmp=None):
    """ Take a source HDF5 file and extract images from it into a destination
    directory
    """
    extract_dir = tmp if tmp is not None else dest

    extracted_filenames = []

    with h5py.File(src, "r") as file_h5:
        num_elements = len(file_h5["encoded_images"])
        num_targets = len(file_h5["targets"])

        start = start
        end = min(start + size, num_elements) if size else num_elements

        for i in range(start, end):
            filename = file_h5["filenames"][i][0].decode("utf-8")
            filename = _make_index_filepath(filename, i)
            extract_filepath = os.path.join(extract_dir, filename)
            target_filepath = _make_target_filepath(extract_filepath)

            extracted_filenames.append(extract_filepath)

            if not os.path.exists(extract_filepath):
                img = bytes(file_h5["encoded_images"][i])

                with open(extract_filepath, "xb") as file:
                    file.write(img)

            if not os.path.exists(target_filepath) and i < num_targets:
                target = file_h5["targets"][i].astype(np.int64).tobytes()

                with open(target_filepath, "xb") as file:
                    file.write(target)

    return extracted_filenames


@TaskGenerator
def extract_tar(src, dest, start, size, tmp=None):
    """ Take a source tar file and extract images from it into a destination
    directory
    """
    extract_dir = tmp if tmp is not None else dest

    extracted_filenames = []

    index = 0
    end = start + size if size else ctypes.c_ulonglong(-1).value

    with tarfile.open(src, "r") as file_tar:
        for target_idx, member in enumerate(file_tar):
            if index >= end:
                break
            sub_tar = file_tar.extractfile(member)
            file_sub_tar = tarfile.open(fileobj=sub_tar, mode="r")
            for sub_member in file_sub_tar:
                if index >= end:
                    break

                if index >= start:
                    filename = sub_member.name
                    filename = _make_index_filepath(filename, index)
                    extract_filepath = os.path.join(extract_dir, filename)
                    target_filepath = _make_target_filepath(extract_filepath)

                    extracted_filenames.append(extract_filepath)

                    if not os.path.exists(extract_filepath):
                        file_sub_tar.extract(sub_member, extract_dir)
                        os.rename(os.path.join(extract_dir, sub_member.name),
                                  extract_filepath)

                    if not os.path.exists(target_filepath):
                        target = target_idx.to_bytes(8, byteorder="little")

                        with open(target_filepath, "xb") as file:
                            file.write(target)

                index += 1

    return extracted_filenames


def extract_batch(archive_type, **kwargs):
    if archive_type == "hdf5":
        return extract_hdf5(**kwargs)
    else:
        return extract_tar(**kwargs)


def extract(_action=None, **kwargs):
    """ Take a source archive file and extract images from it into a
    destination directory.
    """
    del _action

    if kwargs["size"] == 0:
        kwargs["size"] = _get_archive_size(kwargs["src"],
                                           kwargs["archive_type"])

    if kwargs["size"] and kwargs["batch_size"]:
        kwargs["batch_size"] = min(kwargs["size"], kwargs["batch_size"])
        processes_kwargs = []
        for start in range(kwargs["start"], kwargs["start"] + kwargs["size"],
                           kwargs["batch_size"]):
            process_kwargs = copy.deepcopy(kwargs)
            del process_kwargs["batch_size"]
            process_kwargs["start"] = start
            process_kwargs["size"] = kwargs["batch_size"]
            processes_kwargs.append(process_kwargs)
    else:
        del kwargs["batch_size"]
        processes_kwargs = [kwargs]

    # Minimize async issues when trying to create the same directory multiple
    # times and at the same time
    tmp_dir = kwargs.get("tmp", None)
    extract_dir = tmp_dir if tmp_dir is not None else kwargs["dest"]
    if extract_dir and not os.path.exists(extract_dir):
        os.makedirs(extract_dir)

    return [extract_batch(**kwargs) for kwargs in processes_kwargs]


FileDesc = namedtuple("FileDesc", ["name", "mode"])


class ChainAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        action = option_string.lstrip('-')
        setattr(namespace, self.dest, [action] + values)


class CheckFileType(argparse.FileType):
    def __call__(self, string):
        f = super(CheckFileType, self).__call__(string)
        f.close()
        return FileDesc(f.name, f.mode)


def build_base_parser():
    parser = argparse.ArgumentParser(description="Benzina HEIF Concatenation",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("_action", metavar="action",
                        choices=list(ACTIONS.keys()), help="action to execute")
    parser.add_argument('args', nargs=argparse.REMAINDER,
                        help="action's arguments")

    return parser


def build_concat_parser():
    parser = argparse.ArgumentParser(description="Benzina HEIF Concatenation action: "
                                                 "concat",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("_action", metavar="concat", help="action to execute")
    parser.add_argument("src", metavar="source",
                        help="directory containing a subdirectory 'queue' of "
                             "files to concatenate")
    parser.add_argument("dest", metavar="destination",
                        help="concatenated file")

    return parser


def build_transcode_parser():
    parser = argparse.ArgumentParser(description="Benzina HEIF Concatenation action: "
                                                 "transcode",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("_action", metavar="transcode",
                        help="action to execute")
    parser.add_argument("src", metavar="source",
                        help="file or ',' separated files to transcode or a "
                             "file named 'list' containing the explicit list "
                             "of files to transcode")
    parser.add_argument("dest", metavar="destination",
                        help="directory to write the transcoded file(s)")
    parser.add_argument("--excludes", default=None, type=CheckFileType('r'),
                        help="a text file containing the list of files to exclude")
    parser.add_argument("--mp4", default=False, action="store_true",
                        help="use image2mp4 instead of image2heif")
    parser.add_argument("--ssh-remote", metavar="REMOTE",
                        help="optional remote to use to transfer the transcoded "
                             "file to destination")
    parser.add_argument("--tmp", metavar="DIR",
                        help="the directory to use to store temporary file(s)")

    parser.add_argument("--concat", metavar="...", action=ChainAction,
                        dest="_chain", nargs=argparse.REMAINDER,
                        help="chain the concat action. concat will be fed by "
                             "transcode's dest through its src.")

    return parser


def build_extract_parser():
    parser = argparse.ArgumentParser(description="Benzina HEIF Concatenation action: "
                                                 "extract",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("_action", metavar="extract", help="action to execute")
    parser.add_argument("src", metavar="source",
                        help="archive file to extract the images from")
    parser.add_argument("dest", metavar="destination",
                        help="directory to write the extracted file(s)")
    parser.add_argument("archive_type",
                        choices=["hdf5", "tar"] if is_h5py_installed
                        else ["tar"],
                        help="type of the archive")
    parser.add_argument("--start", metavar="IDX", default=0, type=int,
                        help="the start element index to transcode from source")
    parser.add_argument("--size", default=0, metavar="NUM", type=int,
                        help="the number of elements to extract from source")
    parser.add_argument("--batch-size", default=512, metavar="NUM", type=int,
                        help="the batch size for a single job.")

    parser.add_argument("--transcode", metavar="...", action=ChainAction,
                        dest="_chain", nargs=argparse.REMAINDER,
                        help="chain the transcode action. transcode will be "
                             "fed by extract's dest through its src.")

    return parser


def parse_args(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    is_help_request = "-h" in argv

    if is_help_request:
        if len(argv) == 1:
            build_base_parser().parse_args(argv)
        argv.remove("-h")
        base_args = build_base_parser().parse_args(argv)
        ACTIONS_PARSER.get(base_args._action, None).parse_args(argv + ["-h"])

    base_args = build_base_parser().parse_args(argv)
    args = ACTIONS_PARSER.get(base_args._action, None) \
        .parse_args([base_args._action] + base_args.args)
    try:
        argv = args._chain or tuple()
        del args._chain
    except AttributeError:
        argv = tuple()
    return args, argv


def pybenzinaconcat(args, argv=None):
    result_arr = ACTIONS.get(args._action, None)(**vars(args))

    if isinstance(result_arr, jug.Task):
        result_arr = [result_arr]

    while argv:
        # Insert src placeholder
        argv.insert(1, "_")
        args, argv = parse_args(argv)
        if args._action == "transcode":
            args = list(map(lambda result: {**vars(args), "src": result},
                        result_arr))
        elif args._action == "concat":
            args = [{**vars(args), "src": result_arr}]
        else:
            args = [vars(args)]
        result_arr = [ACTIONS.get(_args["_action"], None)(**_args)
                      for _args in args]
    
    return result_arr


ACTIONS = {"transcode": transcode, "concat": concat, "extract": extract}
ACTIONS_PARSER = {"concat": build_concat_parser(),
                  "transcode": build_transcode_parser(),
                  "extract": build_extract_parser(),
                  "_": build_base_parser()}
