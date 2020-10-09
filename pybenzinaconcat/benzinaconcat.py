import argparse
import copy
import glob
import importlib.util
import logging
import os
import subprocess
import sys
from collections import namedtuple

import jug
import jug.utils
from jug import TaskGenerator
from PIL import Image

import pybenzinaconcat.datasets as datasets
from pybenzinaconcat import utils

h5py_spec = importlib.util.find_spec("h5py")
is_h5py_installed = h5py_spec is not None

LOGGER = logging.getLogger(os.path.basename(__file__))
LOGGER.setLevel(logging.INFO)

STREAM_HANDLER = logging.StreamHandler()
STREAM_HANDLER.setLevel(logging.INFO)
FORMATTER = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
STREAM_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(STREAM_HANDLER)


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


def _concat_file(concat_f, filepath):
    # Concat file only once
    task = jug.utils.identity("concat_file:" + filepath)

    # No other thread should own the lock
    if not task.lock():
        raise RuntimeError("Could not obtain concat file task lock "
                           "for image [{}]. Task's hash is [{}]"
                           .format(filepath, task.hash()))

    try:
        # If filepath task has already executed, then the file has
        # already been concatenated
        if task.can_load():
            LOGGER.warning("Ignoring [{}] since it has already been "
                           "concatenated".format(filepath))
            return None

        task.run()
        with open(filepath, "rb") as f:
            concat_f.write(f.read())
    except Exception:
        task.fail()
        task.invalidate()
        raise
    finally:
        if not task.is_failed():
            task.unlock()

    os.remove(filepath)
    return filepath


@TaskGenerator
def _concat_batch(batch, last_batch, dest):
    last_batch_file_idx = -1

    if last_batch is not None:
        last_concat_files, _ = last_batch
        if last_concat_files:
            last_batch_file_idx = utils._get_file_index(last_concat_files[-1])

    # "Lock" concat file using jug
    task = jug.utils.identity("concat_file:" + dest)

    # No other thread should own the lock
    if not task.lock():
        raise RuntimeError("Could not obtain concat file task lock for "
                           "archive [{}]. Task's hash is [{}]"
                           .format(dest, task.hash()))

    concatenated_files = []

    try:
        with open(dest, "ab") as concat_f:
            for i, filepath in enumerate(batch):
                # Concat files sequentially
                file_index = utils._get_file_index(filepath)
                if file_index is not None:
                    assert utils._get_file_index(filepath) > last_batch_file_idx
                    last_batch_file_idx = utils._get_file_index(filepath)

                if _concat_file(concat_f, filepath) is not None:
                    concatenated_files.append(filepath)

        if len(concatenated_files) == 0 and os.stat(dest).st_size == 0:
            os.remove(dest)
    finally:
        task.unlock()

    concatenated_set = set(concatenated_files)

    print("\n".join(concatenated_files))
    return concatenated_files, \
           [f for f in batch if f not in concatenated_set]


def concat(src, dest):
    """ Take a source directory containing files and append/concatenate them
    into a single destination file

    Files contained in the subdirectory 'queue' of the source directory will be
    concatenated

    If they don't exist, the subdirectories 'upload' and 'queue' of the source
    directory will be created
    """
    if isinstance(src, str):
        src_dir = src
        _, queue_dir = _make_concat_dirs(src_dir)
        queued_files = glob.glob(os.path.join(queue_dir, '*'))
        queued_files.sort()
        batches = [jug.utils.identity(queued_files)]
    else:
        batches = src

    dest_dir = os.path.dirname(dest)

    if dest_dir and not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    result = [None]
    for batch in batches:
        # Concat batches sequentially
        result.append(_concat_batch(batch, result[-1], dest))
    return result[1:]


def to_bmp(input_path, dest_dir):
    im = Image.open(input_path, 'r')
    filename = os.path.basename(input_path)
    filename = os.path.join(dest_dir, os.path.splitext(filename)[0] + ".BMP")
    target_path = utils._make_target_filepath(input_path)
    if os.path.isfile(target_path):
        with open(target_path, "rb") as f:
            target = f.read()
        target_filename = utils._make_target_filepath(filename)
        with open(target_filename, "wb") as f:
            f.write(target)
    im.save(filename, "BMP")
    return filename


@TaskGenerator
def transcode_img(input_path, dest_dir, clean_basename, mp4, ssh_remote=None,
                  tmp=None):
    upload_dir, queue_dir = _get_dir_hierarchy(dest_dir)
    tmp_dir = tmp if tmp is not None else \
              os.path.dirname(input_path)
    filename = os.path.basename(input_path)
    target_path = utils._make_target_filepath(input_path)

    if tmp_dir and not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    output_path = utils._make_transcoded_filepath(os.path.join(tmp_dir, filename))
    command = ["python", "-m", "pybenzinaconcat.image2mp4"] \
              if mp4 else ["image2heif"]
    cmd_arguments = " --codec=h265 --tile=512:512:yuv420 --crf=10 " \
                    "--output={dest} " \
                    "--primary --thumb --name={name} " \
                    "--item=path={src}" \
                    .format(name=clean_basename,
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
        return None

    uploaded_path = os.path.join(upload_dir, os.path.basename(output_path))

    try:
        subprocess.run(["rsync", "-v", "--remove-source-files", output_path,
                        utils._get_remote_path(ssh_remote, upload_dir)],
                       check=True)
    except subprocess.CalledProcessError:
        LOGGER.error("Could not move file [{}] to upload dir [{}]"
                     .format(output_path, upload_dir))
        return None

    queued_path = os.path.join(queue_dir, os.path.basename(output_path))
    if ssh_remote:
        try:
            subprocess.run(["ssh", ssh_remote,
                            "mv -v {} {}".format(uploaded_path, queued_path)],
                           check=True)
        except subprocess.CalledProcessError:
            LOGGER.error("Could not move file [{}] to queue dir [{}]"
                         .format(uploaded_path, queued_path))
            return None
    else:
        os.rename(uploaded_path, queued_path)

    return queued_path


def try_transcode_task(task, input_path):
    # Unlock to retry the task
    if task.is_failed():
        task.unlock()

    # No other thread should own the lock
    if not task.lock():
        raise RuntimeError("Could not obtain transcoding task lock for image "
                           "[{}]. Task's hash is [{}]"
                           .format(input_path, task.hash()))

    try:
        # If task has already executed, then the file has already been
        # transcoded and queued
        if task.can_load():
            LOGGER.warning("Ignoring [{}] since it has already been transcoded"
                           .format(input_path))
            return not task.is_failed()

        task.run()
        if task.result is None:
            task.fail()
            task.invalidate()
            return False
    except Exception:
        task.fail()
        task.invalidate()
        raise
    finally:
        if not task.is_failed():
            task.unlock()

    return not task.is_failed()


@TaskGenerator
def transcode_batch(src, dest, exclude_files=tuple(), mp4=True,
                    ssh_remote=None, tmp=None):
    exclude_files = set(exclude_files)

    transcoded_imgs = []
    failed_imgs = []
    for input_path in src:
        clean_basename = utils._get_clean_filepath(input_path, basename=True)
        if clean_basename in exclude_files:
            LOGGER.info("Ignoring [{}] since [{}] is excluded"
                        .format(input_path, clean_basename))
            continue

        # Transcode files sequentially and only once
        task = transcode_img(input_path, dest, clean_basename, mp4,
                             ssh_remote=ssh_remote, tmp=tmp)

        for i in range(2):
            if try_transcode_task(task, input_path):
                break
        else:
            try:
                # Transcode to BMP prior to H.265 to work around ffmpeg errors
                bmp_path = to_bmp(input_path, tmp)
                LOGGER.warning("Extra transcode step on [{}]: BMP written at "
                               "[{}]".format(input_path, bmp_path))
                task = transcode_img(bmp_path, dest, clean_basename, mp4,
                                     ssh_remote=ssh_remote, tmp=tmp)
                try_transcode_task(task, input_path)
            except FileNotFoundError:
                pass

        if task.is_failed():
            failed_imgs.append(input_path)
        else:
            transcoded_imgs.append(jug.value(task))

    if failed_imgs:
        raise RuntimeError("Could not transcode all images [{}]"
                           .format(','.join(src[:2] + ["..."] + src[-1:])))

    return transcoded_imgs


def transcode(src, dest, excludes=None, mp4=True, ssh_remote=None, tmp=None):
    """ Take a list of images and transcode them into a destination directory

    The suffix ".transcoded" will be appended to the file's base name

    Subdirectories 'upload' and 'queue' will be created in destination
    directory where 'upload' contains the files that are being uploaded and
    'queue' contains the files which are ready to be concatenated
    """
    if isinstance(src, str):
        source = src.split(',')
        if len(source) == 1 and os.path.basename(source[0]) == "list":
            with open(source[0], 'r') as files_list:
                source = files_list.read().split('\n')
        source.sort()
    else:
        source = src

    if excludes is not None:
        with open(excludes.name, excludes.mode) as f:
            exclude_files = f.read().split('\n')

        for i, exclude in enumerate(exclude_files):
            exclude_files[i] = utils._get_clean_filepath(exclude,
                                                         basename=True)
        exclude_files.sort()
    else:
        exclude_files = []

    source = jug.utils.identity(source)
    exclude_files = jug.utils.identity(exclude_files)
    return transcode_batch(source, dest, exclude_files, mp4, ssh_remote, tmp)


def extract(src, dest, dataset_id, dataset_format, start=0, size=0,
            batch_size=0):
    """ Take a source archive file and extract images from it into a
    destination directory.
    """
    kwargs = {**locals()}
    del kwargs["src"]
    del kwargs["dataset_id"]
    del kwargs["dataset_format"]

    dataset_cls = datasets.get_cls(dataset_id)
    dataset = dataset_cls(src, dataset_format)

    if size == 0:
        size = dataset.size

    if size and batch_size:
        batch_size = min(size, batch_size)
        processes_kwargs = []
        for start in range(start, start + size, batch_size):
            process_kwargs = copy.deepcopy(kwargs)
            del process_kwargs["batch_size"]
            process_kwargs["start"] = start
            process_kwargs["size"] = batch_size
            processes_kwargs.append(process_kwargs)
    else:
        del kwargs["batch_size"]
        processes_kwargs = [kwargs]

    # Minimize async issues when trying to create the same directory multiple
    # times and at the same time
    if dest and not os.path.exists(dest):
        os.makedirs(dest, exist_ok=True)

    dataset = jug.utils.identity(dataset)
    return [dataset_cls.extract(dataset, **kwargs) for kwargs in processes_kwargs]


FileDesc = namedtuple("FileDesc", ["name", "mode"])


class ChainAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        action = option_string.lstrip('-')
        setattr(namespace, self.dest, [action] + values)


class DatasetAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        dataset_id, ar_format = values.split(':')
        dest = self.dest.lstrip('_')
        setattr(namespace, dest + "_id", dataset_id)
        setattr(namespace, dest + "_format", ar_format)
        delattr(namespace, self.dest) 


class CheckFileType(argparse.FileType):
    def __call__(self, string):
        f = super(CheckFileType, self).__call__(string)
        f.close()
        return FileDesc(f.name, f.mode)


def _map_datasets_formats():
    for label in datasets.ids:
        for ar_format in datasets.get_cls(label).supported_formats():
            yield "{}:{}".format(label, ar_format)


def build_base_parser():
    parser = argparse.ArgumentParser(description="Benzina HEIF Concatenation",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("_action", metavar="action",
                        choices=list(ACTIONS.keys()), help="action to execute")
    parser.add_argument('args', nargs=argparse.REMAINDER,
                        help="action's arguments")

    return parser


def build_concat_parser():
    parser = argparse.ArgumentParser(description="Benzina HEIF Concatenation "
                                                 "action: concat",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("_action", metavar="concat", help="action to execute")
    parser.add_argument("src", metavar="source",
                        help="directory containing a subdirectory 'queue' of "
                             "files to concatenate")
    parser.add_argument("dest", metavar="destination",
                        help="concatenated file")

    return parser


def build_transcode_parser():
    parser = argparse.ArgumentParser(description="Benzina HEIF Concatenation "
                                                 "action: transcode",
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
    parser = argparse.ArgumentParser(description="Benzina HEIF Concatenation "
                                                 "action: extract",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("_action", metavar="extract", help="action to execute")
    parser.add_argument("src", metavar="source",
                        help="archive file to extract the images from")
    parser.add_argument("dest", metavar="destination",
                        help="directory to write the extracted file(s)")
    parser.add_argument("_dataset", choices=list(_map_datasets_formats()),
                        action=DatasetAction, help="dataset id and format")
    parser.add_argument("--start", metavar="IDX", default=0, type=int,
                        help="the start element index to transcode from "
                             "source")
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


def _run_action(_action=None, **kwargs):
    return ACTIONS.get(_action, None)(**kwargs)


def main(args=None, argv=None):
    if args is None:
        args, argv = parse_args()
    else:
        try:
            args, argv = args
        except TypeError:
            pass

    result_arr = _run_action(**vars(args))

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
        result_arr = [_run_action(**_args) for _args in args]
    
    return result_arr


ACTIONS = {"concat": concat, "extract": extract, "transcode": transcode}
ACTIONS_PARSER = {"concat": build_concat_parser(),
                  "extract": build_extract_parser(),
                  "transcode": build_transcode_parser(),
                  "_": build_base_parser()}
