import argparse, copy, ctypes, glob, importlib.util, logging, os, subprocess, \
    sys, tarfile
from multiprocessing import Pool

LOGGER = logging.getLogger(os.path.basename(__file__))
LOGGER.setLevel(logging.INFO)

STREAM_HANDLER = logging.StreamHandler()
STREAM_HANDLER.setLevel(logging.INFO)
FORMATTER = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
STREAM_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(STREAM_HANDLER)

h5py_spec = importlib.util.find_spec("h5py")
is_h5py_accessible = h5py_spec is not None
if is_h5py_accessible:
    import h5py
    import numpy as np

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


def _get_completed_list(dest_dir, ssh_remote):
    completed_list_filepath = os.path.join(dest_dir, "completed_list")
    process = subprocess.Popen(["rsync", "-v",
                                _get_remote_path(ssh_remote,
                                                 completed_list_filepath),
                                '.'])
    process.wait()

    if os.path.exists("completed_list"):
        with open("completed_list", 'r') as file:
            content = file.read()

        completed_list = []
        for filepath in content.split('\n'):
            clean_basename = _get_clean_filepath(filepath, basename=True)
            completed_list.append(clean_basename)

        return completed_list
    else:
        return []


def concat(args):
    """ Takes a source directory containing files and append/concatenate them
    into a single destination file

    Files contained in the subdirectory 'queue' of the source directory will be
    concatenated

    A 'completed_list' containing the concatenated file will be created in the
    source directory

    Files with a base name that is contained in the 'completed_list' file of
    the source directory will be ignored

    If they don't exist, the subdirectories 'upload' and 'queue' of the source
    directory will be created

    :param args: parsed arguments
    """
    src_dir = args.src
    dest_dir = os.path.dirname(args.dest)
    queue_dir = os.path.join(src_dir, "queue/")
    queued_files = glob.glob(os.path.join(queue_dir, '*'))
    queued_files.sort()

    # Setup directories hierarchy that will be used by "transcode". Needed in
    # remote situation, where it simplifies the creation of the hierarchy
    upload_dir = os.path.join(src_dir, "upload/")
    if upload_dir and not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    if queue_dir and not os.path.exists(queue_dir):
        os.makedirs(queue_dir)

    if dest_dir and not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    completed_list = _get_completed_list(src_dir, None)

    with open(args.dest, "ab") as concat_file, \
         open(os.path.join(src_dir, "completed_list"), "a") \
         as completed_list_file:
        if not len(queued_files):
            LOGGER.warning("No queued files in [{}] to append to [{}]"
                           .format(queue_dir, args.dest))
        for queued_filepath in queued_files:
            clean_basename = _get_clean_filepath(queued_filepath, basename=True)
            if clean_basename in completed_list:
                LOGGER.warning("Ignoring [{}] since [{}] is in [{}]"
                               .format(queued_filepath, clean_basename,
                                       os.path.join(src_dir, "completed_list")))
                continue
            with open(queued_filepath, "rb") as queued_file:
                concat_file.write(queued_file.read())

            completed_list_file.write(queued_filepath + '\n')
            os.remove(queued_filepath)


def transcode_img(input_path, dest_dir, args):
    upload_dir = os.path.join(dest_dir, "upload/")
    queue_dir = os.path.join(dest_dir, "queue/")
    tmp_dir = args.tmp if args.tmp is not None else \
              os.path.dirname(input_path)
    filename = os.path.basename(input_path)
    target_path = _make_target_filepath(input_path)

    if tmp_dir and not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    output_path = _make_transcoded_filepath(os.path.join(tmp_dir, filename))
    command = ["python", "image2mp4"] if args.mp4 else ["image2heif"]
    cmd_arguments = " --codec=h265 --tile=512:512:yuv420 --crf=10 " \
                    "--output={dest} " \
                    "--primary --thumb --name={name} " \
                    "--item=path={src}" \
                    .format(name=os.path.basename(input_path),
                            src=input_path, dest=output_path)
    if os.path.exists(target_path):
        cmd_arguments += " --hidden --name=target " \
                         "--mime=application/octet-stream " \
                         "--item=type=mime,path={target}" \
                         .format(target=target_path)

    process = subprocess.Popen(command +
                               ["--" + arg for arg in cmd_arguments.split(" --")[1:]])
    process.wait()

    if process.wait() != 0:
        LOGGER.error("Could transcode file [{}] with target [{}] to [{}]"
                     .format(input_path, target_path, output_path))
        return

    uploaded_path = os.path.join(upload_dir, os.path.basename(output_path))
    process = subprocess.Popen(["rsync", "-v", "--remove-source-files", output_path,
                                _get_remote_path(args.ssh_remote,
                                                 upload_dir)])

    if process.wait() != 0:
        LOGGER.error("Could not move file [{}] to upload dir [{}]"
                     .format(output_path, upload_dir))
        return

    queued_path = os.path.join(queue_dir, os.path.basename(output_path))
    if args.ssh_remote:
        process = subprocess.Popen(["ssh", args.ssh_remote,
                                    "mv -v {} {}".format(uploaded_path, queued_path)])
        if process.wait() != 0:
            LOGGER.error("Could not move file [{}] to queue dir [{}]"
                         .format(uploaded_path, queued_path))
    else:
        os.rename(uploaded_path, queued_path)


def transcode(args):
    """ Takes a list of images and transcodes them into a destination directory

    The suffix ".transcoded" will be appended to the file a base name

    Subdirectories 'upload' and 'queue' will be created in destination
    directory where 'upload' contains the files that are being uploaded and
    'queue' contains the files which are ready to be concatenated

    Files with a base name that is contained in the 'completed_list' file of
    the destination directory will be ignored

    :param args: parsed arguments
    """
    dest_dir = args.dest

    completed_list = _get_completed_list(dest_dir, args.ssh_remote)

    for input_path in args.src.split(','):
        clean_basename = _get_clean_filepath(input_path, basename=True)
        if clean_basename in completed_list:
            LOGGER.info("Ignoring [{}] since [{}] is in [{}]/completed_list"
                        .format(input_path, clean_basename, dest_dir))
            continue
        transcode_img(input_path, dest_dir, args)


def extract_hdf5(args):
    """ Takes a source HDF5 file and extracts images from it into a destination
    directory

    :param args: parsed arguments
    """
    tmp_dir = args.tmp
    extract_dir = tmp_dir if tmp_dir is not None else args.dest

    extracted_filenames = []

    with h5py.File(args.src, "r") as file_h5:
        num_elements = len(file_h5["encoded_images"])

        start = args.start
        end = min(args.start + args.number, num_elements) \
              if args.number else num_elements

        if tmp_dir and not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        if extract_dir and not os.path.exists(extract_dir):
            os.makedirs(extract_dir)

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

            if not os.path.exists(target_filepath):
                target = file_h5["targets"][i].astype(np.int64).tobytes()

                with open(target_filepath, "xb") as file:
                    file.write(target)

    return extracted_filenames


def extract_tar(args):
    """ Takes a source tar file and extracts images from it into a destination
    directory

    :param args: parsed arguments
    """
    tmp_dir = args.tmp
    extract_dir = tmp_dir if tmp_dir is not None else args.dest

    extracted_filenames = []

    index = 0
    start = args.start
    end = args.start + args.number if args.number else \
                                   ctypes.c_ulonglong(-1).value

    with tarfile.open(args.src, "r") as file_tar:
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


def single_process_extract_archive(args):
    """ Takes a source archive file and extracts images from it into a destination
    directory. If the --transcode parameter is set, images will also be
    transcoded

    args.jobs is ignored in this function

    :param args: parsed arguments
    """
    args.jobs = None

    if not args.transcode:
        args.ssh_remote = None
        args.tmp = None

    if args.type == "hdf5":
        extracted_filenames = extract_hdf5(args)
    else:
        extracted_filenames = extract_tar(args)

    if args.transcode:
        transcode_args = build_transcode_parser() \
            .parse_args(["transcode",
                         ','.join(extracted_filenames),
                         args.dest,
                         "--ssh-remote", args.ssh_remote,
                         "--tmp", args.tmp] +
                        (["--mp4"] if args.mp4 else []))

        transcode(transcode_args)


def extract_archive(args):
    """ Takes a source archive file and extracts images from it into a destination
    directory. If the --transcode parameter is set, images will also be
    transcoded

    This will split the process to use all cores available

    :param args: parsed arguments
    """
    if not args.transcode:
        args.ssh_remote = None
        args.tmp = None

    if args.jobs == 0:
        args.jobs = os.cpu_count()

    if args.number:
        processes_args = []
        split_number = args.number / args.jobs
        start = args.start
        for process_i in range(args.jobs):
            next_start = start + split_number

            process_args = copy.deepcopy(args)
            process_args.jobs = None
            process_args.start = int(round(start))
            process_args.number = args.start + args.number - process_args.start \
                                  if process_i == args.jobs - 1 \
                                  else int(round(next_start)) - int(round(start))
            processes_args.append(process_args)

            start = next_start
    else:
        processes_args = [args]

    # Minimize async issues when trying to create the same directory multiple
    # times and at the same time
    tmp_dir = args.tmp
    extract_dir = tmp_dir if tmp_dir is not None else args.dest
    if extract_dir and not os.path.exists(extract_dir):
        os.makedirs(extract_dir)

    with Pool(args.jobs) as pool:
        pool.map(single_process_extract_archive, processes_args)


def build_base_parser():
    parser = argparse.ArgumentParser(description="Benzina HEIF Concatenation",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("action", choices=list(ACTIONS.keys()),
                        help="action to execute")
    parser.add_argument('_', metavar="...", nargs=argparse.REMAINDER,
                        help="use -h {} to view the action's arguments"
                             .format('{'+','.join(ACTIONS.keys())+'}'))

    return parser


def build_concat_parser():
    parser = argparse.ArgumentParser(description="Benzina HEIF Concatenation action: "
                                                 "\"concat\"",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("action", metavar="\"concat\"",
                        help="action to execute")
    parser.add_argument("src", metavar="source",
                        help="the source directory containing a 'queue' "
                             "directory of files to concatenate")
    parser.add_argument("dest", metavar="destination",
                        help="the destination concatenated file")

    return parser


def build_transcode_parser():
    parser = argparse.ArgumentParser(description="Benzina HEIF Concatenation action: "
                                                 "\"transcode\"",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("action", metavar="\"transcode\"",
                        help="action to execute")
    parser.add_argument("src", metavar="source",
                        help="the source file or ',' separated files to transcode")
    parser.add_argument("dest", metavar="destination",
                        help="the destination directory for the transcoded file(s)")
    parser.add_argument("--mp4", default=False, action="store_true",
                        help="use image2mp4 instead of image2heif")
    parser.add_argument("--ssh-remote", metavar="REMOTE",
                        help="optional remote to use to transfer the transcoded "
                             "file to destination")
    parser.add_argument("--tmp", metavar="DIR",
                        help="the directory to use to store temporary file(s)")

    return parser


def build_extract_archive_parser():
    parser = argparse.ArgumentParser(description="Benzina HEIF Concatenation action: "
                                                 "\"extract_archive\"",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("action", metavar="\"extract_archive\"",
                        help="action to execute")
    parser.add_argument("type", choices=["hdf5", "tar"] if is_h5py_accessible else
                                        ["tar"],
                        help="type of the archive")
    parser.add_argument("src", metavar="source",
                        help="the archive file to draw the images to extract and "
                             "transcode, if --transcode is True")
    parser.add_argument("dest", metavar="destination",
                        help="the destination directory of extracted "
                             "file(s) or transcoded file(s) if --transcode "
                             "is True")
    parser.add_argument("--start", default=0, type=int,
                        help="the start element index to transcode from source")
    parser.add_argument("--number", default=1000, metavar="NUM", type=int,
                        help="the number of elements to extract from source")
    parser.add_argument("--jobs", default=0, metavar="NUM", type=int,
                        help="the number of workers to work in parallel. Use '0' "
                             "to use as much workers as possible. Note that number "
                             "needs to be specified")

    parser.add_argument("--transcode", default=False, action="store_true",
                        help="follow the extraction with a transcoding")
    parser.add_argument("--mp4", default=False, action="store_true",
                        help="use image2mp4 instead of image2heif")
    parser.add_argument("--ssh-remote", metavar="REMOTE",
                        help="if --transcode is True, optional remote to "
                             "use to transfer the transcoded file to "
                             "destination")
    parser.add_argument("--tmp", metavar="DIR",
                        help="if --transcode is True, the directory to "
                             "store temporary file(s)")

    return parser


def parse_args(raw_arguments=None):
    argv = sys.argv[1:] if raw_arguments is None else raw_arguments
    is_help_request = "-h" in argv

    if is_help_request:
        if len(argv) == 1:
            build_base_parser().parse_args(argv)
        argv.remove("-h")
        base_args = build_base_parser().parse_args(argv)
        ACTIONS_PARSER.get(base_args.action, None).parse_args(argv + ["-h"])

    base_args = build_base_parser().parse_args(argv)
    return ACTIONS_PARSER.get(base_args.action, None).parse_args(argv)


def main(args):
    ACTIONS.get(args.action, None)(args)


ACTIONS = {"transcode": transcode, "concat": concat, "extract_archive": extract_archive}
ACTIONS_PARSER = {"concat": build_concat_parser(),
                  "transcode": build_transcode_parser(),
                  "extract_archive": build_extract_archive_parser(),
                  "_": build_base_parser()}


if __name__ == "__main__":
    main(parse_args())
