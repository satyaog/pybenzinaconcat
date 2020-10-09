import os

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



