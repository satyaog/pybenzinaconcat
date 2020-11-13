import os

from bitstring import ConstBitStream
from jug import TaskGenerator
import numpy as np

from pybenzinaparse import Parser
from pybenzinaparse.utils import find_boxes, find_traks

from pybenzinaconcat.utils import fnutils
from pybenzinaconcat.datasets import Dataset


def _get_chunk_offset(f, co_box):
    f.seek(co_box.header.start_pos + co_box.header.header_size +
           4)  # entry_count: uint32
    co_buf = f.read(co_box.header.box_size -
                    co_box.header.header_size -
                    4)  # entry_count: uint32
    co = np.frombuffer(co_buf,
                       np.dtype(">u4") if co_box.header.box_type == b"stco"
                       else np.dtype(">u8"))
    return co, co_buf


def _get_sample_size(f, sz_box):
    if sz_box.sample_size > 0:
        f.seek(sz_box.header.start_pos + sz_box.header.header_size)
        sz_buf = f.read(4)
    else:
        f.seek(sz_box.header.start_pos + sz_box.header.header_size +
               4 +  # sample_size: uint32
               4)  # sample_count: uint32
        sz_buf = f.read(sz_box.header.box_size - sz_box.header.header_size -
                        4 -  # sample_size: uint32
                        4)   # sample_count: uint32
    sz = np.frombuffer(sz_buf, np.dtype(">u4"))
    return sz, sz_buf


class Benzina(Dataset):
    SUPPORTED_FORMATS = ("mp4", "bzna")

    def __init__(self, src, ar_format="mp4"):
        super().__init__(src, ar_format)

        bstr = ConstBitStream(filename=self._src)
        moov = next(find_boxes(Parser.parse(bstr, recursive=False), [b"moov"]))
        moov.parse_boxes(bstr, recursive=False)
        for trak in find_boxes(moov.boxes, [b"trak"]):
            trak.parse_boxes(bstr, recursive=False)
            mdia = next(find_boxes(trak.boxes, [b"mdia"]))
            mdia.parse_boxes(bstr, recursive=False)

        trak = next(find_traks(moov.boxes, [b"bzna_input\0"]))
        mdia = next(find_boxes(trak.boxes, [b"mdia"]))
        minf = next(find_boxes(mdia.boxes, [b"minf"]))
        minf.parse_boxes(bstr, recursive=False)
        stbl = next(find_boxes(minf.boxes, [b"stbl"]))
        stbl.parse_boxes(bstr, recursive=False)
        co = next(find_boxes(stbl.boxes, {b"stco", b"co64"}))
        sz = next(find_boxes(stbl.boxes, [b"stsz"]))

        self._input_co = co
        self._input_sz = sz

        trak = next(find_traks(moov.boxes, [b"bzna_fname\0"]))
        mdia = next(find_boxes(trak.boxes, [b"mdia"]))
        minf = next(find_boxes(mdia.boxes, [b"minf"]))
        minf.parse_boxes(bstr, recursive=False)
        stbl = next(find_boxes(minf.boxes, [b"stbl"]))
        stbl.parse_boxes(bstr, recursive=False)
        co = next(find_boxes(stbl.boxes, {b"stco", b"co64"}))
        sz = next(find_boxes(stbl.boxes, [b"stsz"]))

        self._fname_co = co
        self._fname_sz = sz

        self._size = max(self._input_co.entry_count,
                         self._input_sz.sample_count)

    @property
    def size(self):
        return self._size

    def get_input_locations(self, f):
        co, _ = _get_chunk_offset(f, self._input_co)
        sz, _ = _get_sample_size(f, self._input_sz)
        return np.broadcast_arrays(co, sz)

    def get_fname_locations(self, f):
        co, _ = _get_chunk_offset(f, self._fname_co)
        sz, _ = _get_sample_size(f, self._fname_sz)
        return np.broadcast_arrays(co, sz)

    @staticmethod
    @TaskGenerator
    def extract(dataset, dest, start=0, size=512):
        extract(dataset, dest, start, size)


def extract(dataset, dest, start, size):
    """ Take a source mp4/bzna file and extract samples from it into a
    destination directory
    """
    extract_dir = dest

    extracted_filenames = []
    
    with open(dataset.src, "rb") as ds_f:
        start = start
        end = min(start + size, dataset.size) if size else dataset.size

        input_co, input_sz = dataset.get_input_locations(ds_f)
        fname_co, fname_sz = dataset.get_fname_locations(ds_f)

        for i in range(start, end):
            fname_offset, fname_size = fname_co[i], fname_sz[i]
            ds_f.seek(fname_offset)
            filename = ds_f.read(fname_size).rstrip(b'\0').decode("utf-8")
            extract_filepath = os.path.join(extract_dir, filename)
            extract_filepath = fnutils._make_index_filepath(extract_filepath,
                                                            i)
            if not os.path.exists(extract_filepath):
                input_offset, input_size = input_co[i], input_sz[i]
                ds_f.seek(input_offset)
                with open(os.path.join(extract_dir, filename), "wb") as f:
                    f.write(ds_f.read(input_size))
                os.rename(os.path.join(extract_dir, filename),
                          extract_filepath)

            extracted_filenames.append(extract_filepath)

    return extracted_filenames
