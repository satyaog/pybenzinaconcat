import argparse
import os
import sys
from collections import namedtuple

from bitstring import ConstBitStream
import jug
from jug import TaskGenerator

from pybenzinaparse import Parser, boxes as bx_def
from pybenzinaparse.headers import BoxHeader
from pybenzinaparse.utils import get_trak_sample_bytes, find_boxes, \
    find_traks, make_meta_trak, make_vide_trak, make_mvhd


@TaskGenerator
def _init_mdat_size(filename):
    bstr = ConstBitStream(filename=filename)
    mdat = next(find_boxes(Parser.parse(bstr, recursive=False), [b"mdat"]))
    del bstr

    mdat.header.box_ext_size = os.path.getsize(filename) - \
                               mdat.header.start_pos

    # Update mdat header
    with open(filename, "rb+") as container_file:
        container_file.seek(mdat.header.start_pos)
        container_file.write(bytes(mdat.header))

    return mdat.header.start_pos + mdat.header.header_size


@TaskGenerator
def _get_samples_headers(filename, mdat_input_offset):
    samples_headers = []
    with open(filename, "rb") as f:
        f.seek(mdat_input_offset)

        headers = (Parser.parse_header(ConstBitStream(chunk))
                   for chunk in iter(lambda: f.read(32), b''))
        for header in headers:
            # Done reading samples
            if header.type != b"ftyp":
                break
            # Parse FTYP
            samples_headers.append(header)
            f.seek(f.tell() - 32 + header.box_size)
            # Parse MDAT
            header = next(headers)
            samples_headers.append(header)
            f.seek(f.tell() - 32 + header.box_size)
            # Parse MOOV
            header = next(headers)
            samples_headers.append(header)
            f.seek(f.tell() - 32 + header.box_size)

    return samples_headers


@TaskGenerator
def _create_moov_file(filename, samples_headers):
    moov_filename = "{}.moov".format(filename)

    creation_time = 0
    modification_time = 0

    moov = bx_def.MOOV(BoxHeader())
    moov.header.type = b"moov"

    # MOOV.MVHD
    mvhd = make_mvhd(creation_time, modification_time,
                     len([header for header in samples_headers
                          if header.type == b"ftyp"]))
    mvhd.next_track_id = 1

    moov.append(mvhd)
    moov.refresh_box_size()

    with open(moov_filename, "wb") as moov_file:
        moov_file.write(bytes(moov))

    return moov_filename


def _load_moov(filename):
    moov_bstr = ConstBitStream(filename=filename)
    moov = next(Parser.parse(moov_bstr))
    moov.load(moov_bstr)
    return moov


@TaskGenerator
def _index_bzna_input(filename, moov_filename, mdat_input_offset):
    moov = _load_moov(moov_filename)
    mvhd = next(find_boxes(moov.boxes, [b"mvhd"]))

    samples_headers = _get_samples_headers(filename, mdat_input_offset)
    samples_headers = jug.value(samples_headers)

    # bzna_input trak
    if next(find_traks(moov.boxes, [b"bzna_input\0"]), None) is not None:
        trak = next(find_traks(moov.boxes, [b"bzna_input\0"]))
        moov.boxes = [box for box in moov.boxes if box is not trak]

    samples_size = 0
    sample_size = -1
    sizes = []

    for sample_header in samples_headers:
        # Every sample starts with a ftyp box
        if sample_header.type == b"ftyp":
            if sample_size >= 0:
                sizes.append(sample_size)
                samples_size += sample_size
            sample_size = 0

        sample_size += sample_header.box_size

    sizes.append(sample_size)
    samples_size += sample_size

    # MOOV.TRAK
    trak = _make_bzna_input_trak(sizes, mdat_input_offset, mvhd.next_track_id)
    moov.append(trak)
    mvhd.next_track_id += 1
    moov.refresh_box_size()

    with open(moov_filename, "wb") as moov_file:
        moov_file.write(bytes(moov))

    return mdat_input_offset + samples_size


@TaskGenerator
def _append_index_bzna_target(filename, moov_filename, mdat_targets_offset):
    moov = _load_moov(moov_filename)
    mvhd = next(find_boxes(moov.boxes, [b"mvhd"]))

    samples_trak = next(find_traks(moov.boxes, [b"bzna_input\0"]))
    # TRAK.MDIA.MINF.STBL
    stbl = samples_trak.boxes[-1].boxes[-1].boxes[-1]
    samples_offsets = next(find_boxes(stbl.boxes, [b"stco", b"co64"]))
    samples_sizes = next(find_boxes(stbl.boxes, [b"stsz"]))

    # bzna_target trak
    if next(find_traks(moov.boxes, [b"bzna_target\0"]), None) is not None:
        trak = next(find_traks(moov.boxes, [b"bzna_target\0"]))
        moov.boxes = [box for box in moov.boxes if box is not trak]

    targets_size = 0
    targets = []
    sizes = []

    with open(filename, "rb") as container_file:
        for offset, size in zip((o.chunk_offset for o in samples_offsets.entries),
                                (s.entry_size for s in samples_sizes.samples)):
            container_file.seek(offset)
            sample_bstr = ConstBitStream(container_file.read(size))
            # Create a new tmp object to hold the content
            sample_moov = next(find_boxes(Parser.parse(sample_bstr),
                                          [b"moov"]))
            sample_moov.load(sample_bstr)

            target = get_trak_sample_bytes(sample_bstr, sample_moov.boxes,
                                           b"bzna_target\0", 0)
            # Test subset is reached meaning that remaining entries will
            # not contain a target
            if target is None:
                break

            targets.append(target)
            sizes.append(len(target))
            targets_size += len(target)

    # MOOV.TRAK
    trak = _make_bzna_target_trak(sizes, mdat_targets_offset,
                                  mvhd.next_track_id)
    moov.append(trak)
    mvhd.next_track_id += 1
    moov.refresh_box_size()

    with open(filename, "rb+") as container_file, \
         open(moov_filename, "wb") as moov_file:
        moov_file.write(bytes(moov))
        container_file.seek(mdat_targets_offset)
        container_file.write(b''.join(targets))

    return mdat_targets_offset + targets_size


@TaskGenerator
def _append_index_bzna_fname(filename, moov_filename, mdat_fnames_offset):
    moov = _load_moov(moov_filename)
    mvhd = next(find_boxes(moov.boxes, [b"mvhd"]))

    samples_trak = next(find_traks(moov.boxes, [b"bzna_input\0"]))
    # TRAK.MDIA.MINF.STBL
    stbl = samples_trak.boxes[-1].boxes[-1].boxes[-1]
    samples_offsets = next(find_boxes(stbl.boxes, [b"stco", b"co64"]))
    samples_sizes = next(find_boxes(stbl.boxes, [b"stsz"]))

    # bzna_target trak
    if next(find_traks(moov.boxes, [b"bzna_fname\0"]), None) is not None:
        trak = next(find_traks(moov.boxes, [b"bzna_fname\0"]))
        moov.boxes = [box for box in moov.boxes if box is not trak]

    fnames_size = 0
    filenames = []
    sizes = []

    with open(filename, "rb") as container_file:
        for offset, size in zip((o.chunk_offset for o in samples_offsets.entries),
                                (s.entry_size for s in samples_sizes.samples)):
            container_file.seek(offset)
            sample_bstr = ConstBitStream(container_file.read(size))
            # Create a new tmp object to hold the content
            sample_moov = next(find_boxes(Parser.parse(sample_bstr), [b"moov"]))
            sample_moov.load(sample_bstr)

            fn = get_trak_sample_bytes(sample_bstr, sample_moov.boxes,
                                       b"bzna_fname\0", 0)

            filenames.append(fn)
            sizes.append(len(fn))
            fnames_size += len(fn)

    # MOOV.TRAK
    trak = _make_bzna_fname_trak(sizes, mdat_fnames_offset, mvhd.next_track_id)
    moov.append(trak)
    mvhd.next_track_id += 1
    moov.refresh_box_size()

    with open(filename, "rb+") as container_file, \
         open(moov_filename, "wb") as moov_file:
        moov_file.write(bytes(moov))
        container_file.seek(mdat_fnames_offset)
        container_file.write(b''.join(filenames))

    return mdat_fnames_offset + fnames_size


@TaskGenerator
def _index_bzna_thumb(filename, moov_filename, mdat_data_end):
    moov = _load_moov(moov_filename)
    mvhd = next(find_boxes(moov.boxes, [b"mvhd"]))

    samples_trak = next(find_traks(moov.boxes, [b"bzna_input\0"]))
    # TRAK.MDIA.MINF.STBL
    stbl = samples_trak.boxes[-1].boxes[-1].boxes[-1]
    samples_offsets = next(find_boxes(stbl.boxes, [b"stco", b"co64"]))
    samples_sizes = next(find_boxes(stbl.boxes, [b"stsz"]))

    # bzna_target trak
    if next(find_traks(moov.boxes, [b"bzna_thumb\0"]), None) is not None:
        trak = next(find_traks(moov.boxes, [b"bzna_thumb\0"]))
        moov.boxes = [box for box in moov.boxes if box is not trak]

    hvc1 = None
    offsets = []
    sizes = []

    with open(filename, "rb") as container_file:
        for offset, size in zip((o.chunk_offset for o in samples_offsets.entries),
                                (s.entry_size for s in samples_sizes.samples)):
            container_file.seek(offset)
            sample_bstr = ConstBitStream(container_file.read(size))
            # Create a new tmp object to hold the content
            sample_moov = next(find_boxes(Parser.parse(sample_bstr),
                                          [b"moov"]))
            sample_moov.load(sample_bstr)

            # MOOV.TRAK
            trak = next(find_traks(sample_moov.boxes, [b"bzna_thumb\0"]))
            # MOOV.TRAK.MDIA.MINF.STBL
            stbl = trak.boxes[-1].boxes[-1].boxes[-1]
            co = next(find_boxes(stbl.boxes, [b"stco", b"co64"]))
            stsz = next(find_boxes(stbl.boxes, [b"stsz"]))

            if hvc1 is None:
                # MOOV.TRAK.MDIA.MINF.STBL.STSD._VC1
                hvc1 = stbl.boxes[0].boxes[-1]

            offsets.append(offset + co.entries[0].chunk_offset)
            sizes.append(stsz.sample_size)

    # MOOV.TRAK
    trak = _make_bzna_thumb_trak(hvc1, sizes, offsets, mvhd.next_track_id)
    moov.append(trak)
    mvhd.next_track_id += 1
    moov.refresh_box_size()

    with open(moov_filename, "wb") as moov_file:
        moov_file.write(bytes(moov))

    return mdat_data_end


@TaskGenerator
def _update_mdat_size(filename, mdat_data_end):
    bstr = ConstBitStream(filename=filename)
    mdat = next(find_boxes(Parser.parse(bstr, recursive=False), [b"mdat"]))
    del bstr

    mdat.header.box_ext_size = mdat_data_end - mdat.header.start_pos

    # Update mdat header
    with open(filename, "rb+") as container_file:
        container_file.seek(mdat.header.start_pos)
        container_file.write(bytes(mdat.header))

    return mdat_data_end


@TaskGenerator
def _append_moov(filename, moov_filename, mdat_data_end):
    moov = _load_moov(moov_filename)

    with open(filename, "rb+") as container_file:
        container_file.seek(mdat_data_end)
        container_file.write(bytes(moov))

    os.remove(moov_filename)


def _make_bzna_input_trak(samples_sizes, samples_offset, track_id):
    creation_time = 0
    modification_time = 0

    # MOOV.TRAK
    trak = make_meta_trak(creation_time, modification_time, b"bzna_input\0",
                          samples_sizes, samples_offset)

    # MOOV.TRAK.TKHD
    tkhd = trak.boxes[0]

    # "\x00\x00\x01" trak is enabled
    # "\x00\x00\x02" trak is used in the presentation
    # "\x00\x00\x04" trak is used in the preview
    # "\x00\x00\x08" trak size in not in pixel but in aspect ratio
    tkhd.header.flags = b"\x00\x00\x00"
    tkhd.track_id = track_id
    tkhd.width = [0, 0]
    tkhd.height = [0, 0]

    # MOOV.TRAK.MDIA.MINF.STBL.STSD.METT
    mett = trak.boxes[-1].boxes[-1].boxes[-1].boxes[0].boxes[0]
    mett.mime_format = b'video/mp4\0'

    return trak


def _make_bzna_target_trak(samples_sizes, samples_offset, track_id):
    creation_time = 0
    modification_time = 0

    # MOOV.TRAK
    trak = make_meta_trak(creation_time, modification_time, b"bzna_target\0",
                          samples_sizes, samples_offset)

    # MOOV.TRAK.TKHD
    tkhd = trak.boxes[0]

    # "\x00\x00\x01" trak is enabled
    # "\x00\x00\x02" trak is used in the presentation
    # "\x00\x00\x04" trak is used in the preview
    # "\x00\x00\x08" trak size in not in pixel but in aspect ratio
    tkhd.header.flags = b"\x00\x00\x00"
    tkhd.track_id = track_id
    tkhd.width = [0, 0]
    tkhd.height = [0, 0]

    # MOOV.TRAK.MDIA.MINF.STBL.STSD.METT
    mett = trak.boxes[-1].boxes[-1].boxes[-1].boxes[0].boxes[0]
    mett.mime_format = b'application/octet-stream\0'

    return trak


def _make_bzna_fname_trak(samples_sizes, samples_offset, track_id):
    creation_time = 0
    modification_time = 0

    # MOOV.TRAK
    trak = make_meta_trak(creation_time, modification_time, b"bzna_fname\0",
                          samples_sizes, samples_offset)

    # MOOV.TRAK.TKHD
    tkhd = trak.boxes[0]

    # "\x00\x00\x01" trak is enabled
    # "\x00\x00\x02" trak is used in the presentation
    # "\x00\x00\x04" trak is used in the preview
    # "\x00\x00\x08" trak size in not in pixel but in aspect ratio
    tkhd.header.flags = b"\x00\x00\x00"
    tkhd.track_id = track_id
    tkhd.width = [0, 0]
    tkhd.height = [0, 0]

    # MOOV.TRAK.MDIA.MINF.STBL.STSD.METT
    mett = trak.boxes[-1].boxes[-1].boxes[-1].boxes[0].boxes[0]
    mett.mime_format = b'text/plain\0'

    return trak


def _make_bzna_thumb_trak(hvc1, samples_sizes, samples_offsets, track_id):
    creation_time = 0
    modification_time = 0

    # MOOV.TRAK
    trak = make_vide_trak(creation_time, modification_time, b"bzna_thumb\0",
                          samples_sizes, samples_offsets)

    # MOOV.TRAK.TKHD
    tkhd = trak.boxes[0]

    # "\x00\x00\x01" trak is enabled
    # "\x00\x00\x02" trak is used in the presentation
    # "\x00\x00\x04" trak is used in the preview
    # "\x00\x00\x08" trak size in not in pixel but in aspect ratio
    tkhd.header.flags = b"\x00\x00\x03"
    tkhd.track_id = track_id
    tkhd.width = [512, 512]
    tkhd.height = [512, 512]

    # MOOV.TRAK.MDIA.MINF.STBL.STSD
    stsd = trak.boxes[-1].boxes[-1].boxes[-1].boxes[0]
    # pop _VC1
    stsd.pop()

    hvc1_bstr = ConstBitStream(bytes(hvc1))
    hvc1 = next(Parser.parse(hvc1_bstr))
    hvc1.load(hvc1_bstr)

    # pop CLAP and PASP
    hvc1.pop()
    hvc1.pop()

    stsd.append(hvc1)

    return trak


def index_metadata(container):
    if isinstance(container, str):
        container_filename = container
    else:
        container_filename = container.name

    # Init mdat size
    mdat_input_offset = _init_mdat_size(container_filename)
    if not mdat_input_offset.can_load() and mdat_input_offset.can_run():
        mdat_input_offset.run()
    mdat_input_offset = jug.value(mdat_input_offset)

    # Parse samples mp4 boxes headers  b'4c04f56f18ab3a4cd039c28535df69471abab913'
    samples_headers = _get_samples_headers(container_filename,
                                           mdat_input_offset)

    # Create external moov file to avoid corruptions in the container
    moov_filename = _create_moov_file(container_filename, samples_headers)

    mdat_input_end = _index_bzna_input(container_filename, moov_filename,
                                       mdat_input_offset)
    mdat_target_end = _append_index_bzna_target(container_filename,
                                                moov_filename, mdat_input_end)
    mdat_fname_end = _append_index_bzna_fname(container_filename,
                                              moov_filename, mdat_target_end)
    mdat_data_end = _index_bzna_thumb(container_filename, moov_filename,
                                      mdat_fname_end)
    mdat_data_end = _update_mdat_size(container_filename, mdat_data_end)
    return _append_moov(container_filename, moov_filename, mdat_data_end)


FileDesc = namedtuple("FileDesc", ["name", "mode"])


class CheckFileType(argparse.FileType):
    def __call__(self, string):
        f = super(CheckFileType, self).__call__(string)
        f.close()
        return FileDesc(f.name, f.mode)


def build_parser():
    parser = argparse.ArgumentParser(description="Benzina Index Metadata",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("container", type=CheckFileType('rb'),
                        help="the container file to index the metadata")
    return parser


def parse_args(raw_arguments=None):
    argv = sys.argv[1:] if raw_arguments is None else raw_arguments
    args = build_parser().parse_args(argv)
    return args


def main(args=None):
    if args is None:
        args = parse_args()
    return index_metadata(**vars(args))
