import argparse
import os
import sys

from bitstring import ConstBitStream, ReadError

from pybzparse import Parser, boxes as bx_def
from pybzparse.headers import BoxHeader
from pybzparse.utils import get_trak_sample, find_boxes, find_traks, \
                            make_meta_trak, make_mvhd


def _get_samples_boxes(samples_bstr):
    # Parse samples mp4 boxes
    samples_boxes = []
    sample_parser = Parser.parse(samples_bstr)
    while samples_bstr.bitpos < samples_bstr.length:
        sample_header = Parser.parse_header(samples_bstr)
        # Done reading samples
        if sample_header.type != b"ftyp":
            break
        samples_bstr.bytepos = sample_header.start_pos
        # Parse FTYP
        samples_boxes.append(next(sample_parser))
        # Parse MDAT
        samples_boxes.append(next(sample_parser))
        # Parse MOOV
        sample_moov = next(sample_parser)
        samples_boxes.append(sample_moov)

        for trak in find_boxes(sample_moov.boxes, b"trak"):
            # TRAK.MDIA.MINF.STBL
            stbl = trak.boxes[-1].boxes[-1].boxes[-1]
            next(find_boxes(stbl.boxes, b"stco")).load(samples_bstr)
            next(find_boxes(stbl.boxes, b"stsz")).load(samples_bstr)

        samples_bstr.bytepos = sample_moov.header.start_pos + \
                               sample_moov.header.box_size

    return samples_boxes


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


def index_metadata(args):
    args.container.close()

    container_filename = args.container.name
    moov_filename = "{}.moov".format(container_filename)

    bstr = ConstBitStream(filename=container_filename)
    container_len = int(bstr.length / 8)
    mdat = next(find_boxes(Parser.parse(bstr, recursive=False), b"mdat"))
    mdat_data_offset = mdat.header.start_pos + mdat.header.header_size
    # If the next box is a valid moov box, then the indexing has already been done
    try:
        moov_header = Parser.parse_header(bstr)
        if moov_header.type == b"moov":
            moov = Parser.parse_box(bstr, moov_header)
            moov.load(bstr)
            return
    except ReadError:
        pass
    del bstr

    # Parse samples mp4 boxes
    samples_bstr = ConstBitStream(filename=container_filename, offset=mdat_data_offset * 8)
    samples_boxes = _get_samples_boxes(samples_bstr)
    del samples_bstr

    # If the box content length is not set in the header
    if mdat.header.box_size == mdat.header.header_size:
        mdat.header.box_size = container_len - mdat.header.start_pos

    # Update mdat header
    with open(container_filename, "rb+") as file:
        file.seek(mdat.header.start_pos)
        file.write(bytes(mdat.header))

    # MOOV
    if os.path.exists(moov_filename):
        moov_bstr = ConstBitStream(filename=moov_filename)
        moov = next(Parser.parse(moov_bstr))
        moov.load(moov_bstr)
        del moov_bstr

        mvhd = next(find_boxes(moov.boxes, b"mvhd"))

    else:
        creation_time = 0
        modification_time = 0

        moov = bx_def.MOOV(BoxHeader())
        moov.header.type = b"moov"

        # MOOV.MVHD
        mvhd = make_mvhd(creation_time, modification_time,
                         len([box for box in samples_boxes if box.header.type == b"ftyp"]))
        mvhd.next_track_id = 1

        moov.append(mvhd)
        moov.refresh_box_size()

        with open(moov_filename, "wb") as moov_file:
            moov_file.write(bytes(moov))

    # bzna_input trak
    if next(find_traks(moov.boxes, b"bzna_input\0"), None) is None:
        samples_offset = mdat_data_offset
        sample_size = -1
        samples_sizes = []

        for sample_box in samples_boxes:
            # Every sample starts with a ftyp box
            if sample_box.header.type == b"ftyp":
                if sample_size >= 0:
                    samples_sizes.append(sample_size)
                sample_size = 0

            sample_size += sample_box.header.box_size

        samples_sizes.append(sample_size)

        # MOOV.TRAK
        trak = _make_bzna_input_trak(samples_sizes, samples_offset, mvhd.next_track_id)

        moov.append(trak)
        mvhd.next_track_id += 1

        moov.refresh_box_size()

        with open(moov_filename, "wb") as moov_file:
            moov_file.write(bytes(moov))

    samples_trak = next(find_traks(moov.boxes, b"bzna_input\0"))
    # TRAK.MDIA.MINF.STBL
    stbl = samples_trak.boxes[-1].boxes[-1].boxes[-1]
    samples_offsets = next(find_boxes(stbl.boxes, b"stco"))

    # bzna_target trak
    if next(find_traks(moov.boxes, b"bzna_target\0"), None) is None:
        samples_offset = container_len
        targets = []
        samples_sizes = []

        for sample_moov, sample_offset in zip(find_boxes(samples_boxes, b"moov"),
                                              samples_offsets.entries):
            sample_bstr = ConstBitStream(filename=container_filename,
                                         offset=sample_offset.chunk_offset * 8)
            target = get_trak_sample(sample_bstr, sample_moov.boxes, b"bzna_target\0", 0)
            targets.append(target)
            samples_sizes.append(len(target))

        # MOOV.TRAK
        trak = _make_bzna_target_trak(samples_sizes, samples_offset, mvhd.next_track_id)

        moov.append(trak)
        mvhd.next_track_id += 1

        moov.refresh_box_size()

        with open(moov_filename, "wb") as moov_file, \
             open(container_filename, "ab") as container_file:
            moov_file.write(bytes(moov))
            for target in targets:
                container_file.write(target)
                container_len += len(target)

    # bzna_fname trak
    if next(find_traks(moov.boxes, b"bzna_fname\0"), None) is None:
        samples_offset = container_len
        filenames = []
        samples_sizes = []

        for sample_moov, sample_offset in zip(find_boxes(samples_boxes, b"moov"),
                                              samples_offsets.entries):
            sample_bstr = ConstBitStream(filename=container_filename,
                                         offset=sample_offset.chunk_offset * 8)
            filename = get_trak_sample(sample_bstr, sample_moov.boxes, b"bzna_fname\0", 0)
            filenames.append(filename)
            samples_sizes.append(len(filename))

        # MOOV.TRAK
        trak = _make_bzna_fname_trak(samples_sizes, samples_offset, mvhd.next_track_id)

        moov.append(trak)
        mvhd.next_track_id += 1

        moov.refresh_box_size()

        with open(moov_filename, "wb") as moov_file, \
             open(container_filename, "ab") as container_file:
            moov_file.write(bytes(moov))
            for filename in filenames:
                container_file.write(filename)
                container_len += len(filename)

    mdat.header.box_size = container_len - mdat.header.start_pos

    with open(container_filename, "rb+") as file:
        file.seek(mdat.header.start_pos)
        file.write(bytes(mdat.header))
        file.flush()
        file.seek(0, 2)
        file.write(bytes(moov))

    os.remove(moov_filename)


def build_parser():
    parser = argparse.ArgumentParser(description="Benzina Index Metadata",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("container", type=argparse.FileType('rb'),
                        help="the container file to index the metadata")
    return parser


def parse_args(raw_arguments=None):
    argv = sys.argv[1:] if raw_arguments is None else raw_arguments
    args = build_parser().parse_args(argv)
    return args
