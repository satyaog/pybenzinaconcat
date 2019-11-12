import os
import shutil

from bitstring import ConstBitStream

from pybzparse import Parser, boxes as bx_def
from pybzparse.headers import BoxHeader

from pyheifconcat.image2mp4 import image2mp4, parse_args

from pyheifconcat.image2mp4.image2mp4 import clap_traks, _clean_boxes, \
    make_filenames_trak, make_targets_trak, reset_traks_id


PWD = "tests_tmp"

if PWD and not os.path.exists(PWD):
    os.makedirs(PWD)

os.chdir(PWD)


def test__clean_boxes():
    src = "../test_datasets/transcode_steps/n02100735_8211.mp4"

    bstr = ConstBitStream(filename=src)
    boxes = [box for box in Parser.parse(bstr)]
    for box in boxes:
        box.load(bstr)

    _clean_boxes(boxes)

    assert len(boxes) == 3

    ftyp = boxes[0]
    moov = boxes[2]

    ftyp.refresh_box_size()
    moov.refresh_box_size()

    assert ftyp.major_brand == 1769172845           # b"isom"
    assert ftyp.header.box_size == 20
    assert ftyp.minor_version == 0
    assert ftyp.compatible_brands == [1769172845]   # b"isom"

    assert moov.header.box_size == 5493
    assert len(moov.boxes) == 3

    traks = [box for box in moov.boxes if box.header.type == b"trak"]

    assert len(traks) == 2

    assert len(traks[0].boxes) == 2

    # moov.trak.tkhd
    assert traks[0].boxes[0].header.flags == b"\x00\x00\x00"
    # moov.trak.mdia.hdlr
    assert traks[0].boxes[1].boxes[1].name == b"bzna_input\0"

    # moov.trak.mdia.minf.stbl.stco
    stco = traks[0].boxes[-1].boxes[-1].boxes[-1].boxes[-1]
    assert stco.entries[0].chunk_offset == 28

    # moov.trak.tkhd
    assert traks[1].boxes[0].header.flags == b"\x00\x00\x03"
    # moov.trak.mdia.hdlr
    assert traks[1].boxes[1].boxes[1].name == b"bzna_thumb\0"

    # moov.trak.mdia.minf.stbl.stco
    stco = traks[1].boxes[-1].boxes[-1].boxes[-1].boxes[-1]
    assert stco.entries[0].chunk_offset == 221612


def test_clap_traks():
    src = "../test_datasets/transcode_steps/n02100735_8211.mp4"
    width = 600
    height = 535
    thumb_width = 512
    thumb_height = 456

    bstr = ConstBitStream(filename=src)
    boxes = [box for box in Parser.parse(bstr)]
    for box in boxes:
        box.load(bstr)

    _clean_boxes(boxes)

    moov = boxes[2]

    traks = [box for box in moov.boxes if box.header.type == b"trak"]

    clap_traks(traks, width, height, thumb_width, thumb_height)

    trak_input, trak_thumb = traks

    # moov.trak.tkhd
    tkhd = trak_input.boxes[0]

    assert tkhd.width == 600
    assert tkhd.height == 535

    # moov.trak.mdia.minf.stbl.stsd.hvc1.clap
    clap = trak_input.boxes[-1].boxes[-1].boxes[-1].boxes[0].boxes[0].boxes[-2]

    assert clap.header.type == b"clap"
    assert clap.clean_aperture_width_n == 600
    assert clap.clean_aperture_width_d == 1
    assert clap.clean_aperture_height_n == 535
    assert clap.clean_aperture_height_d == 1
    assert clap.horiz_off_n == -424
    assert clap.horiz_off_d == 2
    assert clap.vert_off_n == -489
    assert clap.vert_off_d == 2

    # moov.trak.tkhd
    tkhd = trak_thumb.boxes[0]

    assert tkhd.width == 512
    assert tkhd.height == 456

    # moov.trak.mdia.minf.stbl.stsd.hvc1.clap
    clap = trak_thumb.boxes[-1].boxes[-1].boxes[-1].boxes[0].boxes[0].boxes[-2]

    assert clap.header.type == b"clap"
    assert clap.clean_aperture_width_n == 512
    assert clap.clean_aperture_width_d == 1
    assert clap.clean_aperture_height_n == 456
    assert clap.clean_aperture_height_d == 1
    assert clap.horiz_off_n == 0
    assert clap.horiz_off_d == 2
    assert clap.vert_off_n == -56
    assert clap.vert_off_d == 2


def test_make_filenames_trak():
    # MDAT
    mdat = bx_def.MDAT(BoxHeader())
    mdat.header.type = b"mdat"
    mdat.data = (b"0123456789",)
    mdat.refresh_box_size()

    filename_trak = make_filenames_trak(mdat, 20, [b"0001/n02100735_8211.JPEG"])

    # MOOV.TRAK.TKHD
    assert filename_trak.boxes[0].header.flags == b"\x00\x00\x03"
    assert filename_trak.boxes[0].width == 0
    assert filename_trak.boxes[0].height == 0

    # MOOV.TRAK.MDIA.HDLR
    assert filename_trak.boxes[1].boxes[1].name == b"bzna_fname\0"

    # MOOV.TRAK.MDIA.MINF.STBL.STSD.METT
    mett = filename_trak.boxes[-1].boxes[-1].boxes[-1].boxes[0].boxes[0]
    assert mett.mime_format == b"text/plain\0"

    # MOOV.TRAK.MDIA.MINF.STBL.STSZ
    stsz = filename_trak.boxes[-1].boxes[-1].boxes[-1].boxes[2]
    assert stsz.sample_count == 1
    assert stsz.samples[0].entry_size == 24

    # MOOV.TRAK.MDIA.MINF.STBL.STCO
    stsz = filename_trak.boxes[-1].boxes[-1].boxes[-1].boxes[4]
    assert stsz.entry_count == 1
    assert stsz.entries[0].chunk_offset == 38


def test_make_targets_trak():
    # MDAT
    mdat = bx_def.MDAT(BoxHeader())
    mdat.header.type = b"mdat"
    mdat.data = (b"0123456789",)
    mdat.refresh_box_size()

    target_trak = make_targets_trak(mdat, 20, "application/octet-stream",
                                    [int.to_bytes(100, 8, byteorder="little")])

    # MOOV.TRAK.TKHD
    assert target_trak.boxes[0].header.flags == b"\x00\x00\x00"
    assert target_trak.boxes[0].width == 0
    assert target_trak.boxes[0].height == 0

    # MOOV.TRAK.MDIA.HDLR
    assert target_trak.boxes[1].boxes[1].name == b"bzna_target\0"

    # MOOV.TRAK.MDIA.MINF.STBL.STSD.METT
    mett = target_trak.boxes[-1].boxes[-1].boxes[-1].boxes[0].boxes[0]
    assert mett.mime_format == b"application/octet-stream\0"

    # MOOV.TRAK.MDIA.MINF.STBL.STSZ
    stsz = target_trak.boxes[-1].boxes[-1].boxes[-1].boxes[2]
    assert stsz.sample_count == 1
    assert stsz.samples[0].entry_size == 8

    # MOOV.TRAK.MDIA.MINF.STBL.STCO
    stsz = target_trak.boxes[-1].boxes[-1].boxes[-1].boxes[4]
    assert stsz.entry_count == 1
    assert stsz.entries[0].chunk_offset == 38


def test_reset_traks_id():
    moov = bx_def.MOOV(BoxHeader())
    mvhd = bx_def.MVHD(BoxHeader())
    moov.append(mvhd)

    for i in range(4):
        trak = bx_def.TRAK(BoxHeader())
        trak.header.type = b"trak"
        tkhd = bx_def.TKHD(BoxHeader())
        tkhd.header.type = b"tkhd"
        trak.append(tkhd)
        moov.append(trak)

    reset_traks_id(moov)

    assert moov.boxes[0].next_track_id == 5

    for i, trak in enumerate([box for box in moov.boxes
                              if box.header.type == b"trak"]):
        assert trak.boxes[0].track_id == i + 1


def test_image2mp4():
    dir = "../test_datasets/mini_dataset_to_transcode/0001"
    name = "n02100735_8211.JPEG"
    out = "n02100735_8211.mp4"

    try:
        args = parse_args(["--codec=h265", "--tile=512:512:yuv420", "--crf=10",
                           "--output={}".format(out),
                           "--primary", "--thumb", "--name={}".format(name),
                           "--item=path={}/{}".format(dir, name),
                           "--hidden", "--name=target", "--mime=application/octet-stream",
                           "--item=type=mime,path={}/{}.target".format(dir, name)])

        image2mp4(args)

        bstr = ConstBitStream(filename=out)
        ftyp, mdat, moov = list(Parser.parse(bstr))
        for box in [ftyp, mdat, moov]:
            box.load(bstr)
        del bstr

        assert ftyp.header.type == b"ftyp"
        assert mdat.header.type == b"mdat"
        assert moov.header.type == b"moov"

        # MOOV.TRAK.MDIA.HDLR
        trak = moov.boxes[1]
        assert trak.boxes[-1].boxes[1].name == b"bzna_input\0"

        trak = moov.boxes[2]
        assert trak.boxes[-1].boxes[1].name == b"bzna_fname\0"

        # MOOV.TRAK.MDIA.MINF.STBL
        stbl = trak.boxes[-1].boxes[-1].boxes[-1]
        # MOOV.TRAK.MDIA.MINF.STBL.STSZ
        size = stbl.boxes[2].samples[0].entry_size
        # MOOV.TRAK.MDIA.MINF.STBL.STCO
        offset = stbl.boxes[4].entries[0].chunk_offset - \
                 mdat.header.start_pos - mdat.header.header_size
        assert mdat.data[offset:offset + size] == bytes(name, "utf-8")

        trak = moov.boxes[3]
        assert trak.boxes[-1].boxes[1].name == b"bzna_target\0"

        # MOOV.TRAK.MDIA.MINF.STBL
        stbl = trak.boxes[-1].boxes[-1].boxes[-1]
        # MOOV.TRAK.MDIA.MINF.STBL.STSZ
        size = stbl.boxes[2].samples[0].entry_size
        # MOOV.TRAK.MDIA.MINF.STBL.STCO
        offset = stbl.boxes[4].entries[0].chunk_offset - \
                 mdat.header.start_pos - mdat.header.header_size
        assert mdat.data[offset:offset + size] == int.to_bytes(1, 8, "little")

        trak = moov.boxes[4]
        assert trak.boxes[-1].boxes[1].name == b"bzna_thumb\0"

    finally:
        shutil.rmtree(".", ignore_errors=True)


def test_parse_args():
    raw_arguments = ["--codec=h265", "--tile=512:512:yuv420", "--crf=10",
                     "--output=out.mp4",
                     "--primary", "--thumb", "--name=n02100735_8211.JPEG",
                     "--item=path=n02100735_8211.JPEG",
                     "--hidden", "--name=target", "--mime=application/octet-stream",
                     "--item=type=mime,path=n02100735_8211.JPEG.target"]

    raw_arguments_no_target = \
        ["--codec=h265", "--tile=512:512:yuv420", "--crf=10",
         "--output=out.mp4",
         "--primary", "--thumb", "--name=n02100735_8211.JPEG",
         "--item=path=n02100735_8211.JPEG"]

    args = parse_args(raw_arguments)

    assert args.codec == "h265"
    assert args.tile.width == 512
    assert args.tile.height == 512
    assert args.tile.pixel_fmt == "yuv420"
    assert args.crf == 10
    assert args.output == "out.mp4"

    assert len(args.items) == 2

    assert args.items[0].primary is True
    assert args.items[0].hidden is False
    assert args.items[0].name == "n02100735_8211.JPEG"
    assert args.items[0].thumb is True
    assert args.items[0].mime is None
    assert args.items[0].item.id is None
    assert args.items[0].item.type is None
    assert args.items[0].item.path == "n02100735_8211.JPEG"

    assert args.items[1].primary is False
    assert args.items[1].hidden is True
    assert args.items[1].name == "target"
    assert args.items[1].thumb is False
    assert args.items[1].mime == "application/octet-stream"
    assert args.items[1].item.id is None
    assert args.items[1].item.type == "mime"
    assert args.items[1].item.path == "n02100735_8211.JPEG.target"

    args = parse_args(raw_arguments_no_target)

    assert args.codec == "h265"
    assert args.tile.width == 512
    assert args.tile.height == 512
    assert args.tile.pixel_fmt == "yuv420"
    assert args.crf == 10
    assert args.output == "out.mp4"

    assert len(args.items) == 1

    assert args.items[0].primary is True
    assert args.items[0].hidden is False
    assert args.items[0].name == "n02100735_8211.JPEG"
    assert args.items[0].thumb is True
    assert args.items[0].mime is None
    assert args.items[0].item.id is None
    assert args.items[0].item.type is None
    assert args.items[0].item.path == "n02100735_8211.JPEG"
