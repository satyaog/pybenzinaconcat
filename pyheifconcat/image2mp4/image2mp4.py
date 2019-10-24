import argparse
import os
import subprocess
import sys

from bitstring import ConstBitStream
from PIL import Image

from pybzparse import Parser, boxes as bx_def
from pybzparse.headers import BoxHeader
from pybzparse.utils import make_meta_trak

CODECS_DICT = {"h264": ["-c:v", "libx264"],
               "h265": ["-c:v", "libx265", "-tag:v", "hvc1"]}
PIXEL_FMTS_DICT = {"yuv420": "yuvj420p"}
TYPES_LIST = ["mime"]


def _clean_boxes(boxes):
    # remove 'free' box
    boxes[:] = boxes[:1] + boxes[2:]

    ftyp = boxes[0]
    mdat = boxes[1]
    moov = boxes[2]

    # remove 'udta' box
    moov.boxes.pop()

    # moov.mvhd
    moov.boxes[0].timescale = 20
    moov.boxes[0].duration = 20

    traks = [box for box in moov.boxes if box.header.type == b"trak"]

    for i, trak in enumerate(traks):
        # remove 'edts' box
        mdia = trak.pop()
        trak.pop()
        trak.append(mdia)

        # moov.trak.mdia.hdlr
        hdlr = mdia.boxes[1]

        if hdlr.handler_type == b"vide":
            # moov.trak.tkhd
            tkhd = trak.boxes[0]

            # "\x00\x00\x01" trak is enabled
            # "\x00\x00\x02" trak is used in the presentation
            # "\x00\x00\x04" trak is used in the preview
            # "\x00\x00\x08" trak size in not in pixel but in aspect ratio
            tkhd.header.flags = b"\x00\x00\x00" if i == 0 else b"\x00\x00\x03"

            hdlr.name = b"bzna_input" if i == 0 else b"bzna_thumb"

    ftyp.major_brand = 1769172845           # b"isom"
    ftyp.minor_version = 0
    ftyp.compatible_brands = [1769172845]   # b"isom"
    ftyp.refresh_box_size()

    chunk_offset_diff = ftyp.header.box_size - mdat.header.start_pos

    for i, trak in enumerate(traks):
        # moov.trak.mdia.minf.stbl
        stbl = trak.boxes[-1].boxes[-1].boxes[-1]

        # moov.trak.mdia.minf.stbl.stco
        stco = stbl.boxes[-1]
        for entry in stco.entries:
            entry.chunk_offset = entry.chunk_offset + chunk_offset_diff


def clap_traks(traks, width, height, thumb_width, thumb_height):
    for i, trak in enumerate(traks):
        # moov.trak.tkhd
        tkhd = trak.boxes[0]

        # moov.trak.mdia.hdlr
        hdlr = trak.boxes[-1].boxes[1]

        if hdlr.handler_type == b"vide":
            clap_width = width if i == 0 else thumb_width
            clap_height = height if i == 0 else thumb_height

            tkhd.width = [clap_width, 0]
            tkhd.height = [clap_height, 0]

            # moov.trak.mdia.minf.stbl.stsd._vc1
            _vc1 = trak.boxes[-1].boxes[-1].boxes[-1].boxes[0].boxes[0]

            # moov.trak.mdia.minf.stbl.stsd._vc1.clap
            clap = bx_def.CLAP(BoxHeader())
            clap.header.type = b"clap"
            clap.clean_aperture_width_n = clap_width
            clap.clean_aperture_width_d = 1
            clap.clean_aperture_height_n = clap_height
            clap.clean_aperture_height_d = 1
            clap.horiz_off_n = clap_width - _vc1.width
            clap.horiz_off_d = 2
            clap.vert_off_n = clap_height - _vc1.height
            clap.vert_off_d = 2

            # insert 'clap' before 'pasp'
            pasp = _vc1.pop()
            _vc1.append(clap)
            _vc1.append(pasp)


def insert_filenames_trak(traks, mdat, mdat_start_pos, filenames):
    creation_time = 0
    modification_time = creation_time

    chunk_offset = mdat_start_pos + mdat.header.box_size
    sizes = [len(filename) for filename in filenames]

    mdat.data = mdat.data + b''.join(filenames)
    mdat.header.box_size = mdat.header.box_size + sum(sizes)

    filename_trak = make_meta_trak(creation_time, modification_time,
                                   b"bzna_fname",
                                   sizes, chunk_offset)

    # MOOV.TRAK.TKHD
    tkhd = filename_trak.boxes[0]

    # "\x00\x00\x01" trak is enabled
    # "\x00\x00\x02" trak is used in the presentation
    # "\x00\x00\x04" trak is used in the preview
    # "\x00\x00\x08" trak size in not in pixel but in aspect ratio
    tkhd.header.flags = b"\x00\x00\x03"

    tkhd.width = [0, 0]
    tkhd.height = [0, 0]

    # MOOV.TRAK.MDIA.MINF.STBL.STSD.METT
    mett = filename_trak.boxes[-1].boxes[-1].boxes[-1].boxes[0].boxes[0]
    mett.mime_format = b"text/plain\0"

    traks[:] = traks[0:1] + [filename_trak] + traks[1:]


def insert_targets_trak(traks, mdat, mdat_start_pos, mime, targets):
    creation_time = 0
    modification_time = creation_time

    chunk_offset = mdat_start_pos + mdat.header.box_size
    sizes = [8] * len(targets)

    mdat.data = mdat.data + b''.join(targets)
    mdat.header.box_size = mdat.header.box_size + sum(sizes)

    target_trak = make_meta_trak(creation_time, modification_time,
                                 b"bzna_target",
                                 sizes, chunk_offset)

    # MOOV.TRAK.TKHD
    tkhd = target_trak.boxes[0]

    # "\x00\x00\x01" trak is enabled
    # "\x00\x00\x02" trak is used in the presentation
    # "\x00\x00\x04" trak is used in the preview
    # "\x00\x00\x08" trak size in not in pixel but in aspect ratio
    tkhd.header.flags = b"\x00\x00\x00"

    tkhd.width = [0, 0]
    tkhd.height = [0, 0]

    # MOOV.TRAK.MDIA.MINF.STBL.STSD.METT
    mett = target_trak.boxes[-1].boxes[-1].boxes[-1].boxes[0].boxes[0]
    mett.mime_format = bytes(mime, "utf8") + b"\0"

    traks[:] = traks[0:1] + [target_trak] + traks[1:]


def reset_traks_id(moov):
    traks = [box for box in moov.boxes if box.header.type == b"trak"]
    track_id = 1

    for i, trak in enumerate(traks):
        # moov.trak.tkhd
        trak.boxes[0].track_id = track_id
        track_id += 1

    # moov.mvhd
    moov.boxes[0].next_track_id = track_id


def i2m_frame_pad_filter(width, height, tile_width, tile_height):
    pad_width = int((width + tile_width - 1) / tile_width) * tile_width
    pad_height = int((height + tile_height - 1) / tile_height) * tile_height

    pad_filter = []

    while pad_width != width or pad_height != height:
        intermediate_width = min(width * 2, pad_width)
        intermediate_height = min(height * 2, pad_height)
        pad_filter.append("pad={pad_width}:{pad_height}:0:0,"
                          "fillborders=0:{border_right}:0:{border_bottom}:smear,"
                          "format=pix_fmts=yuvj444p"
                          .format(pad_width=intermediate_width,
                                  pad_height=intermediate_height,
                                  border_right=intermediate_width - width,
                                  border_bottom=intermediate_height - height))
        width = intermediate_width
        height = intermediate_height

    # TODO: this is only to prevent too much changes since the last version but
    #       format=pix_fmts=yuvj444p should be removed from the previous loop
    #       and appended at the end of the loop
    if not pad_filter:
        pad_filter.append("format=pix_fmts=yuvj444p")

    return ','.join(pad_filter)


def i2m_frame_scale_and_pad(src, dest, src_width, src_height, codec, crf,
                            tile, make_thumb):
    width_factor = tile.width / src_width if src_width > tile.width else 1
    height_factor = tile.height / src_height if src_height > tile.height else 1
    factor = min(width_factor, height_factor)
    thumb_width = int(src_width * factor)
    thumb_height = int(src_height * factor)
    make_thumb = make_thumb and factor != 1

    # Input filter
    ffmpeg_filter = ["[0:0]format=pix_fmts=yuvj444p[i]",
                     "[i]" +
                     i2m_frame_pad_filter(src_width, src_height, tile.width, tile.height) +
                     "[i]"]
    mapping = ["-map", "[i]"]
    codec_settings = CODECS_DICT[codec] + \
                     ["-pix_fmt", PIXEL_FMTS_DICT[tile.pixel_fmt],
                      "-crf", str(crf)]

    # Thumbnail filter
    if make_thumb:
        ffmpeg_filter.insert(1, "[i]split[i][t]")
        ffmpeg_filter.append("[t]scale=w={width}:h={height},"
                             .format(width=thumb_width, height=thumb_height) +
                             i2m_frame_pad_filter(thumb_width, thumb_height,
                                                  tile.width, tile.height) +
                             "[t]")
        mapping += ["-map", "[t]"]

    ffmpeg_filter = ";".join(ffmpeg_filter)

    subprocess.run(["ffmpeg", "-y", "-framerate", "1", "-i", src,
                    "-filter_complex", ffmpeg_filter] +
                   mapping + codec_settings + ["-f", "mp4", dest],
                   check=True)

    bstr = ConstBitStream(filename=dest)
    boxes = [box for box in Parser.parse(bstr)]
    for box in boxes:
        box.load(bstr)
    del bstr
    os.remove(dest)

    # moov
    moov = boxes[-1]
    # moov.traks
    traks = [box for box in moov.boxes if box.header.type == b"trak"]

    # append thumbnail trak
    if not make_thumb:
        trak_input = traks[0]
        trak_bstr = ConstBitStream(bytes(trak_input))
        trak_thumb = next(Parser.parse(trak_bstr))
        trak_thumb.load(trak_bstr)

        # moov.mvhd
        mvhd = moov.boxes[0]

        # moov.trak.tkhd
        trak_thumb.boxes[0].track_id = mvhd.next_track_id

        # insert before 'udta'
        udta = moov.pop()
        moov.append(trak_thumb)
        moov.append(udta)

        mvhd.next_track_id = mvhd.next_track_id + 1

    return boxes


def image2mp4(args):
    src_item, target_item = args.items if len(args.items) == 2 \
                            else (args.items[0], None)
    tile = args.tile

    with Image.open(src_item.item.path) as src_file:
        src_width, src_height = src_file.size
    width_factor = tile.width / src_width if src_width > tile.width else 1
    height_factor = tile.height / src_height if src_height > tile.height else 1
    factor = min(width_factor, height_factor)
    thumb_width = int(src_width * factor)
    thumb_height = int(src_height * factor)

    boxes = i2m_frame_scale_and_pad(src_item.item.path, args.output,
                                    src_width, src_height, args.codec, args.crf,
                                    tile, src_item.thumb)

    _clean_boxes(boxes)

    ftyp, mdat, moov = boxes
    ftyp.refresh_box_size()

    traks = moov.boxes[1:]

    for i in range(len(traks)):
        moov.pop()

    if target_item is not None:
        with open(target_item.item.path, "rb") as target_file:
            insert_targets_trak(traks, mdat, ftyp.header.box_size, target_item.mime,
                                [target_file.read()])
    insert_filenames_trak(traks, mdat, ftyp.header.box_size,
                          [bytes(src_item.name, "utf8")])
    clap_traks(traks, src_width, src_height, thumb_width, thumb_height)

    for trak in traks:
        moov.append(trak)
    reset_traks_id(moov)

    mp4_bytes_buffer = []
    for box in boxes:
        box.refresh_box_size()
        mp4_bytes_buffer.append(bytes(box))

    with open(args.output, "wb") as output:
        output.write(b''.join(mp4_bytes_buffer))


def build_parser():
    parser = argparse.ArgumentParser(description="Benzina Image2MP4",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--codec", default="h265", choices=list(CODECS_DICT.keys()),
                        help="codec to use in the transcoded image")
    parser.add_argument("--tile", default="512:512:yuv420p",
                        help="tile configuration (WIDTH:HEIGHT:PIXEL_FORMAT)")
    parser.add_argument("--crf", default="10", type=int,
                        help="constant rate factor to use for the transcoded image")
    parser.add_argument("--output", help="target output filename")
    parser.add_argument("-_", action='store', dest='items', nargs='*', default=[], help="")

    return parser


def build_tile_config_parser():
    tile_config_parser = argparse.ArgumentParser(description="Benzina Image2MP4 item config parser",
                                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    tile_config_parser.add_argument("width", type=int, help="The width of the tile")
    tile_config_parser.add_argument("height", type=int, help="The height of the tile")
    tile_config_parser.add_argument("pixel_fmt", choices=list(PIXEL_FMTS_DICT.keys()),
                                    help="The pixel format of the tile")

    return tile_config_parser


def build_item_parser():
    item_parser = argparse.ArgumentParser(description="Benzina Image2MP4 item parser",
                                          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    item_parser.add_argument("--primary", default=False, action="store_true",
                             help="identify the primary item")
    item_parser.add_argument("--hidden", default=False, action="store_true",
                             help="set hidden bit (incompatible with `--primary`)")
    item_parser.add_argument("--name", help="name of the item")
    item_parser.add_argument("--thumb", default=False, action="store_true",
                             help="create a thumbnail")
    item_parser.add_argument("--mime", default="application/octet-stream",
                             help="MIME-type of item (if the item's type is 'mime')")
    item_parser.add_argument("--item", help="item configuration "
                                            "([id=INT,][type=TYPE,]path=PATH)")

    return item_parser


def build_item_config_parser():
    item_config_parser = argparse.ArgumentParser(description="Benzina Image2MP4 item config parser",
                                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    item_config_parser.add_argument("--id", type=int, help="The id of the item")
    item_config_parser.add_argument("--type", choices=TYPES_LIST,
                                    help="The type of the item")
    item_config_parser.add_argument("--path", help="The path of the item's source")

    return item_config_parser


def parse_args(raw_arguments=None):
    argv = sys.argv[1:] if raw_arguments is None else raw_arguments
    parser = build_parser()
    tile_config_parser = build_tile_config_parser()
    item_parser = build_item_parser()
    item_config_parser = build_item_config_parser()

    args, items_argv = parser.parse_known_args(argv)

    tile_config_args = tile_config_parser.parse_args(args.tile.split(":"))
    args.tile = tile_config_args

    item_argv = []

    for arg in items_argv:
        item_argv.append(arg)
        if arg.startswith("--item"):
            item_args = item_parser.parse_args(item_argv)
            item_config_argv = item_args.item.split(",")
            item_config_argv = ["--" + arg for arg in item_config_argv]
            item_config_args = item_config_parser.parse_args(item_config_argv)
            if item_config_args.type != "mime":
                item_args.mime = None
            item_args.item = item_config_args
            args.items.append(item_args)
            item_argv = []

    return args
