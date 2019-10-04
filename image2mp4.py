import subprocess
from datetime import datetime

from bitstring import ConstBitStream
from PIL import Image

from pybzparse import Parser, boxes as bx_def
from pybzparse.headers import BoxHeader
from pybzparse.utils import make_meta_trak, to_mp4_time


def parse_and_load_boxes(filepath):
    bstr = ConstBitStream(filename=filepath)
    boxes = [box for box in Parser.parse(bstr)]
    for box in boxes:
        box.load(bstr)
    return boxes


def clean_boxes(boxes):
    # remove 'free' box
    boxes[:] = boxes[:1] + boxes[2:]

    ftyp = boxes[0]
    mdat = boxes[1]
    moov = boxes[2]

    # remove 'udta' box
    moov.boxes.pop()

    # moov.mvhd
    moov.boxes[0].timescale = (20,)
    moov.boxes[0].duration = (20,)

    traks = [box for box in moov.boxes if box.header.type == b"trak"]

    # append thumb trak
    if len(traks) == 1:
        trak_input = traks[0]
        trak_bstr = ConstBitStream(bytes(trak_input))
        trak_thumb = next(Parser.parse(trak_bstr))
        trak_thumb.load(trak_bstr)

        # moov.mvhd
        mvhd = moov.boxes[0]

        # moov.trak.tkhd
        trak_thumb.boxes[0].track_id = (mvhd.next_track_id,)

        moov.append(trak_thumb)

        mvhd.next_track_id = (mvhd.next_track_id + 1,)

        traks.append(trak_thumb)

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
            tkhd.header.flags = (b"\x00\x00\x00" if i == 0 else b"\x00\x00\x03",)

            hdlr.name = (b"bzna_input" if i == 0 else b"bzna_thumb",)

    ftyp.major_brand = (1769172845,)  # b"isom"
    ftyp.minor_version = (0,)
    ftyp.compatible_brands = ([1769172845],)  # b"isom"
    ftyp.refresh_box_size()

    chunk_offset_diff = ftyp.header.box_size - mdat.header.start_pos

    for i, trak in enumerate(traks):
        # moov.trak.mdia.minf.stbl
        stbl = trak.boxes[-1].boxes[-1].boxes[-1]

        # moov.trak.mdia.minf.stbl.stco
        stco = stbl.boxes[-1]
        for entry in stco.entries:
            entry.chunk_offset = (entry.chunk_offset + chunk_offset_diff,)


def clap_traks(traks, width, height, thumb_width, thumb_height):
    for i, trak in enumerate(traks):
        # moov.trak.tkhd
        tkhd = trak.boxes[0]

        # moov.trak.mdia.hdlr
        hdlr = trak.boxes[-1].boxes[1]

        if hdlr.handler_type == b"vide":
            clap_width = width if i == 0 else thumb_width
            clap_height = height if i == 0 else thumb_height

            tkhd.width = ([clap_width, 0],)
            tkhd.height = ([clap_height, 0],)

            # moov.trak.mdia.minf.stbl.stsd.avc1
            avc1 = trak.boxes[-1].boxes[-1].boxes[-1].boxes[0].boxes[0]

            # moov.trak.mdia.minf.stbl.stsd.avc1.clap
            clap = bx_def.CLAP(BoxHeader())
            clap.header.type = b"clap"
            clap.clean_aperture_width_n = (clap_width,)
            clap.clean_aperture_width_d = (1,)
            clap.clean_aperture_height_n = (clap_height,)
            clap.clean_aperture_height_d = (1,)
            clap.horiz_off_n = (clap_width - avc1.width,)
            clap.horiz_off_d = (2,)
            clap.vert_off_n = (clap_height - avc1.height,)
            clap.vert_off_d = (2,)

            # insert 'clap' before 'pasp'
            pasp = avc1.pop()
            avc1.append(clap)
            avc1.append(pasp)


def insert_filenames_trak(traks, mdat, mdat_start_pos, filenames):
    creation_time = to_mp4_time(datetime.utcnow())
    modification_time = creation_time

    chunk_offset = mdat_start_pos + mdat.header.box_size
    filenames = [bytes(filename, "utf8") for filename in filenames]
    sizes = [len(filename) for filename in filenames]

    mdat.data = (mdat.data + b''.join(filenames),)
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
    tkhd.header.flags = (b"\x00\x00\x03",)

    tkhd.width = ([0, 0],)
    tkhd.height = ([0, 0],)

    # MOOV.TRAK.MDIA.MINF.STBL.STSD.METT
    mett = filename_trak.boxes[-1].boxes[-1].boxes[-1].boxes[0].boxes[0]
    mett.mime_format = (b"text/plain\0",)

    traks[:] = traks[0:1] + [filename_trak] + traks[1:]


def insert_targets_trak(traks, mdat, mdat_start_pos, targets):
    creation_time = to_mp4_time(datetime.now())
    modification_time = creation_time

    chunk_offset = mdat_start_pos + mdat.header.box_size
    targets = [target.to_bytes(8, byteorder="little") for target in targets]
    sizes = [8] * len(targets)

    mdat.data = (mdat.data + b''.join(targets),)
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
    tkhd.header.flags = (b"\x00\x00\x00",)

    tkhd.width = ([0, 0],)
    tkhd.height = ([0, 0],)

    # MOOV.TRAK.MDIA.MINF.STBL.STSD.METT
    mett = target_trak.boxes[-1].boxes[-1].boxes[-1].boxes[0].boxes[0]
    mett.mime_format = (b"text/plain\0",)

    traks[:] = traks[0:1] + [target_trak] + traks[1:]


def reset_traks_id(moov):
    traks = [box for box in moov.boxes if box.header.type == b"trak"]
    track_id = 1

    for i, trak in enumerate(traks):
        # moov.trak.tkhd
        trak.boxes[0].track_id = (track_id,)
        track_id += 1

    # moov.mvhd
    moov.boxes[0].next_track_id = (track_id,)


def i2m_frame_pad_filter(width, height, tile_width, tile_height):
    pad_width = int((width + tile_width - 1) / tile_width) * tile_width
    pad_height = int((height + tile_height - 1) / tile_height) * tile_height

    return "pad={pad_width}:{pad_height}:0:0," \
           "fillborders=0:{border_right}:0:{border_bottom}:smear," \
           "format=pix_fmts=yuv420p" \
           .format(pad_width=pad_width, pad_height=pad_height,
                   border_right=pad_width - width,
                   border_bottom=pad_height - height)


def i2m_frame_scale_and_pad(src, dest, src_width, src_height, tile_width, tile_height):
    width_factor = tile_width / src_width if src_width > tile_width else 1
    height_factor = tile_height / src_height if src_height > tile_height else 1
    factor = min(width_factor, height_factor)
    thumb_width = int(src_width * factor)
    thumb_height = int(src_height * factor)

    # srt = filepath2srt(src)

    # Input filter
    ffmpeg_filter = ["[0:0]",
                     i2m_frame_pad_filter(src_width, src_height,
                                          tile_width, tile_height),
                     "[i]"]
    map = ["-map", "[i]"]

    # Thumbnail filter
    if factor != 1:
        ffmpeg_filter += [";[0:0]scale=w={width}:h={height},"
                          .format(width=thumb_width, height=thumb_height),
                          i2m_frame_pad_filter(thumb_width, thumb_height,
                                               tile_width, tile_height),
                          "[t]"]
        map += ["-map", "[t]"]

    # # Subtitle track
    # map += ["-map", "1", "-c:s", "mov_text"]

    ffmpeg_filter = "".join(ffmpeg_filter)

    process = subprocess.Popen(["ffmpeg", "-y", "-framerate", "1",
                                "-i", src, # "-i", srt,
                                "-filter_complex", ffmpeg_filter] + map + [dest])
    process.wait()


def image2mp4(src, dest, target, tile_width, tile_height):
    with Image.open(src) as src_file:
        src_width, src_height = src_file.size
    width_factor = tile_width / src_width if src_width > tile_width else 1
    height_factor = tile_height / src_height if src_height > tile_height else 1
    factor = min(width_factor, height_factor)
    thumb_width = int(src_width * factor)
    thumb_height = int(src_height * factor)

    i2m_frame_scale_and_pad(src, dest, src_width, src_height, tile_width, tile_height)

    boxes = parse_and_load_boxes(dest)
    clean_boxes(boxes)

    ftyp, mdat, moov = boxes
    ftyp.refresh_box_size()

    traks = moov.boxes[1:]

    for i in range(len(traks)):
        moov.pop()

    insert_filenames_trak(traks, mdat, ftyp.header.box_size, [src])
    insert_targets_trak(traks, mdat, ftyp.header.box_size, [target])
    clap_traks(traks, src_width, src_height, thumb_width, thumb_height)

    for trak in traks:
        moov.append(trak)
    reset_traks_id(moov)

    mp4_bytes_buffer = []
    for box in boxes:
        box.refresh_box_size()
        mp4_bytes_buffer.append(bytes(box))

    with open(dest, "wb") as output:
        output.write(b''.join(mp4_bytes_buffer))
