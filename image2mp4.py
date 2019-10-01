import subprocess

from PIL import Image
from bitstring import ConstBitStream

from pybzparse import Parser, boxes as bx_def
from pybzparse.headers import BoxHeader


def clean_and_clap_mp4(mp4_path, width, height, thumb_width, thumb_height):
    bstr = ConstBitStream(filename=mp4_path)
    boxes = [box for box in Parser.parse(bstr)]

    # remove 'free' box
    boxes = boxes[:1] + boxes[2:]

    ftyp = boxes[0]
    mdat = boxes[-2]
    moov = boxes[-1]

    # remove 'udta' box
    moov.boxes.pop()

    traks = [box for box in moov.boxes if box.header.type == b"trak"]

    for i, trak in enumerate(traks):
        clap_width = width if i == 0 else thumb_width
        clap_height = height if i == 0 else thumb_height

        # remove 'edts' box
        mdia = trak.pop()
        trak.pop()
        trak.append(mdia)

        # moov.trak.tkhd
        tkhd = trak.boxes[0]

        if i == 0:
            # "\x00\x00\x01" trak is enabled
            # "\x00\x00\x02" trak is used in the presentation
            # "\x00\x00\x04" trak is used in the preview
            # "\x00\x00\x08" trak size in not in pixel but in aspect ratio
            tkhd.header.flags = (b"\x00\x00\x04",)

        elif i == 1:
            # "\x00\x00\x01" trak is enabled
            # "\x00\x00\x02" trak is used in the presentation
            # "\x00\x00\x04" trak is used in the preview
            # "\x00\x00\x08" trak size in not in pixel but in aspect ratio
            tkhd.header.flags = (b"\x00\x00\x03",)

        tkhd.width = ([clap_width, 0],)
        tkhd.height = ([clap_height, 0],)

        # moov.trak.mdia.minf.stbl
        stbl = trak.boxes[-1].boxes[-1].boxes[-1]

        # moov.trak.mdia.minf.stbl.stsd.avc1
        avc1 = stbl.boxes[0].boxes[0]

        # remove 'pasp' box
        avc1.pop()

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

        avc1.append(clap)

    for box in boxes:
        box.load(bstr)
        box.refresh_box_size()

    if len(traks) == 1:
        trak_input = traks[0]
        trak_bstr = ConstBitStream(bytes(trak_input))
        trak_thumb = next(Parser.parse(trak_bstr))
        trak_thumb.load(trak_bstr)

        # moov.trak.tkhd
        trak_thumb.boxes[0].flags = (b"\x00\x00\x03",)
        trak_thumb.boxes[0].track_id = (trak_input.boxes[0].track_id + 1,)

        moov.append(trak_thumb)

        # moov.mvhd
        moov.boxes[0].next_track_id = (trak_thumb.boxes[0].track_id + 1,)

        traks.append(trak_thumb)

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

    moov.refresh_box_size()

    return b''.join(bytes(box) for box in boxes)


def i2m_frame_pad_filter(width, height, tile_width, tile_height):
    pad_width = int((width + tile_width - 1) / tile_width) * tile_width
    pad_height = int((height + tile_height - 1) / tile_height) * tile_height

    return "pad={pad_width}:{pad_height}:0:0," \
           "fillborders=0:{border_right}:0:{border_bottom}:smear," \
           "format=pix_fmts=yuv420p"\
           .format(pad_width=pad_width, pad_height=pad_height,
                   border_right=pad_width - width,
                   border_bottom=pad_height - height)


def i2m_frame_scale_and_pad(src_path, dst_path, tile_width, tile_height):
    with Image.open(src_path) as src:
        src_width, src_height = src.size
    width_factor = tile_width / src_width if src_width > tile_width else 1
    height_factor = tile_height / src_height if src_height > tile_height else 1
    factor = min(width_factor, height_factor)
    thumb_width = int(src_width * factor)
    thumb_height = int(src_height * factor)

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

    ffmpeg_filter = "".join(ffmpeg_filter)

    process = subprocess.Popen(["ffmpeg", "-y", "-framerate", "1",
                                "-i", "{}".format(src_path),
                                "-filter_complex", ffmpeg_filter] + map + [dst_path])
    process.wait()

    tuned_mp4_bytes = clean_and_clap_mp4(dst_path, src_width, src_height,
                                         thumb_width, thumb_height)

    with open(dst_path, "wb") as output:
        output.write(tuned_mp4_bytes)
