import numpy as np
import pytest
from PIL import Image
from pybenzinaconcat.test.psnrhma import PilResizeMaxSide, get_image_resize


@pytest.mark.parametrize("img_size", [(256, 256), (128, 256), (256, 128),
                                      (64, 128), (128, 64), (100, 70)])
@pytest.mark.parametrize("resize_edge", [256, 128, 64])
def test_pilresizemaxside(img_size, resize_edge):
    img_x, img_y = img_size
    # In channel 0, values will increase on X axis
    # In channel 1, values will increase on Y axis
    arr = np.array([[list(range(256))] * 256] * 3).transpose([1, 2, 0]) \
        .astype(np.uint8)
    arr[:, :, 0] = arr[:, :, 0].T
    arr[:, :, 2] = 0
    img = Image.fromarray(arr[:img_x, :img_y, :])

    assert img.size == (img_y, img_x)

    resize = PilResizeMaxSide(resize_edge)
    resize_img = resize(img)

    assert resize_img.size == get_image_resize(img.size, resize_edge)

    if img_size == (resize_edge, resize_edge) or \
            (img_x <= resize_edge and img_y <= resize_edge):
        assert resize_img.size == img.size
        assert resize_img == img
    elif img_x >= resize_edge and img_y >= resize_edge:
        factor = max(np.array(img.size) / np.array(resize_img.size))
        # Apparently, np.array(resize_img)[0, 0, :] - np.array(img)[0, 0, :]
        # does not give the expected result where the first vector gets
        # subtracted by the second as the max value is as high as 255
        img_corners = np.array([0] * (3 * 4))
        resize_img_corners = np.array([0] * (3 * 4))
        img_corners[0:3] = np.array(img)[0, 0, :]
        img_corners[3:6] = np.array(img)[0, -1, :]
        img_corners[6:9] = np.array(img)[-1, 0, :]
        img_corners[9:12] = np.array(img)[-1, -1, :]
        resize_img_corners[0:3] = np.array(resize_img)[0, 0, :]
        resize_img_corners[3:6] = np.array(resize_img)[0, -1, :]
        resize_img_corners[6:9] = np.array(resize_img)[-1, 0, :]
        resize_img_corners[9:12] = np.array(resize_img)[-1, -1, :]
        assert (np.abs(resize_img_corners - img_corners) <= factor / 2).all()
    else:
        assert False
