"""Optional in-memory RGB overlay. Never writes an image automatically."""
import numpy as np
import torch
from spatial_conditioner import project_wrist


def overlay(rgb, debug, camera, current_wrist_xyz_m=None, batch_index=0):
    """Return annotated RGB at ENCODER resolution (224x304), no file creation.

    Caller supplies the corresponding crop-aligned RGB; drawing onto an unrelated
    original image would hide a preprocessing alignment error. Current wrists must
    be in the exact same physical KP frame as the future anchors.
    """
    import cv2
    if rgb.shape != (224, 304, 3):
        raise ValueError('Overlay requires the corresponding 224x304 encoder RGB')
    result = np.array(rgb, copy=True)
    if 'grid' not in debug:
        return result
    grid = debug['grid'][batch_index].detach().cpu().numpy()
    valid = debug['valid'][batch_index].detach().cpu().numpy()
    pixels = (grid + 1) * np.array([304, 224]) / 2 - .5
    colors = ((0, 255, 0), (255, 80, 0))  # RGB left/right.
    for hand in range(2):
        previous = None
        for uv, ok in zip(pixels[:, hand], valid[:, hand]):
            if not ok:
                previous = None
                continue
            point = tuple(np.rint(uv).astype(int))
            cv2.circle(result, point, 3, colors[hand], -1)
            if previous is not None:
                cv2.line(result, previous, point, colors[hand], 1)
            previous = point
    if current_wrist_xyz_m is not None:
        with torch.no_grad():
            grid, valid = project_wrist(current_wrist_xyz_m[:, None], camera)
        for hand in range(2):
            if bool(valid[batch_index, 0, hand]):
                uv = ((grid[batch_index, 0, hand].cpu().numpy()+1)*[304, 224]/2-.5)
                cv2.drawMarker(result, tuple(np.rint(uv).astype(int)), colors[hand], cv2.MARKER_CROSS, 12, 2)
    return result
