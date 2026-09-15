import unittest
from unittest.mock import Mock, patch

import numpy as np

from hdt.online_hat_service import RecoveringImageSource
from hdt import zed2i_image_source as zed


class CameraRecoveryTest(unittest.TestCase):
    def test_outage_skips_images_then_recovers_with_current_state(self):
        frame = np.zeros((1, 3, 240, 320), dtype=np.uint8)
        callback = Mock(side_effect=[frame, zed.ImageSourceUnavailable('stale'),
                                    zed.ImageSourceUnavailable('missing'), frame])
        reader = RecoveringImageSource(callback, zed.RECOVERABLE_IMAGE_ERRORS)
        self.assertIs(reader(robot_state=1), frame)
        self.assertIsNone(reader(robot_state=2))
        self.assertIsNone(reader(robot_state=3))
        self.assertIs(reader(robot_state=4), frame)
        self.assertIsNone(reader.failed_at)
        self.assertEqual(callback.call_args.kwargs['robot_state'], 4)

    def test_configuration_and_receiver_errors_are_not_hidden(self):
        for error in (ValueError('camera count'), RuntimeError('receiver died')):
            reader = RecoveringImageSource(Mock(side_effect=error),
                                           zed.RECOVERABLE_IMAGE_ERRORS)
            with self.assertRaises(type(error)):
                reader()

    def test_actual_source_rejects_stale_and_corrupt_then_accepts_fresh(self):
        receiver = Mock()
        receiver._frames = 1
        image = np.zeros((zed.HEIGHT, zed.WIDTH, 3), dtype=np.uint8)
        ok, jpeg = zed.cv2.imencode('.jpg', image)
        self.assertTrue(ok)
        receiver.latest_jpeg.side_effect = [
            (jpeg.tobytes(), 0.0, 1), (b'bad jpeg', 100.0, 2),
            (b'', 100.0, 3), (jpeg.tobytes(), 100.0, 4),
        ]
        reader = RecoveringImageSource(zed.get_images, zed.RECOVERABLE_IMAGE_ERRORS)
        with patch.object(zed, '_get_receiver', return_value=receiver), \
                patch.object(zed.time, 'monotonic', return_value=100.0):
            for _ in range(3):
                self.assertIsNone(reader(camera_names=['top']))
            self.assertEqual(reader(camera_names=['top']).shape,
                             (1, 3, zed.HEIGHT, zed.WIDTH))


if __name__ == '__main__':
    unittest.main()
