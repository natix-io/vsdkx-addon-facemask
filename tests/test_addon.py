import numpy as np
import unittest

from vsdkx.core.structs import AddonObject, Inference
from vsdkx.addon.facemask.processor import EntranceProcessor


class MyTestCase(unittest.TestCase):
    addon_config = {
        "camera_direction": "down",
        "mask_threshold": 0.01,
        "line_border": 0.08
    }

    model_config = {
        'mask_on': 1,
        'mask_off': 0
    }

    def test_cross_entrance(self):
        w, h = 640, 640
        bb_true = [120, 150, 170, 200]
        bb_down_right_false = [10, 10, 17, 20]
        bb_up_left_false = [590, 580, 620, 630]

        # down
        current_config = self.addon_config.copy()
        current_config['camera_direction'] = "down"
        addon_processor = EntranceProcessor(current_config, {}, self.model_config, {})

        down_true = addon_processor.cross_entrance(bb_true, w, h)
        down_false = addon_processor.cross_entrance(bb_down_right_false, w, h)

        # up
        current_config = self.addon_config.copy()
        current_config['camera_direction'] = "up"
        addon_processor = EntranceProcessor(current_config, {}, self.model_config, {})

        up_true = addon_processor.cross_entrance(bb_true, w, h)
        up_false = addon_processor.cross_entrance(bb_up_left_false, w, h)

        # left
        current_config = self.addon_config.copy()
        current_config['camera_direction'] = "left"
        addon_processor = EntranceProcessor(current_config, {}, self.model_config, {})

        left_true = addon_processor.cross_entrance(bb_true, w, h)
        left_false = addon_processor.cross_entrance(bb_up_left_false, w, h)

        # right
        current_config = self.addon_config.copy()
        current_config['camera_direction'] = "right"
        addon_processor = EntranceProcessor(current_config, {}, self.model_config, {})

        right_true = addon_processor.cross_entrance(bb_true, w, h)
        right_false = addon_processor.cross_entrance(bb_down_right_false, w, h)

        self.assertTrue(down_true)
        self.assertTrue(up_true)
        self.assertTrue(left_true)
        self.assertTrue(right_true)

        self.assertFalse(down_false)
        self.assertFalse(up_false)
        self.assertFalse(left_false)
        self.assertFalse(right_false)

    def test_post_process(self):
        addon_processor = EntranceProcessor(self.addon_config, {}, self.model_config, {})

        frame = (np.random.rand(640, 640, 3) * 100).astype('uint8')
        inference = Inference()

        bb_1 = np.array([120, 150, 170, 200])
        class_1 = 1

        bb_2 = np.array([50, 60, 250, 380])
        class_2 = 0

        bb_3 = np.array([10, 10, 640, 640])
        class_3 = 1

        inference.boxes = [bb_1, bb_2, bb_3]
        inference.classes = [class_1, class_2, class_3]

        test_object = AddonObject(frame=frame, inference=inference, shared={})
        result = addon_processor.post_process(test_object)

        self.assertIn("entrance_check", result.inference.extra)

        people_on_frame, masks_on, num_face_masks = result.inference.extra["entrance_check"]
        self.assertTrue(people_on_frame)
        self.assertFalse(masks_on)
        self.assertEqual(num_face_masks, 2)


if __name__ == '__main__':
    unittest.main()
