from vsdkx.core.interfaces import Addon
from vsdkx.core.structs import AddonObject, Inference
from numpy import ndarray


class EntranceProcessor(Addon):
    """
    Abstraction of the entrance on image to control if the object has
    entered the given space by crossing an abstract entrance line.
    """

    def __init__(self, addon_config: dict, model_settings: dict,
                 model_config: dict, drawing_config: dict):
        super().__init__(addon_config, model_settings, model_config,
                         drawing_config)
        self.direction = addon_config['camera_direction']
        self.mask_threshold = addon_config['mask_threshold']
        self.line = addon_config['line_border']
        self.mask_on = model_config['mask_on']
        self.mask_off = model_config['mask_off']

    def post_process(self, addon_object: AddonObject) -> AddonObject:
        """
        Check if there are people on frame close to camera, if they are
        wearing masks and if someone has entered without a mask

        Args:
            addon_object (AddonObject): addon object containing information
            about inference,
            frame, other addons shared data

        Returns:
            (AddonObject): addon object has updated information for inference
            result and/or shared information:
        """
        masks_on = True
        people_on_frame = False
        no_mask_entrance = 0
        num_face_masks = 0

        for box, class_id in zip(addon_object.inference.boxes,
                                 addon_object.inference.classes):
            box_height = box[3] - box[1]
            box_width = box[2] - box[0]

            if class_id == self.mask_on:
                num_face_masks += 1

            # filter face boxes by threshold
            if (box_height * box_width) > \
                    (self.mask_threshold *
                     addon_object.frame.shape[1] *
                     addon_object.frame.shape[0]):
                people_on_frame = True

                # if class id is 0 then mask is missing so masks_on isn't True
                # anymore and there is no purpose to continue the loop as long
                # as masks_on and people_on_frame will stay the same anyway.
                if class_id == self.mask_off:
                    masks_on = False

                    # as the mask is missing, check for the violation
                    if self.cross_entrance(box,
                                           addon_object.frame.shape[1],
                                           addon_object.frame.shape[0]):
                        no_mask_entrance += 1
        addon_object.inference.extra["entrance_check"] = (people_on_frame,
                                                          masks_on,
                                                          num_face_masks)
        return addon_object

    def cross_entrance(self, box, width, height):
        """
        Check if the object has crossed the entrance line

        Args:
            box (list): Coordinates of box
            width (int): Frame width
            height (int): Frame height

        Returns:
            (bool): Flag whether the object has entered
        """
        x_center = (box[2] + box[0]) / 2
        y_center = (box[3] + box[1]) / 2
        if self.direction == 'up':
            if y_center / height < (1 - self.line):
                return True
        elif self.direction == 'down':
            if y_center / height > self.line:
                return True
        elif self.direction == 'left':
            if x_center / width < (1 - self.line):
                return True
        elif self.direction == 'right':
            if x_center / width > self.line:
                return True
        return False
