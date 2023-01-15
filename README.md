# Facemask Entrance 

Abstraction of the entrance on image to control if the object has entered the given space by crossing an abstract entrance line.
    
### Addon Config

```yaml
 entrance:
    class: vsdkx.addon.facemask.processor.EntranceProcessor
    camera_direction: "down" # Direction of movement
    mask_threshold: 0.01
    line_border: 0.08
```

### Debug 

Object initialization:

```python
from vsdkx.addon.facemask.processor import EntranceProcessor

add_on_config = {
  'class': 'vsdkx.addon.facemask.processor.EntranceProcessor',
  'camera_direction': 'down',
  'mask_threshold': 0.01,
  'line_border': 0.08
}

...

entrance_processor = EntranceProcessor(add_on_config,  model_settings, model_config)

```

#### Input

This addon relys on the following inputs:

```python
addon_object.inference.boxes
addon_object.inference.classes
addon_object.frame
```

#### Output

The addon produces the following output:

```python
  addon_object.inference.extra["entrance_check"] = (people_on_frame,
                                                          masks_on,
                                                          num_face_masks)
                                                          
```
