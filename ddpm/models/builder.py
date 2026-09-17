
import logging
from typing import Dict, List, Tuple, Any, Union

import torch
from torch import nn
from .diffusion_denoising import DiffusionModel, DenoisingModel
from .unet_openai import create_unet_openai

LOGGER = logging.getLogger(__name__)


def build_model(
        time_steps: int,
        schedule: str,
        schedule_params: Union[dict, None],
        input_shapes: List[Tuple[int, int, int]],
        label_smoothing: float,
        backbone: str,
        backbone_params: Dict[str, Any],
        diffusion_type: str = 'categorical',
        bits: int = None,
        analog_bits_scale: float = 1.0,
        params=None
) -> DenoisingModel:

    img_shape, label_shape = input_shapes
    img_channels = img_shape[0]

    # diffusion type: affects input/output shape
    if diffusion_type == 'categorical':
        num_classes = label_shape[0]  # can be accessed from this cause label is in one_hot encoding
        LOGGER.info(f"Using diffusion model [{diffusion_type}] with num_classes [{num_classes}]")
        diffusion = DiffusionModel(schedule, time_steps, num_classes, schedule_params=schedule_params)
        input_channels = num_classes
        output_channels = num_classes
    elif diffusion_type == 'continuous_analog_bits':
        num_classes = label_shape[0]  # can be accessed from this cause label is in one_hot encoding
        LOGGER.info(f"Using diffusion model [{diffusion_type}] with num_classes [{num_classes}]"
                    f" and input channels = bits: [{bits}]")
        input_channels = bits
        output_channels = bits
        backbone_params.update({"softmax_output": False})
    else:
        raise NotImplementedError(f'unknown diffusion {diffusion_type} in params.yml')

    backbone_params_implicit = {}
    backbone_params_implicit.update({'in_channels': input_channels,
                                     'out_channels': output_channels,
                                     'num_res_blocks': 2})

    LOGGER.info(f"Implicit backbone params: {backbone_params_implicit}")
    LOGGER.info(f"Explicit backbone params: {backbone_params}")
    backbone_params.update(backbone_params_implicit)

    model: nn.Module
    if backbone == "unet_openai":
        model = create_unet_openai(image_size=min(img_shape[1], img_shape[2]), **backbone_params, params=params)
    else:
        raise NotImplementedError(f"backbone {backbone}")

    num_of_parameters = sum(map(torch.numel, model.parameters()))
    LOGGER.info("%s trainable params: %d", backbone, num_of_parameters)
    LOGGER.info(f"unet lighweight: {backbone_params.get('is_lightweight', False)}")

    if params is not None:
        # for logging only
        params['num_params'] = num_of_parameters
        params['unet_openai'] = backbone_params

    if diffusion_type == 'categorical':
        ret = DenoisingModel(diffusion, model, guidance_scale=None, guidance_scale_weighting=None, guidance_loss_fn=None)
    return ret
