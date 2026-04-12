import numpy as np
from pydantic import Field

from iris.callbacks.callback_interface import Callback
from iris.io.class_configs import Algorithm
from iris.io.dataclasses import IRImage


class ImageDownsampler(Algorithm):
    """Implementation of an image downsampling algorithm that reduces the resolution of an input image by a scaling factor.

    Algorithm steps:
        1) Resize the raw image data using OpenCV's Lanczos interpolation (INTER_LANCZOS4) for high-quality decimation.
        2) Instantiate a new IRImage object with the downsampled data while preserving the original image metadata (ID and eye side).
    """

    class Parameters(Algorithm.Parameters):
        """ImageDownsampler parameters."""

        factor: float = Field(..., ge=0.0, le=1.0)

    __parameters_type__ = Parameters
    
    def __init__(self, factor: float = 1.0) -> None:
        super().__init__(
            factor=factor,
        )

    def run(self, image: IRImage) -> IRImage:
        new_img_data = cv2.resize(iris_img.img_data, None, fx=self.factor, fy=self.factor, interpolation=cv2.INTER_LANCZOS4)
        new_image = iris.IRImage(img_data=new_img_data, image_id=image.image_id, eye_side=image.eye_side)
        return new_image