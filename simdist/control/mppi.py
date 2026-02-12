from simdist.control.controller import ControllerBase
from simdist.modeling import models


class MppiController(ControllerBase):
    def __init__(self, model: models.ModelBase, model_cfg: dict, *args, **kwargs):
        super().__init__(model, model_cfg, *args, **kwargs)
