import os
import copy
import torch


def export_torch_as_jit(
    net: object,
    path: str,
    normalizer: object | None = None,
    filename="net.onnx",
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    exporter = _TorchExporter(net, normalizer)
    exporter.export(path, filename)


class _TorchExporter(torch.nn.Module):
    """Exporter of the actor or critic into JIT files."""

    def __init__(self, net, normalizer=None):
        super().__init__()
        self.net = copy.deepcopy(net)

        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward(self, x):
        return self.net(self.normalizer(x))

    @torch.jit.export
    def reset(self):
        pass

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)
