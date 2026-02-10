import os
import copy
import torch
from simdist.utils import paths


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


def get_actor_critic_from_iteration(
    run_name: str, iteration: int, device: str | torch.device
) -> tuple[torch.jit.ScriptModule, torch.jit.ScriptModule]:
    policy_dir = paths.get_rl_policies_dir(run_name)
    critic_dir = paths.get_rl_critics_dir(run_name)
    policy = f"policy_{iteration}.pt"
    critic = f"critic_{iteration}.pt"
    policy_path = os.path.join(policy_dir, policy)
    critic_path = os.path.join(critic_dir, critic)
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy file not found: {policy_path}")
    if not os.path.exists(critic_path):
        raise FileNotFoundError(f"Critic file not found: {critic_path}")
    print(f"Found policy: {policy_path}")
    print(f"Found critic: {critic_path}")

    policy = torch.jit.load(policy_path).eval().to(device)
    critic = torch.jit.load(critic_path).eval().to(device)

    return policy, critic


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
