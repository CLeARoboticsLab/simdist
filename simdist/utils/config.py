import numpy as np


def proprio_obs_dim_from_sys_config(sys_cfg: dict) -> int:
    """
    Extracts the proprioceptive observation dimension from the system configuration.

    Args:
        cfg (dict): Configuration dictionary containing system parameters.

    Returns:
        int: The observation dimension.
    """
    proprio_obs_dims = [ob["dim"] for ob in sys_cfg["proprio_obs"]["types"]]
    return sum(proprio_obs_dims)


def proprio_obs_names_from_sys_config(sys_cfg: dict) -> list[str]:
    """
    Extracts the proprioceptive observation names from the system configuration.

    Args:
        cfg (dict): Configuration dictionary containing system parameters.

    Returns:
        list[str]: The observation names.
    """
    return [ob["name"] for ob in sys_cfg["proprio_obs"]["types"]]


def extero_obs_dim_from_sys_config(sys_cfg: dict) -> int:
    """
    Extracts the exteroceptive observation dimension from the system configuration.

    Args:
        cfg (dict): Configuration dictionary containing system parameters.

    Returns:
        int: The observation dimension.
    """
    extero_obs = sys_cfg["extero_obs"]
    dim = 0
    for ob in extero_obs["types"]:
        if isinstance(ob["dim"], list):
            dim += np.prod(ob["dim"])
        else:
            dim += ob["dim"]

    return dim


def extero_obs_names_from_sys_config(sys_cfg: dict) -> list[str]:
    """
    Extracts the exteroceptive observation names from the system configuration.

    Args:
        cfg (dict): Configuration dictionary containing system parameters.

    Returns:
        list[str]: The observation names.
    """
    return [ob["name"] for ob in sys_cfg["extero_obs"]["types"]]


def height_map_dims_from_sys_cfg(sys_cfg: dict) -> tuple[int, int]:
    """
    Extracts the height map dimensions from the system configuration.

    Args:
        cfg (dict): Configuration dictionary containing system parameters.

    Returns:
        tuple[int, int]: The length and width of the height map.
    """
    for ob in sys_cfg["extero_obs"]["types"]:
        if ob["name"] == "height_scan":
            hx, hy = ob["dim"]
            return hx, hy
    raise ValueError(
        "`height_scan` observation type not found in extero_obs in system config."
    )


def action_dim_from_sys_config(sys_cfg: dict) -> int:
    """
    Extracts the action dimension from the system configuration.

    Args:
        cfg (dict): Configuration dictionary containing system parameters.

    Returns:
        int: The action dimension.
    """
    return len(sys_cfg["actions"])


def cmd_dim_from_sys_config(sys_cfg: dict) -> int:
    """
    Extracts the command dimension from the system configuration.

    Args:
        cfg (dict): Configuration dictionary containing system parameters.

    Returns:
        int: The command dimension.
    """
    return sys_cfg["cmd"]["dim"]
