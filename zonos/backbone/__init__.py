try:
    from ._mamba_ssm import ZonosBackbone
except ImportError:
    from ._torch import ZonosBackbone
