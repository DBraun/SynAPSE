"""Optional pretrained-weight loaders (SynthRL).

Importing this subpackage does not require ``torch``; ``torch`` is only needed
the first time a downloaded PyTorch checkpoint is converted to ``.safetensors``
(install the ``synthrl-pretrained`` extra). After that, loads are torch-free.
"""

from .synthrl import load_model

__all__ = ["load_model"]
