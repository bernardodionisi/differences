from __future__ import annotations

from copy import deepcopy
from typing import Mapping


def update_nested_dict(template: dict, user_in: Mapping):
    """Recursively update nested dictionary"""

    for key, val in user_in.items():
        if isinstance(val, Mapping):
            if key in template and isinstance(template[key], Mapping):
                template[key] = update_nested_dict(template.get(key, {}), val)
            else:
                template[key] = deepcopy(val)
        else:
            template[key] = val
    return template
