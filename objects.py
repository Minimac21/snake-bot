from typing import Tuple, Any
from dataclasses import dataclass
from enum import Enum


class GameObjectType(Enum):
    FOOD = 1
    ENEMY = 2


@dataclass
class GameObject:
    moment: Tuple[int, int]
    area: float
    contour: Any
    dist_moment_to_center: float = None
