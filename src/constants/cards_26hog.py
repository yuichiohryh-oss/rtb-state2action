from __future__ import annotations

from typing import Dict, List

CARD_LABELS: Dict[int, str] = {
    1: "HOG_RIDER",
    2: "MUSKETEER",
    3: "CANNON",
    4: "ICE_GOLEM",
    5: "SKELETONS",
    6: "ICE_SPIRIT",
    7: "FIREBALL",
    8: "THE_LOG",
}

CARD_ORDER: List[str] = [CARD_LABELS[idx] for idx in range(1, 9)]
