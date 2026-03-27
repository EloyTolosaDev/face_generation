from dataclasses import dataclass
from typing import Optional, Self
from enum import Enum

# ---------------------------------------
# 3) Simple intensity -> weight mapping
# ---------------------------------------

@dataclass
class ActionUnitSemanticTiers:
    light: str
    medium: str
    strong: str

    def __mul__(self, x: float) -> str:
        if 1.1 <= x < 1.3:
            return self.light
        elif 1.3 <= x < 1.7:
            return self.medium
        else:
            return self.strong

@dataclass
class ActionUnit:
    id: int
    desc: str
    tiers: Optional[ActionUnitSemanticTiers] = None

    def __init__(self, id: int, desc: str, light: Optional[str] = None, medium: Optional[str] = None, strong: Optional[str] = None):
        self.id = id
        self.desc = desc
        if all([light, medium, strong]):
            self.tiers = ActionUnitSemanticTiers(light, medium, strong)

    def __eq__(self, other):
        if not isinstance(other, ActionUnit):
            return NotImplemented
        
        return self.id == other.id
    
    def __hash__(self):
        return hash(self.id)

    def __mul__(self, intensity: float) -> str:
        if intensity < 1.0 or intensity > 1.8:
            raise ValueError("Invalid intensity value. Must be >= 1.0 and <= 1.8")
        
        v = f"({self.desc}:{intensity})"
        if self.tiers:
            v = f"({v}, {self.tiers*intensity})"
        
        return v
    
ACTION_UNITS = {
    1: ActionUnit(1, "inner eyebwows raised"),
    4: ActionUnit(4, "brows lowered"),
    6: ActionUnit(6, "cheeks lifted"),
    7: ActionUnit(7, "lids tightened"),
    23: ActionUnit(23, "lips tightened"),
    24: ActionUnit(24, "lips pressed"),
    25: ActionUnit(25, "lips parted"),

    # 1: ActionUnit(
    #     1,
    #     light_desc="inner eyebrows slightly raised",
    #     medium_desc="inner eyebrows raised, subtle forehead tension",
    #     strong_desc="inner eyebrows strongly raised, pronounced central forehead lines",
    # ),
    # 2: ActionUnit(
    #     2,
    #     light_desc="outer eyebrows slightly raised",
    #     medium_desc="outer eyebrows raised",
    #     strong_desc="outer eyebrows strongly raised, high arched brow shape",
    # ),
    # 6: ActionUnit(
    #     6,
    #     light_desc="cheeks slightly lifted",
    #     medium_desc="cheeks lifted, faint smile lines",
    #     strong_desc="cheeks strongly raised, visible smile lines, slight eye squint",
    # ),
    # 12: ActionUnit(
    #     12,
    #     light_desc="mouth corners slightly upturned",
    #     medium_desc="mouth corners upturned",
    #     strong_desc="mouth corners strongly upturned, pronounced grin",
    # ),
    # 25: ActionUnit(
    #     25,
    #     light_desc="lips slightly parted",
    #     medium_desc="lips parted",
    #     strong_desc="lips clearly parted",
    # ),
    # 4: ActionUnit(
    #     4,
    #     light_desc="brows slightly drawn together",
    #     medium_desc="brows furrowed, eyebrows pulled down and together",
    #     strong_desc="brows strongly furrowed, deep crease between eyebrows",
    # ),
    # 7: ActionUnit(
    #     7,
    #     light_desc="eyes slightly narrowed",
    #     medium_desc="eyes narrowed, tense eyelids",
    #     strong_desc="eyes tightly narrowed, intense eyelid tension",
    # ),
    # 23: ActionUnit(
    #     23,
    #     light_desc="mouth slightly tense",
    #     medium_desc="lips tightened, mouth tense",
    #     strong_desc="lips strongly tightened, rigid tense mouth",
    # ),
    # 24: ActionUnit(
    #     24,
    #     light_desc="lips gently pressed together",
    #     medium_desc="lips pressed together firmly",
    #     strong_desc="lips pressed tightly together, strong mouth tension",
    # ),
    # 17: ActionUnit(
    #     17,
    #     light_desc="chin slightly raised",
    #     medium_desc="chin raised, lower lip pushed upward",
    #     strong_desc="chin strongly raised, pronounced lower-lip and chin tension",
    # ),
    # 5: ActionUnit(
    #     5,
    #     light_desc="eyes slightly more open",
    #     medium_desc="upper eyelids raised, eyes more open",
    #     strong_desc="upper eyelids strongly raised, eyes wide open",
    # ),
    # 10: ActionUnit(
    #     10,
    #     light_desc="upper lip slightly raised",
    #     medium_desc="upper lip raised, faint sneer",
    #     strong_desc="upper lip strongly raised (risk: drifts toward disgust)",
    # ),
    # 9: ActionUnit(
    #     9,
    #     light_desc="nose slightly wrinkled",
    #     medium_desc="nose wrinkled",
    #     strong_desc="nose strongly wrinkled, clear nasal crease",
    # ),
    # 14: ActionUnit(
    #     14,
    #     light_desc="mouth corner slightly tightened",
    #     medium_desc="mouth corner tightened (dimpler)",
    #     strong_desc="mouth corner strongly tightened, marked dimpling",
    # ),
    # 15: ActionUnit(
    #     15,
    #     light_desc="mouth corners slightly downturned",
    #     medium_desc="mouth corners downturned",
    #     strong_desc="mouth corners strongly downturned, pronounced sadness shape",
    # ),
    # 16: ActionUnit(
    #     16,
    #     light_desc="lower lip slightly depressed",
    #     medium_desc="lower lip depressed",
    #     strong_desc="lower lip strongly depressed, exposed lower teeth tendency",
    # ),
    # 20: ActionUnit(
    #     20,
    #     light_desc="lips slightly stretched horizontally",
    #     medium_desc="lips stretched horizontally",
    #     strong_desc="lips strongly stretched horizontally, tense fear mouth",
    # ),
    # 26: ActionUnit(
    #     26,
    #     light_desc="jaw slightly dropped",
    #     medium_desc="jaw dropped",
    #     strong_desc="jaw strongly dropped, mouth clearly open",
    # ),
    # 27: ActionUnit(
    #     27,
    #     light_desc="mouth slightly stretched open",
    #     medium_desc="mouth stretched open",
    #     strong_desc="mouth strongly stretched wide open",
    # ),
    # 43: ActionUnit(
    #     43,
    #     light_desc="eyelids softly lowered",
    #     medium_desc="eyes closed briefly",
    #     strong_desc="eyes tightly closed",
    # ),
}


