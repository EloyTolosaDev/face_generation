from dataclasses import dataclass

# ---------------------------------------
# 3) Simple intensity -> weight mapping
# ---------------------------------------


@dataclass
class AU:
    id: int
    light_desc: str
    medium_desc: str
    strong_desc: str

    def __intensity_to_weight(self, x: float) -> float:
        """
        0.0 -> 0 (omit)
        (0, 0.5] -> light -> 1.3
        (0.5, 1.0] -> strong -> 1.7
        """
        if x <= 0.0:
            return 0.0
        if x <= 0.5:
            return 1.3
        return 1.7

    # Multiplying an AU by a float returns a weighted phrase for prompt generation.
    def __mul__(self, intensity: float) -> str:
        weight = self.__intensity_to_weight(intensity)
        if intensity <= 0:
            return ""
        if intensity <= 0.33:
            return f"{self.light_desc}:{weight}"
        if intensity <= 0.66:
            return f"{self.medium_desc}:{weight}"
        return f"{self.strong_desc}:{weight}"


ACTION_UNITS = {
    1: AU(
        1,
        light_desc="inner eyebrows slightly raised",
        medium_desc="inner eyebrows raised, subtle forehead tension",
        strong_desc="inner eyebrows strongly raised, pronounced central forehead lines",
    ),
    2: AU(
        2,
        light_desc="outer eyebrows slightly raised",
        medium_desc="outer eyebrows raised",
        strong_desc="outer eyebrows strongly raised, high arched brow shape",
    ),
    6: AU(
        6,
        light_desc="cheeks slightly lifted",
        medium_desc="cheeks lifted, faint smile lines",
        strong_desc="cheeks strongly raised, visible smile lines, slight eye squint",
    ),
    12: AU(
        12,
        light_desc="mouth corners slightly upturned",
        medium_desc="mouth corners upturned",
        strong_desc="mouth corners strongly upturned, pronounced grin",
    ),
    25: AU(
        25,
        light_desc="lips slightly parted",
        medium_desc="lips parted",
        strong_desc="lips clearly parted",
    ),
    4: AU(
        4,
        light_desc="brows slightly drawn together",
        medium_desc="brows furrowed, eyebrows pulled down and together",
        strong_desc="brows strongly furrowed, deep crease between eyebrows",
    ),
    7: AU(
        7,
        light_desc="eyes slightly narrowed",
        medium_desc="eyes narrowed, tense eyelids",
        strong_desc="eyes tightly narrowed, intense eyelid tension",
    ),
    23: AU(
        23,
        light_desc="mouth slightly tense",
        medium_desc="lips tightened, mouth tense",
        strong_desc="lips strongly tightened, rigid tense mouth",
    ),
    24: AU(
        24,
        light_desc="lips gently pressed together",
        medium_desc="lips pressed together firmly",
        strong_desc="lips pressed tightly together, strong mouth tension",
    ),
    17: AU(
        17,
        light_desc="chin slightly raised",
        medium_desc="chin raised, lower lip pushed upward",
        strong_desc="chin strongly raised, pronounced lower-lip and chin tension",
    ),
    5: AU(
        5,
        light_desc="eyes slightly more open",
        medium_desc="upper eyelids raised, eyes more open",
        strong_desc="upper eyelids strongly raised, eyes wide open",
    ),
    10: AU(
        10,
        light_desc="upper lip slightly raised",
        medium_desc="upper lip raised, faint sneer",
        strong_desc="upper lip strongly raised (risk: drifts toward disgust)",
    ),
    9: AU(
        9,
        light_desc="nose slightly wrinkled",
        medium_desc="nose wrinkled",
        strong_desc="nose strongly wrinkled, clear nasal crease",
    ),
    14: AU(
        14,
        light_desc="mouth corner slightly tightened",
        medium_desc="mouth corner tightened (dimpler)",
        strong_desc="mouth corner strongly tightened, marked dimpling",
    ),
    15: AU(
        15,
        light_desc="mouth corners slightly downturned",
        medium_desc="mouth corners downturned",
        strong_desc="mouth corners strongly downturned, pronounced sadness shape",
    ),
    16: AU(
        16,
        light_desc="lower lip slightly depressed",
        medium_desc="lower lip depressed",
        strong_desc="lower lip strongly depressed, exposed lower teeth tendency",
    ),
    20: AU(
        20,
        light_desc="lips slightly stretched horizontally",
        medium_desc="lips stretched horizontally",
        strong_desc="lips strongly stretched horizontally, tense fear mouth",
    ),
    26: AU(
        26,
        light_desc="jaw slightly dropped",
        medium_desc="jaw dropped",
        strong_desc="jaw strongly dropped, mouth clearly open",
    ),
    27: AU(
        27,
        light_desc="mouth slightly stretched open",
        medium_desc="mouth stretched open",
        strong_desc="mouth strongly stretched wide open",
    ),
    43: AU(
        43,
        light_desc="eyelids softly lowered",
        medium_desc="eyes closed briefly",
        strong_desc="eyes tightly closed",
    ),
}


