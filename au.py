from enum import Enum


class ActionUnits(str, Enum):
    AU4_HIGH = "(eyebrows strongly lowered and drawn together, pronounced vertical glabellar frown lines between the eyebrows, visible corrugator tension above the nose bridge)1.65"
    AU7_MEDIUM = "(eyelids clearly tightened with narrowed eyes)1.25"
    AU24_HIGH = "(lips tightly pressed together with strong perioral tension, very thin lips)1.5"
    AU23_MEDIUM = "(lips firmly tightened into a tense straight mouth)1.25"


AU_NEGATIVE_MAP: dict[ActionUnits, str] = {
    ActionUnits.AU4_HIGH: "raised eyebrows, relaxed brows, smooth glabella, no brow furrow",
    ActionUnits.AU7_MEDIUM: "wide open eyes, relaxed eyelids",
    ActionUnits.AU24_HIGH: "open lips, parted lips, smiling mouth",
    ActionUnits.AU23_MEDIUM: "smile, grin, upturned mouth corners",
}


def get_au_negative_prompt(au_list: list[ActionUnits]) -> str:
    """Build a comma-separated negative prompt from the selected AUs."""
    negatives: list[str] = []
    seen: set[str] = set()
    for au in au_list:
        phrase = AU_NEGATIVE_MAP.get(au)
        if not phrase or phrase in seen:
            continue
        negatives.append(phrase)
        seen.add(phrase)
    return ", ".join(negatives)


def get_au_negative_prompt_from_names(au_names: list[str]) -> str:
    """Build AU negative prompt from AU enum names present in meta files."""
    aus: list[ActionUnits] = []
    for name in au_names:
        token = str(name).strip().upper()
        if not token:
            continue
        if token.isdigit():
            token = f"AU{token}"

        try:
            aus.append(ActionUnits[token])
        except KeyError:
            # Backward compatibility: map AU4-like names to an available AU4_* entry.
            prefix = token.split("_", 1)[0]
            for candidate in ActionUnits:
                if candidate.name.startswith(f"{prefix}_"):
                    aus.append(candidate)
                    break
    return get_au_negative_prompt(aus)
