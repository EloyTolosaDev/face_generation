from enum import Enum

class ActionUnits(str, Enum):
    AU4_HIGH = "(brows strongly lowered and drawn together, deep glabellar frown line)"
    AU7_MEDIUM = "(eyelids clearly tightened with narrowed eyes)"
    AU24_HIGH = "(lips tightly pressed together with strong perioral tension)"
    AU23_MEDIUM = "(lips firmly tightened into a tense straight mouth)"
