from enum import Enum

class ActionUnits(str, Enum):
    AU4_HIGH = "(brows strongly lowered and drawn together, deep glabellar frown line:1.5)"
    AU7_MEDIUM = "(eyelids closer between them)(eyelids clearly tightened with narrowed eyes:1.25)"
    AU24_HIGH = "(lips tightly pressed together with strong perioral tension, very thin lips:1.5)"
    AU23_MEDIUM = "(lips firmly tightened into a tense straight mouth:1.25)"
