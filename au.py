from enum import Enum

class ActionUnits(str, Enum):
    ### AU4:
    # AU4_HIGH = "(brows strongly lowered and drawn together, deep glabellar frown line:1.5)"
    # AU4_HIGH = "(brows deeply furrowed, eyebrows pulled down and drawn together, deep vertical glabellar frown lines clearly visible between the eyebrows, strong corrugator muscle tension above the nose bridge:1.65)"
    # AU4_HIGH = "(intense angry brow, eyebrows strongly knitted and lowered, pronounced vertical frown creases in the glabella, compressed central forehead skin folds, visible tension between eyebrows:1.7)"


    AU7_MEDIUM = "(eyelids closer between them)(eyelids clearly tightened with narrowed eyes:1.25)"
    AU24_HIGH = "(lips tightly pressed together with strong perioral tension, very thin lips:1.5)"
    AU23_MEDIUM = "(lips firmly tightened into a tense straight mouth:1.25)"
