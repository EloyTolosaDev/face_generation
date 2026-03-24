from dataclasses import dataclass
from typing import List

# ---------------------------------------
# 4) Expression definitions
# ---------------------------------------
@dataclass
class ExpressionSpec:
    name: str
    aus: dict[int, float]  # AU id -> intensity


EXPRESSIONS: List[ExpressionSpec] = [
    ExpressionSpec("simple:smile", {12: 1.0, 6: 0.4, 25: 0.2}),
    ExpressionSpec("simple:duchenne-smile", {12: 1.0, 6: 0.9, 25: 0.2}),
    ExpressionSpec("simple:anger", {4: 1.0, 7: 0.7, 24: 0.8, 23: 0.5}),
    ExpressionSpec("simple:sadness", {1: 0.8, 4: 0.3, 15: 0.9, 17: 0.5}),
    ExpressionSpec("simple:fear", {1: 0.8, 2: 0.8, 4: 0.5, 5: 0.9, 20: 0.8, 25: 0.7}),
    ExpressionSpec("simple:surprise", {1: 0.8, 2: 0.8, 5: 1.0, 26: 0.9}),
    ExpressionSpec("simple:disgust", {9: 0.9, 10: 0.8, 17: 0.5}),
    ExpressionSpec("simple:contempt", {14: 0.9, 12: 0.3, 23: 0.3}),
    ExpressionSpec("simple:smirk", {12: 0.4, 14: 0.7, 23: 0.3}),
    ExpressionSpec("simple:pain", {4: 0.8, 6: 0.6, 7: 0.6, 9: 0.4, 10: 0.4, 43: 0.5}),
    ExpressionSpec("simple:worry", {1: 0.8, 4: 0.6, 5: 0.4, 20: 0.4}),
    ExpressionSpec("simple:embarrassed-smile", {6: 0.4, 12: 0.5, 14: 0.3, 24: 0.4}),
    ExpressionSpec("simple:skepticism", {1: 0.3, 2: 0.6, 14: 0.4, 23: 0.4}),
    ExpressionSpec("simple:determination", {4: 0.6, 7: 0.5, 24: 0.8, 17: 0.4}),
    ExpressionSpec("simple:shock", {1: 1.0, 2: 1.0, 5: 1.0, 26: 1.0, 27: 0.6}),
    ExpressionSpec("simple:scream-fear", {1: 0.9, 2: 0.9, 4: 0.7, 5: 1.0, 20: 0.8, 27: 1.0}),
    ExpressionSpec("simple:crying-face", {1: 0.9, 4: 0.7, 15: 1.0, 17: 0.8, 23: 0.4, 43: 0.5}),
    ExpressionSpec("blend:happy-surprise", {6: 0.8, 12: 0.8, 1: 0.5, 2: 0.5, 5: 0.6, 26: 0.4}),
    ExpressionSpec("blend:fear-surprise", {1: 0.9, 2: 0.9, 5: 1.0, 20: 0.6, 26: 0.7, 4: 0.3}),
    ExpressionSpec("blend:angry-disgust", {4: 0.9, 7: 0.8, 9: 0.7, 10: 0.6, 23: 0.5, 24: 0.6}),
    ExpressionSpec("blend:sad-anger", {1: 0.6, 4: 0.8, 15: 0.7, 24: 0.6, 17: 0.5}),
    ExpressionSpec("blend:sad-smile", {1: 0.6, 15: 0.6, 12: 0.4, 6: 0.2}),
    ExpressionSpec("blend:contempt-smile", {12: 0.5, 14: 0.8, 23: 0.2}),
    ExpressionSpec("blend:relief", {6: 0.5, 12: 0.4, 25: 0.4, 43: 0.3}),
    ExpressionSpec("blend:anxious-smile", {12: 0.5, 20: 0.7, 5: 0.5, 25: 0.6}),
    ExpressionSpec("blend:suppressed-anger", {4: 0.8, 7: 0.7, 23: 0.8, 24: 0.9, 17: 0.5}),
    ExpressionSpec("blend:stoic-pain", {4: 0.6, 7: 0.5, 23: 0.7, 24: 0.7, 17: 0.6}),
]
