from enum import Enum, EnumMeta
from typing import List


class StrEnumMeta(EnumMeta):
    # this is workaround for submitit pickling leading to instance checks failing in hydra for StrEnum, see
    # https://github.com/facebookresearch/hydra/issues/1156
    @classmethod
    def __instancecheck__(cls, other):
        return "enum" in str(type(other))


class StrEnum(Enum, metaclass=StrEnumMeta):
    def __str__(self):
        return self.value

    def __eq__(self, other: str):
        return self.value == other

    def __repr__(self):
        return self.value

    def __hash__(self):
        return hash(str(self))


def ChoiceEnum(choices: List[str]):
    """return the Enum class used to enforce list of choices"""
    return StrEnum("Choices", {k: k for k in choices})


BUCKET_CHOICE = ChoiceEnum(["uniform"])
PERTURBATION_CHOICES = ChoiceEnum(
    [
        "3kg",
        "random_leads_masking",
        "powerline_noise",
        "emg_noise",
        "baseline_shift",
        "baseline_wander",
    ]
)