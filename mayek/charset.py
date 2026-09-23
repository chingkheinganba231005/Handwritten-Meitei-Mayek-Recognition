"""The 55 TUMMHCD classes and the Meitei Mayek characters they stand for.

Class ids come straight from the dataset folder names (``train_000`` ...
``train_054``). Names follow the Unicode character names.
"""

from typing import NamedTuple, Optional


class CharClass(NamedTuple):
    index: int
    char: Optional[str]  # None while the mapping is unconfirmed
    name: str

    @property
    def id(self) -> str:
        return f"{self.index:03d}"

    @property
    def display(self) -> str:
        return self.char if self.char is not None else "?"


_TABLE = [
    # Cheising Iyek: digits. The dataset orders them 1..9, then 0.
    ("꯱", "one"), ("꯲", "two"), ("꯳", "three"), ("꯴", "four"), ("꯵", "five"),
    ("꯶", "six"), ("꯷", "seven"), ("꯸", "eight"), ("꯹", "nine"), ("꯰", "zero"),
    # Iyek Ipee: the 27 base letters (18 Mapum Mayek + 9 Lom Iyek)
    ("ꯀ", "kok"), ("ꯁ", "sam"), ("ꯂ", "lai"), ("ꯃ", "mit"), ("ꯄ", "pa"),
    ("ꯅ", "na"), ("ꯆ", "chil"), ("ꯇ", "til"), ("ꯈ", "khou"), ("ꯉ", "ngou"),
    ("ꯊ", "thou"), ("ꯋ", "wai"), ("ꯌ", "yang"), ("ꯍ", "huk"), ("ꯎ", "un"),
    ("ꯏ", "i"), ("ꯐ", "pham"), ("ꯑ", "atiya"), ("ꯒ", "gok"), ("ꯓ", "jham"),
    ("ꯔ", "rai"), ("ꯕ", "ba"), ("ꯖ", "jil"), ("ꯗ", "dil"), ("ꯘ", "ghou"),
    ("ꯙ", "dhou"), ("ꯚ", "bham"),
    # Lonsum Iyek: final consonants
    ("ꯛ", "kok lonsum"), ("ꯜ", "lai lonsum"), ("ꯝ", "mit lonsum"),
    ("ꯞ", "pa lonsum"), ("ꯟ", "na lonsum"), ("ꯠ", "til lonsum"),
    ("ꯡ", "ngou lonsum"), ("ꯢ", "i lonsum"),
    # Cheitap Iyek: vowel signs
    ("ꯥ", "anap"), ("ꯦ", "yenap"), ("ꯨ", "unap"), ("ꯤ", "inap"),
    ("ꯩ", "cheinap"), ("ꯣ", "onap"), ("ꯧ", "sounap"), ("ꯪ", "nung"),
    # Khudam: punctuation
    ("꯫", "cheikhei"),
    # TODO: 054 has not been matched to a character yet.
    (None, "unconfirmed"),
]

CLASSES = [CharClass(i, char, name) for i, (char, name) in enumerate(_TABLE)]
NUM_CLASSES = len(CLASSES)

SCRIPT_GROUPS = {
    "Cheising Iyek (digits)": range(0, 10),
    "Iyek Ipee (letters)": range(10, 37),
    "Lonsum Iyek (finals)": range(37, 45),
    "Cheitap Iyek (vowel signs)": range(45, 53),
    "Khudam (punctuation)": range(53, 55),
}


def script_group(index: int) -> int:
    """Index of the script sub-category a class belongs to (0-4)."""
    for group, members in enumerate(SCRIPT_GROUPS.values()):
        if index in members:
            return group
    raise ValueError(f"class index out of range: {index}")
