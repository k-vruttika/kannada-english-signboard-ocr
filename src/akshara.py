import unicodedata

VIRAMA = "್"
NUKTA = "಼"
ANUSVARA_VISARGA = "ಂಃ"


def is_consonant(ch):
    return "ಕ" <= ch <= "ಹ" or ch in "ೝೞ"


def is_vowel(ch):
    return "ಅ" <= ch <= "ಔ" or ch in "ೠೡ"


def is_vowel_sign(ch):
    # U+0CD5/U+0CD6 are length marks that survive NFC in a few spellings
    return "ಾ" <= ch <= "ೌ" or ch in "ೕೖೢೣ"


def skip_nukta(text, j):
    if j < len(text) and text[j] == NUKTA:
        return j + 1
    return j


def split_aksharas(text):
    """Kannada text -> aksharas. Any other character (Latin letter, digit) is one unit; spaces are dropped."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("‌", "").replace("‍", "")
    units = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        j = i + 1
        if is_consonant(ch):
            j = skip_nukta(text, j)
            while j < len(text) and text[j] == VIRAMA:
                if j + 1 < len(text) and is_consonant(text[j + 1]):
                    j = skip_nukta(text, j + 2)
                else:
                    j += 1  # word-final halant, e.g. the last letter of ಬ್ಯಾಂಕ್
                    break
            while j < len(text) and is_vowel_sign(text[j]):
                j += 1
        # ಅಂಗಡಿ starts with ಅಂ, so anusvara/visarga also attach to an independent vowel
        if (is_consonant(ch) or is_vowel(ch)) and j < len(text) and text[j] in ANUSVARA_VISARGA:
            j += 1
        units.append(text[i:j])
        i = j
    return units


def n_units(text):
    return len(split_aksharas(text))


if __name__ == "__main__":
    assert split_aksharas("ಕನ್ನಡ") == ["ಕ", "ನ್ನ", "ಡ"]
    assert split_aksharas("ಅಂಗಡಿ") == ["ಅಂ", "ಗ", "ಡಿ"]
    assert split_aksharas("ಬೆಂಗಳೂರು") == ["ಬೆಂ", "ಗ", "ಳೂ", "ರು"]
    assert split_aksharas("ಬ್ಯಾಂಕ್") == ["ಬ್ಯಾಂ", "ಕ್"]
    assert split_aksharas("ಹೋಟೆಲ್") == ["ಹೋ", "ಟೆ", "ಲ್"]
    assert split_aksharas("ಸ್ತ್ರೀ") == ["ಸ್ತ್ರೀ"]
    assert split_aksharas("ಶ್ರೀ ಲಕ್ಷ್ಮಿ") == ["ಶ್ರೀ", "ಲ", "ಕ್ಷ್ಮಿ"]
    assert split_aksharas("ದುಃಖ") == ["ದುಃ", "ಖ"]
    assert split_aksharas("ಉಪಾಹಾರ") == ["ಉ", "ಪಾ", "ಹಾ", "ರ"]
    assert split_aksharas("೧೯೯೦") == ["೧", "೯", "೯", "೦"]
    assert split_aksharas("Hotel") == ["H", "o", "t", "e", "l"]
    # decomposed ಕೇ (ಕ + ೆ + ೕ) must give the same single unit as the composed form
    assert split_aksharas("ಕೇ") == ["ಕೇ"]
    assert n_units("ಮೆಡಿಕಲ್ಸ್") == 4
    print("akshara tests passed")
