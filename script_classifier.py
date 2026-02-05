from collections import Counter
from typing import Optional

# -----------------------------------------------------------------------------
# Language determination functions
# just a basic unicode sieve, for now
# Unicode block reference:  http://unicode.org/charts/

# - U+0000 – U+007F: Latin alphabet
ascii = [
    (0x0000, 0x007F),
]

# - U+0600 - U+06FF	Arabic
# - U+0750 - U+077F	Arabic Supplement
# - U+08A0 - U+08FF	Arabic Extended-A
# - U+0870 - U+089F	Arabic Extended-B
# - U+10EC0 - U+10EFF	Arabic Extended-C
# - U+FB50 - U+FDFF	Arabic Presentation Forms-A
# - U+FE70 - U+FEFF	Arabic Presentation Forms-B
arabic = [
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0x0870, 0x089F),
    (0x10EC0, 0x10EFF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
]

# https://character-table.netlify.app/belarusian/
# only belarusian:
#   ў (U+040E, U+045E)
only_belarusian = [
    (0x040E, 0x040E),
    (0x045E, 0x045E),
]
belarusian = [
    (0x0401, 0x0401),
    (0x0406, 0x0406),
    (0x040E, 0x040E),
    (0x0410, 0x0428),
    (0x042B, 0x0437),
    (0x0439, 0x0448),
    (0x044B, 0x044F),
    (0x0451, 0x0451),
    (0x0456, 0x0456),
    (0x045E, 0x045E),
]


# ---- TODO: look into these unicode blocks
# - U+3200 - U+32FF: Enclosed CJK Letters and Months
# - U+3300 - U+33FF: CJK Compatibility
# - U+3400 - U+4DBF: CJK Unified Ideographs Extension A
# - U+4DC0 - U+4DFF: Yijing Hexagram Symbols (I Ching hexagrams)
# - U+A000 - U+A48F: Yi Syllables (Nuosu language)
# - U+A490 - U+A4CF: Yi Radicals (Nuosu language)

# - U+4E00 – U+9FFF: CJK Unified Ideographs
# - U+3000 - U+303F: CJK Symbols and Punctuation
# - U+FF61 - U+FF64: Halfwidth and Fullwidth Forms
cjk = [
    (0x4E00, 0x9FFF),
    (0x3000, 0x303F),
    (0xFF61, 0xFF64),
]
# Both Ukrainian and Russian use the 33-letter Cyrillic alphabet, but there are
# letters in Ukrainian missing from Russian (ґ, є, і, ї, ‘), and
# letters in Russian missing from Ukrainian (ё, ъ, ы, э).
# Belarusian use 32-letter Cyrillic alphabet and have
# letters missing from Ukrainian(ё, ы, э) and Russian (і, ‘) and have
# 1 letter which Russian and Ukrainian don’t have (ў).
# russian:
#  -add ё (U+0401, U+0451), ъ (U+042A, U+044A), ы (U+042B, U+044B), э (U+042D, U+044D)
#  -sub ґ (U+0490, U+0491), є (U+0404, U+0404), і (U+0406, U+0456), ї (U+0407, U+0457), ‘
# ukrainian:
#  -add: ґ, є, і, ї, ‘
#  -sub: ё, ъ, ы, э
# belarusian:
#  -add: ё, ы, э, і, ‘, ў (U+040E, U+045E)

# - U+0400 - U+04FF: Cyrillic
# - U+0500 - U+052F: Cyrillic Supplement
# - U+2DE0 - U+2DFF: Cyrillic Extended-A
# - U+A640 - U+A69F: Cyrillic Extended-B
# - U+1C80 - U+1C8F: Cyrillic Extended-C
# - U+1E030 - U+1E08F: Cyrillic Extended-D
cyrillic = [
    (0x0400, 0x04FF),
    (0x0500, 0x052F),
    (0x2DE0, 0x2DFF),
    (0xA640, 0xA69F),
    (0x1C80, 0x1C8F),
    (0x1E030, 0x1E08F),
]

# https://en.wikipedia.org/wiki/Devanagari_(Unicode_block)
# https://character-table.netlify.app/hindi/
# - U+0900 - U+097F	Devanagari
# - U+A8E0 - U+A8FF	Devanagari Extended
# - U+11B00 - U+11B5F	Devanagari Extended-A
devanagari = [
    (0x0900, 0x097F),
    (0xA8E0, 0xA8FF),
    (0x11B00, 0x11B5F),
]

# - U+0370 - U+03E1	Greek
# - U+03F0 - U+03FF	Greek
# - U+1F00 - U+1FFF	Greek Extended
greek = [
    (0x0370, 0x03E1),
    (0x03F0, 0x03FF),
    (0x1F00, 0x1FFF),
]

# - U+4E00 – U+9FFF: Han
# - U+3400 – U+4DBF
# - U+F900 – U+FAAD
han = [
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0xF900, 0xFAFF)
]

# - U+AC00 - U+D7AF: Hangul Syllables
# - U+1100 - U+11FF: Hangul Jamo
# - U+A960 - U+A97F: Hangul Jamo Extended-A
# - U+D7B0 - U+D7FF: Hangul Jamo Extended-B
# - U+3130 - U+318F: Hangul Compatibility Jamo
# - U+FFA0 - U+FFDF: Halfwidth and Fullwidth Forms
hangul = [
    (0xAC00, 0xD7AF),
    (0x1100, 0x11FF),
    (0xA960, 0xA97F),
    (0xD7B0, 0xD7FF),
    (0x3130, 0x318F),
    (0xFFA0, 0xFFDF),
]

# - U+0590 – U+305FF): Hebrew
hebrew = [(0x0590, 0x05FF)]

# - U+3040 – U+309F: Hiragana
hiragana = [(0x3040, 0x309F)]

# https://en.wikipedia.org/wiki/List_of_Unicode_characters#Brahmic_(Indic)_scripts
# http://unicode.org/charts/PDF/UA830.pdf
# - U+0900 - U+0DFF: Indic scripts
# - U+A830 - U+A83F: Common Indic Number Forms
indic = [
    (0x0900, 0x0DFF),
    (0xA830, 0xA83F),
]
# - U+30A0 – U+30FF: Katakana
# - U+FF65 – U+FF9F: Halfwidth Katakana
katakana = [
    (0x30A0, 0x30FF),
    (0x31F0, 0x31FF),
    (0xFF65, 0xFF9F),
]

# - U+0000 – U+007F: Latin alphabet
# - U+FF21 – U+FF3A: Fullwidth Latin alphabet
latin = [
    (0x0041, 0x005A),
    (0x0061, 0x007A),
    (0xFF21, 0xFF3A),
    (0xFF41, 0xFF5A),
]

# - U+0000 – U+0040: Basic Latin
# - U+005B – U+0060: Punctuation
# - U+007B – U+007F: Punctuation
# - U+0080 – U+00BF: Latin-1 Supplement
# - U+2000 – U+206F: General Punctuation
# - U+FF01 – U+FF20: Halfwidth and Fullwidth Forms
# - U+FF3B – U+FF40: Halfwidth and Fullwidth Forms
# - U+FF5B – U+FF65: Halfwidth and Fullwidth Forms
# - U+FFE0 – U+FFEE: Halfwidth and Fullwidth Forms
punctuation = [
    (0x0000, 0x0040),
    (0x005B, 0x0060),
    (0x007B, 0x007F),
    (0x0080, 0x00BF),
    (0x2000, 0x206F),
    (0xFF01, 0xFF20),
    (0xFF3B, 0xFF40),
    (0xFF5B, 0xFF65),
    (0xFFE0, 0xFFEE),
]

# https://character-table.netlify.app/russian/
# only russian:
#   ё (U+0401, U+0451),
#   ъ (U+042A, U+044A),
#   ы (U+042B, U+044B),
#   э (U+042D, U+044D),
only_russian = [
    (0x0401, 0x0401),
    (0x042A, 0x042B),
    (0x042D, 0x042D),
    (0x044A, 0x044B),
    (0x044D, 0x044D),
    (0x0451, 0x0451),
]

russian = [
    (0x0401, 0x0401),
    (0x0410, 0x044F),
    (0x0451, 0x0451),
]


thai = [
    (0x0E00, 0x0E7F),
]

# https://en.wikipedia.org/wiki/Turkish_alphabet
# https://character-table.netlify.app/turkish/
#
#  Aa-Pp (U+0041, U+0050), (U+0061, U+0070) Part of the latin characters
#  Rr-Vv (U+0052, U+0056), (U+0072, U+0076)
#  Yy-Zz (U+0059, U+005A), (U+0079, U+007A)
#  Çç (U+00C7, U+00E7)
#  Ğğ (U+011E, U+011F)
#  İ (U+0130) dotted 'i', uppercase
#  ı (U+0131) dotless 'i', lower case
#  Öö (U+00D6, U+00F6)
#  Şş (U+015E, U+015F)
#  Üü (U+00DC, U+00FC)

turkish = [
    (0x0041, 0x0050),
    (0x0061, 0x0070),
    (0x0052, 0x0056),
    (0x0072, 0x0076),
    (0x0059, 0x005A),
    (0x0079, 0x007A),
    (0x00C7, 0x00C7),
    (0x00E7, 0x00E7),
    (0x011E, 0x011F),
    (0x0130, 0x0131),
    (0x00D6, 0x00D6),
    (0x00F6, 0x00F6),
    (0x015E, 0x015F),
    (0x00DC, 0x00DC),
    (0x00FC, 0x00FC),
]

# https://en.wikipedia.org/wiki/Ukrainian_alphabet
# https://character-table.netlify.app/ukrainian/
# only ukrainian:
#   є (U+0404, U+0454),
#   і (U+0406, U+0456), Wrong, also used in Belarusian
#   ї (U+0407, U+0457),
#   ґ (U+0490, U+0491)
only_ukrainian = [
    (0x0404, 0x0404),
    (0x0407, 0x0407),
    (0x0454, 0x0454),
    (0x0457, 0x0457),
    (0x0490, 0x0491),
]

ukrainian = [
    (0x0404, 0x0404),
    (0x0406, 0x0407),
    (0x0410, 0x0429),
    (0x042C, 0x042C),
    (0x042E, 0x0449),
    (0x044C, 0x044C),
    (0x044E, 0x044F),
    (0x0454, 0x0454),
    (0x0456, 0x0457),
    (0x0490, 0x0491),
]


# =============================================================================
def isIn(rune, list):
    """Detects if a particular rune is in a list of ranges.

    Args:
        rune (character): Rune to be tested
        list (List of character code ranges): List to test rune against

    Returns:
        Boolean: True if the rune is in the list, False otherwise
    """
    ordinal = ord(rune)
    for lower, upper in list:
        if ordinal >= lower and ordinal <= upper:
            return True
    return False


def wordIsIn(word, list):
    """Detects if a particular word is in a list of ranges.

    Args:
        word (string): Word to be tested
        list (List of character code ranges): List to test word against

    Returns:
        Boolean: True if the word is in the list, False otherwise
    """
    for rune in word:
        if not isIn(rune, list):
            return False
    return True


# -----------------------------------------------------------------------------
# Dominant script classifier, good enough for bucketing
# We map each char to a coarse script; then take the majority over letters.
_SCRIPT_RANGES = [
    ("Latin", latin),
    ("CJK", cjk),
    ("Cyrillic", cyrillic),
    ("Greek", greek),
    ("Han", han),
    ("Hiragana", hiragana),
    ("Katakana", katakana),
    ("Hangul", hangul),
    ("Arabic", arabic),
    ("Hebrew", hebrew),
    ("Devanagari", devanagari),
    ("Thai", thai),
    ("Turkish", turkish)
]


# -----------------------------------------------------------------------------
def _script_of_char(cp: int) -> Optional[str]:
    for script_name, ranges in _SCRIPT_RANGES:
        for lo, hi in ranges:
            if lo <= cp <= hi:
                return script_name
    # Other scripts lumped as 'Other'
    return None


# -----------------------------------------------------------------------------
def dominant_script(s: str) -> str:
    counts = Counter()
    for ch in s:
        if ch.isalpha():
            sc = _script_of_char(ord(ch))
            if sc:
                counts[sc] += 1
    if not counts:
        return "Other"
    # Merge Japanese into "Han" if mixed with Kana? Keep separate for clarity here.
    script, _ = counts.most_common(1)[0]
    return script


