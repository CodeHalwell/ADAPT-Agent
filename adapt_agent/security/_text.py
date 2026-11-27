"""Text normalisation and log escaping shared by the firewall and the detector.

Both controls match patterns against untrusted text and both store snippets of
it, so both need the same two answers: what a string *means* once every
spelling trick is folded away, and how to store a piece of it without letting
it rewrite the log it lands in. Two implementations of either answer drift --
the detector folded four invisible code points while the firewall folded none
-- so they live here once.

Nothing in this module raises on any ``str``: it runs on attacker text in the
hot path of every request.
"""

from __future__ import annotations

import re
import unicodedata

#: Control characters a reader never sees, removed from the copy that is
#: matched. Only the ones that are *neither* whitespace nor a line boundary:
#: ``\t``, ``\n``, ``\x0b``, ``\x0c``, ``\r``, ``\x1c``-``\x1f`` and ``\x85``
#: are structure (whitespace or line breaks) and the callers decide what to do
#: with those. ``ig\x1bnore`` renders as ``ignore`` in most viewers and a NUL
#: renders as nothing, so leaving either in place split a keyword invisibly.
_INVISIBLE_CONTROL_RE = re.compile("[\x00-\x08\x0e-\x1b\x7f-\x84\x86-\x9f]")

#: Default-ignorable letters: Unicode files these under ``Lo`` rather than a
#: format category, but they render as nothing -- the Hangul fillers are the
#: classic blank-name trick.
_IGNORABLE_LETTERS = frozenset({0x115F, 0x1160, 0x3164, 0xFFA0})

#: The tag block, U+E0000-U+E007F, and the variation selectors supplement
#: beyond it. Every code point in U+E0000-U+E0FFF is default-ignorable.
_TAG_PLANE_START, _TAG_PLANE_END = 0xE0000, 0xE0FFF
#: Tag characters that shadow printable ASCII one-for-one: U+E0041 is a tag
#: ``A``. A run of them is invisible in every renderer and spells a whole
#: hidden sentence to a model that reads code points.
_TAG_ASCII_RE = re.compile("[\U000e0020-\U000e007e]")
_TAG_ASCII_OFFSET = 0xE0000


def _is_invisible(char: str) -> bool:
    """Whether ``char`` is presentation a reader cannot see.

    Every format character (``Cf``: soft hyphen, word joiner, the bidi marks
    and overrides, zero-width space and joiners, the tag block), every
    combining mark (``Mn``/``Me``: a combining grapheme joiner, the variation
    selectors, an accent stacked on a letter), and the default-ignorable
    letters. A rule by category rather than a list, because the list held four
    code points and the other 2,164 each split a keyword unseen.
    """
    code = ord(char)
    if _TAG_PLANE_START <= code <= _TAG_PLANE_END or code in _IGNORABLE_LETTERS:
        return True
    return unicodedata.category(char) in ("Cf", "Mn", "Me")


#: Latin look-alikes in Cyrillic and Greek, mapped to the Latin letter a
#: reader takes them for. Deliberately small and reviewable: only glyphs that
#: are indistinguishable from the Latin letter in ordinary fonts, both cases.
#: It is applied only inside a *mixed-script* word -- see :func:`_fold_word` --
#: so genuine Russian or Greek text is never rewritten.
_HOMOGLYPHS = str.maketrans(
    {
        # Cyrillic lower case
        "\u0430": "a",
        "\u0441": "c",
        "\u0501": "d",
        "\u0435": "e",
        "\u04bb": "h",
        "\u0456": "i",
        "\u0458": "j",
        "\u043a": "k",
        "\u043e": "o",
        "\u0440": "p",
        "\u051b": "q",
        "\u0455": "s",
        "\u051d": "w",
        "\u0445": "x",
        "\u0443": "y",
        "\u04af": "y",
        # Cyrillic upper case
        "\u0410": "A",
        "\u0412": "B",
        "\u0421": "C",
        "\u0415": "E",
        "\u04ba": "H",
        "\u041d": "H",
        "\u0406": "I",
        "\u0408": "J",
        "\u041a": "K",
        "\u041c": "M",
        "\u041e": "O",
        "\u0420": "P",
        "\u051a": "Q",
        "\u0405": "S",
        "\u0422": "T",
        "\u051c": "W",
        "\u0425": "X",
        "\u0423": "Y",
        # Greek lower case
        "\u03b1": "a",
        "\u03b9": "i",
        "\u03ba": "k",
        "\u03bd": "v",
        "\u03bf": "o",
        "\u03c1": "p",
        "\u03c5": "u",
        "\u03c7": "x",
        # Greek upper case
        "\u0391": "A",
        "\u0392": "B",
        "\u0395": "E",
        "\u0397": "H",
        "\u0399": "I",
        "\u039a": "K",
        "\u039c": "M",
        "\u039d": "N",
        "\u039f": "O",
        "\u03a1": "P",
        "\u03a4": "T",
        "\u03a5": "Y",
        "\u03a7": "X",
        "\u0396": "Z",
    }
)
_HOMOGLYPH_CHARS = "".join(chr(code) for code in _HOMOGLYPHS)
_HOMOGLYPH_RE = re.compile(f"[{_HOMOGLYPH_CHARS}]")
#: A word: letters only, so a digit or an underscore ends it and the
#: mixed-script test is per word as read. Deliberately *not* one pattern for
#: "a word that contains a look-alike": ``[^\W\d_]*[X][^\W\d_]*`` backtracks
#: across the whole of a long word that has none, from every start position,
#: and 48,000 letters took most of a minute.
_WORD_RE = re.compile(r"[^\W\d_]+")
_ASCII_LETTER_RE = re.compile("[A-Za-z]")


def _fold_word(match: re.Match[str]) -> str:
    """Fold the look-alikes in one word, if the word also holds Latin letters.

    ``\u0456gn\u043er\u0435`` mixes Latin ``g``, ``n`` and ``r`` with Cyrillic ``\u0456``, ``\u043e``
    and ``\u0435``: no language spells a word that way, so it is a disguise and
    reads as ``ignore``. ``\u0438\u0433\u043d\u043e\u0440\u0438\u0440\u0443\u0439`` is Russian and stays Russian -- folding
    a whole-script word would turn every Cyrillic sentence into a mixed-script
    jumble and invent matches in it.
    """
    word = match.group()
    if _HOMOGLYPH_RE.search(word) is None or _ASCII_LETTER_RE.search(word) is None:
        return word
    return word.translate(_HOMOGLYPHS)


def fold_for_matching(text: str) -> str:
    """``text`` as a reader sees it, for pattern matching. Case and spacing kept.

    Compatibility forms folded (NFKC: full-width letters, mathematical
    alphanumerics, ligatures), invisible characters removed (see
    :func:`_is_invisible`), combining marks dropped so ``pr\u00e9vious`` and
    ``i\u0307gnore`` read as their bare letters, and Cyrillic/Greek look-alikes
    folded inside mixed-script words.

    The marks are removed from the *decomposed* text and the result recomposed:
    NFKC alone composes ``e`` + U+0301 into ``\u00e9``, which no ASCII pattern
    matches, so stripping after composition missed every accent an attacker
    stacked on a letter. Folding accents also folds genuine accented text --
    ``caf\u00e9`` matches ``cafe`` -- which is the intended direction for a matcher,
    and applies to both sides because a custom signature goes through here too.

    ASCII text skips everything but the control strip: it has nothing to fold,
    and that keeps the cost of the common case to a single C-level scan.
    """
    text = _INVISIBLE_CONTROL_RE.sub("", text)
    if text.isascii():
        return text
    decomposed = unicodedata.normalize("NFKD", text)
    visible = "".join(char for char in decomposed if char.isascii() or not _is_invisible(char))
    composed = unicodedata.normalize("NFKC", visible)
    if _HOMOGLYPH_RE.search(composed) is None:
        return composed
    return _WORD_RE.sub(_fold_word, composed)


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_for_matching(text: str) -> str:
    """:func:`fold_for_matching`, then every whitespace run collapsed to one space.

    What the firewall's ``normalize`` option matches against. Case is kept --
    a blocked pattern states its own case-sensitivity with ``(?i)`` -- and so
    are the ends of the string, so ``^`` and ``$`` keep meaning what the
    pattern's author wrote.
    """
    return _WHITESPACE_RE.sub(" ", fold_for_matching(text))


def decode_tag_block(text: str) -> str:
    """The ASCII a run of Unicode tag characters spells, or ``""``.

    U+E0020-U+E007E shadow printable ASCII one-for-one and render as nothing,
    so ``Summarise this.`` followed by tags can carry an entire instruction no
    reviewer sees. Stripping them (which :func:`fold_for_matching` does) closes
    the *insertion* trick; this recovers the *smuggled sentence* so it can be
    scanned on its own.
    """
    if text.isascii():
        return ""
    return "".join(chr(ord(char) - _TAG_ASCII_OFFSET) for char in _TAG_ASCII_RE.findall(text))


#: Characters a log line must never carry raw: every C0/C1 control (``Cc``,
#: which includes ESC, the vertical tab, form feed, NEL and the file/group/
#: record separators that :meth:`str.splitlines` honours), the line and
#: paragraph separators (``Zl``/``Zp``), and the bidi embeddings, overrides
#: and isolates, which reorder what a terminal shows without changing a byte a
#: grep sees. Escaping only ``\n`` and ``\r`` left nine more line breaks and
#: raw terminal escape sequences in stored snippets.
_LOG_UNSAFE_RE = re.compile(
    "[\x00-\x1f\x7f-\x9f\u2028\u2029"  # Cc, Zl, Zp
    "\u061c\u200e\u200f\u202a-\u202e\u2066-\u2069]"  # bidi marks, embeddings, overrides, isolates
)
_NAMED_ESCAPES = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}


def _escaped(char: str) -> str:
    """How one character is written into a log."""
    named = _NAMED_ESCAPES.get(char)
    if named is not None:
        return named
    if _LOG_UNSAFE_RE.match(char) is None:
        return char
    code = ord(char)
    return f"\\x{code:02x}" if code <= 0xFF else f"\\u{code:04x}"


def escape_for_log(text: str, max_length: int) -> str:
    """``text`` with every log-unsafe character escaped, at most ``max_length`` long.

    Bounded *after* escaping, and never in the middle of an escape: cutting
    ``\\x1b`` to ``\\x1`` would leave a dangling sequence a downstream parser
    reads as something else. The input is sliced to ``max_length`` first --
    escaping never shortens text, so nothing past that point can be kept --
    which bounds the work on a huge input to the size of the result.

    A backslash already in the text is left alone. That makes the escaping
    lossy rather than reversible, which is fine for a snippet that exists to be
    read; what it must never do is let the text start a new log line.
    """
    if max_length <= 0:
        return ""
    bounded = text[:max_length]
    if _LOG_UNSAFE_RE.search(bounded) is None:
        return bounded
    pieces: list[str] = []
    used = 0
    for char in bounded:
        piece = _escaped(char)
        if used + len(piece) > max_length:
            break
        pieces.append(piece)
        used += len(piece)
    return "".join(pieces)


__all__ = [
    "decode_tag_block",
    "escape_for_log",
    "fold_for_matching",
    "normalize_for_matching",
]
