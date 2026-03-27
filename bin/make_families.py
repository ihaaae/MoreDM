"""Rule-based prompt family generator for attribution analysis.

For each "special" prompt, identifies key elements (persons, artists,
moods, suggestive words, mediums) and creates single-element variants
to test which element drives the prompt's specialness under minority
generation.

Each variant changes exactly ONE element category, keeping everything
else intact. This enables ablation-style attribution.
"""

import re
import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Substitution vocabularies (lowercase key -> replacement)
# Longer keys are tried first so "donald trump" beats "trump".
# ---------------------------------------------------------------------------

PERSONS = {
    "donald trump": "william shakespeare",
    "bill clinton": "abraham lincoln",
    "hillary clinton": "ada lovelace",
    "justin trudeau": "charles darwin",
    "justin bieber": "ludwig van beethoven",
    "barack obama": "leonardo da vinci",
    "liz truss": "marie curie",
    "pete davidson": "nikola tesla",
    "ellen degeneres": "jane austen",
    "george soros": "benjamin franklin",
    "jonah hill": "mark twain",
    "joe biden": "isaac newton",
    "robert pattinson": "charles dickens",
    "jesse pinkman": "sherlock holmes",
    "pepe the frog": "a cartoon bear",
    "gus fring": "professor moriarty",
    "mike pence": "thomas edison",
    "stormy daniels": "florence nightingale",
    "obama": "einstein",
    "trump": "shakespeare",
    "clinton": "lincoln",
    "biden": "newton",
    "putin": "galileo",
    "stalin": "socrates",
    "hitler": "napoleon",
    "aphrodite": "athena",
    "pepe": "teddy bear",
}

ARTISTS = {
    "zdislaw beksinski": "claude monet",
    "beksinski": "claude monet",
    "junji ito": "studio ghibli",
    "ralph steadman": "norman rockwell",
    "diane arbus": "ansel adams",
    "louis daguerre": "claude monet",
    "greg rutkowski": "hayao miyazaki",
    "greg ruthkowski": "hayao miyazaki",
    "gaston bussiere": "pierre bonnard",
    "craig mullins": "pierre bonnard",
    "j. c. leyendecker": "pierre auguste renoir",
    "sachin teng": "alphonse mucha",
    "sergey kolesov": "gustav klimt",
    "ruan jia": "john singer sargent",
    "heng z": "edgar degas",
    "john howe": "thomas kinkade",
    "stanly artgerm lau": "alphonse mucha",
    "wlop": "john william waterhouse",
    "rossdraws": "norman rockwell",
    "james jean": "pierre bonnard",
    "andrei riabovitchev": "claude monet",
    "marc simonetti": "thomas cole",
    "john kenn mortensen": "beatrix potter",
    "santiago caruso": "henri matisse",
    "tomer hanuka": "norman rockwell",
    "laurie greasley": "mary cassatt",
    "kentaro mirua": "osamu tezuka",
    "takato yamamoto": "alphonse mucha",
    "vania zouravliov": "aubrey beardsley",
    "zhaoming wu": "john singer sargent",
    "nick alm": "anders zorn",
    "ivan albright": "john singer sargent",
    "edward hopper": "pierre bonnard",
    "ron mueck": "auguste rodin",
    "artgrem": "norman rockwell",
    "rene magritte": "claude monet",
    "anna rubincam": "auguste rodin",
    "vanessa beecroft": "mary cassatt",
    "jenny frison": "mary cassatt",
    "sana takeda": "beatrix potter",
    "wes anderson": "steven spielberg",
    "mucha": "pierre bonnard",
}

MOODS = {
    "death metal": "classical music",
    "political cartoon": "nature illustration",
    "trail cam footage": "studio photography",
    "trail cam": "studio camera",
    "night cam": "daylight camera",
    "album cover": "book illustration",
    "horror": "comedy",
    "creepy": "serene",
    "macabre": "cheerful",
    "disturbing": "pleasant",
    "scary": "friendly",
    "horrifying": "delightful",
    "eldritch": "ordinary",
    "evil": "kind",
    "haunting": "welcoming",
    "violent": "gentle",
    "propaganda": "documentary",
    "distorted": "clear",
    "sad": "happy",
    "cursed": "blessed",
    "ominous": "peaceful",
    "noir": "pastel",
    "apocalypse": "celebration",
    "skull": "flower",
    "skeleton": "bouquet",
    "cult ": "club ",
    "cultists": "tourists",
    "demon": "angel",
    "demons": "angels",
    "ghost": "fairy",
    "ghosts": "fairies",
    "dark": "bright",
    "facehugger": "butterfly",
    "knife": "wand",
    "goblin": "fairy",
}

SUGGESTIVE = {
    "seducing poses": "standing poses",
    "lacivious": "elegant",
    "seducing": "standing",
    "nude": "clothed",
    "naked": "dressed",
    "provocative": "modest",
    "sensual": "formal",
    "erotic": "artistic",
    "cuddling and kissing": "standing and talking",
    "cuddling": "standing",
    "kissing": "waving",
    "shirtless": "suited",
    "bikini": "sundress",
}

MEDIUMS = {
    "detailed photo": "detailed painting",
    "blurry film still": "clear oil painting",
    "film still": "oil painting",
    "photo": "painting",
    "portrait": "landscape painting",
    "painting": "photograph",
    "illustration": "photograph",
    "3d render": "watercolor",
    "poster": "sketch",
    "comic": "photograph",
    "pencil drawing": "watercolor painting",
    "photograph": "oil painting",
}


def _replace_all(text: str, vocab: dict[str, str]) -> tuple[str | None, list[tuple[str, str]]]:
    """Replace all vocabulary matches in text (longest-first).

    Returns (new_text, [(matched, replacement), ...]) or (None, [])
    if nothing matched.
    """
    matches: list[tuple[str, str]] = []
    result = text

    for key in sorted(vocab, key=len, reverse=True):
        pattern = re.compile(r'\b' + re.escape(key) + r'\b', re.IGNORECASE)
        found = pattern.search(result)
        if found:
            original_span = found.group()
            replacement = vocab[key]
            if original_span[0].isupper() and replacement[0].islower():
                replacement = replacement[0].upper() + replacement[1:]
            result = pattern.sub(replacement, result)
            matches.append((original_span, replacement))

    return (result, matches) if matches else (None, [])


# ---------------------------------------------------------------------------
# Per-category detectors
# Each returns (variant_text | None, [(orig, repl), ...])
# ---------------------------------------------------------------------------

def detect_persons(text):
    return _replace_all(text, PERSONS)


def detect_artists(text):
    if not re.search(r'\b(by|style\s+of|painted\s+by)\b', text, re.IGNORECASE):
        return None, []
    return _replace_all(text, ARTISTS)


def detect_moods(text):
    return _replace_all(text, MOODS)


def detect_suggestive(text):
    return _replace_all(text, SUGGESTIVE)


def detect_medium(text):
    for key in sorted(MEDIUMS, key=len, reverse=True):
        pattern = re.compile(r'^(an?\s+)?' + re.escape(key) + r'\b', re.IGNORECASE)
        m = re.match(pattern, text)
        if m:
            original = m.group()
            replacement = MEDIUMS[key]
            if original[0].isupper() and replacement[0].islower():
                replacement = replacement[0].upper() + replacement[1:]
            new_text = replacement + text[m.end():]
            return new_text, [(original, replacement)]
    return None, []


DETECTORS = [
    ("person", detect_persons),
    ("artist", detect_artists),
    ("mood", detect_moods),
    ("suggestive", detect_suggestive),
    ("medium", detect_medium),
]


def make_family(prompt_text: str) -> list[tuple[str, str, str, str]]:
    """Generate single-element variants for a prompt.

    Returns [(variant_text, element_type, originals_csv, replacements_csv), ...].
    """
    variants = []
    for element_type, detector in DETECTORS:
        new_text, matches = detector(prompt_text)
        if new_text and new_text != prompt_text:
            originals = ", ".join(m[0] for m in matches)
            replacements = ", ".join(m[1] for m in matches)
            variants.append((new_text, element_type, originals, replacements))
    return variants


def main():
    parser = argparse.ArgumentParser(description="Generate prompt families for attribution")
    parser.add_argument("--manifest", type=str, required=True,
                        help="Path to special.tsv")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Output directory for families")
    args = parser.parse_args()

    entries: list[tuple[str, str]] = []
    with open(args.manifest, encoding="utf-8") as f:
        f.readline()  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t", 6)
            if len(parts) < 7:
                continue
            sp_id, _ds, _sl, _bu, _mu, _d, prompt = parts
            entries.append((sp_id, prompt))

    for sp_id, prompt_text in entries:
        family_dir = Path(args.outdir) / sp_id
        family_dir.mkdir(parents=True, exist_ok=True)

        variants = make_family(prompt_text)

        with open(family_dir / "family.txt", "w", encoding="utf-8") as f:
            f.write(prompt_text + "\n")
            for var_text, _, _, _ in variants:
                f.write(var_text + "\n")

        with open(family_dir / "manifest.tsv", "w", encoding="utf-8") as f:
            f.write("var_line\telement_type\toriginal_values\tnew_values\n")
            for i, (_, etype, originals, replacements) in enumerate(variants, start=2):
                f.write(f"{i}\t{etype}\t{originals}\t{replacements}\n")

        print(f"{sp_id}: {len(variants)} variant(s) + original -> {family_dir / 'family.txt'}")


if __name__ == "__main__":
    main()
