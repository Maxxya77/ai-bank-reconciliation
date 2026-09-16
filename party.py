"""Conservative, deterministic comparison of external and GL party names."""
import re


# Keep this small and explicit. A future CSV/JSON table can replace the dictionary.
ALIASES = {
    '한전': '한국전력공사',
    'kepco': '한국전력공사',
    '한국전력서울본부': '한국전력공사',
    '신한카드결제': '신한카드',
    'shinhancard': '신한카드',
    '서울아리수본부': '서울아리수본부',
}


def normalize_name(value):
    name = str(value or '').strip().lower()
    name = re.sub(r'\(\s*주\s*\)|㈜|주식회사', '', name)
    return re.sub(r'[^0-9a-z가-힣]', '', name)


def party_relation(external, gl_party):
    """Return exact, normalized, alias, or none; never infer legal identity."""
    left, right = str(external or '').strip(), str(gl_party or '').strip()
    if not left or not right:
        return 'none'
    if left == right:
        return 'exact'
    a, b = normalize_name(left), normalize_name(right)
    if not a or not b:
        return 'none'
    if a == b:
        return 'normalized'
    if ALIASES.get(a, a) == ALIASES.get(b, b):
        return 'alias'
    return 'none'
