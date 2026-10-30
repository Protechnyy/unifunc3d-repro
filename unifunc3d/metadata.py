"""Explicit, hash-pinned dataset exceptions; release metadata stays immutable."""
from release_lib.common import ROOT, read_json


def benchmark_metadata(split, overrides=None):
    metadata = read_json(ROOT / 'datasets' / f'{split}.json')
    exceptions = read_json(overrides) if overrides else {}
    for visit, fields in exceptions.items():
        if visit not in metadata or set(fields) != {'descriptions_sha256'}:
            raise ValueError('Only known-scene description hashes may be overridden')
        digest = fields['descriptions_sha256']
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid description SHA256 override')
        metadata[visit].update(fields)
    return metadata, exceptions
