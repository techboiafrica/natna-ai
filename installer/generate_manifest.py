#!/usr/bin/env python3
"""
Generate manifest.json with SHA-256 checksums for all NATNA database files.

Run from the repo root:
    python installer/generate_manifest.py /path/to/SACRED

Produces installer/manifest.json
"""

import hashlib
import json
import os
import sys
from pathlib import Path


def sha256_file(filepath, chunk_size=1 << 20):
    """Compute SHA-256 of a file in 1 MB chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# Files to include in manifest, grouped by destination directory.
# Keys are relative paths under the SACRED dir on the source drive.
# Values become the relative install path.
DATABASE_FILES = {
    # Wikipedia / knowledge databases
    "educational_archive/knowledge/massive_wikipedia.db": "educational_archive/knowledge/massive_wikipedia.db",
    "educational_archive/knowledge/medical_wikipedia.db": "educational_archive/knowledge/medical_wikipedia.db",
    "educational_archive/knowledge/k12_education_wikipedia.db": "educational_archive/knowledge/k12_education_wikipedia.db",
    "educational_archive/knowledge/college_education_wikipedia.db": "educational_archive/knowledge/college_education_wikipedia.db",
    "educational_archive/knowledge/mathematics_wikipedia.db": "educational_archive/knowledge/mathematics_wikipedia.db",
    "educational_archive/knowledge/agriculture_wikipedia.db": "educational_archive/knowledge/agriculture_wikipedia.db",
    "educational_archive/knowledge/technical_education_wikipedia.db": "educational_archive/knowledge/technical_education_wikipedia.db",
    "educational_archive/knowledge/technical_enhanced_wikipedia.db": "educational_archive/knowledge/technical_enhanced_wikipedia.db",
    "educational_archive/knowledge/african_arab_world_massive_wikipedia.db": "educational_archive/knowledge/african_arab_world_massive_wikipedia.db",
    # Translation databases
    "organized_data/databases/massive_tigrinya_database.db": "organized_data/databases/massive_tigrinya_database.db",
    "organized_data/databases/academic_translations_v2.db": "organized_data/databases/academic_translations_v2.db",
    "organized_data/databases/cultural_translations_v2.db": "organized_data/databases/cultural_translations_v2.db",
    "organized_data/databases/academic_translations.db": "organized_data/databases/academic_translations.db",
    "organized_data/databases/cultural_translations.db": "organized_data/databases/cultural_translations.db",
    "organized_data/databases/word_dictionary_v2.db": "organized_data/databases/word_dictionary_v2.db",
    "organized_data/databases/hot_cache_translations_v2.db": "organized_data/databases/hot_cache_translations_v2.db",
    "organized_data/databases/hot_cache_translations.db": "organized_data/databases/hot_cache_translations.db",
    # Curriculum
    "organized_data/curriculum_cache/curriculum_database.db": "organized_data/curriculum_cache/curriculum_database.db",
}


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_manifest.py /path/to/SACRED")
        print("  e.g. python generate_manifest.py '/Volumes/Natna v1/NATNA AI/SACRED'")
        sys.exit(1)

    sacred_dir = Path(sys.argv[1])
    if not sacred_dir.is_dir():
        print(f"Error: {sacred_dir} is not a directory")
        sys.exit(1)

    manifest = {"version": 1, "files": {}}

    for src_rel, dest_rel in DATABASE_FILES.items():
        filepath = sacred_dir / src_rel
        if not filepath.exists():
            print(f"  SKIP (not found): {src_rel}")
            continue

        size = filepath.stat().st_size
        if size <= 1:
            print(f"  SKIP (empty placeholder): {src_rel}")
            continue

        print(f"  Hashing {src_rel} ({size / 1_048_576:.1f} MB)...")
        checksum = sha256_file(filepath)

        manifest["files"][dest_rel] = {
            "sha256": checksum,
            "size": size,
        }

    # Write manifest
    out_path = Path(__file__).parent / "manifest.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote {out_path} with {len(manifest['files'])} files.")


if __name__ == "__main__":
    main()
