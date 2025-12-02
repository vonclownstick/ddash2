#!/usr/bin/env python3
"""
Quick cleanup script to remove known false positive investigators.

This script directly removes investigators who were identified as false positives
without checking PubMed (to avoid rate limiting).

Usage:
    python cleanup_false_positives.py --dry-run    # Preview
    python cleanup_false_positives.py              # Remove
"""

import os
import sys
import argparse
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./local_storage.db")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Setup database
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Known false positives (from investigation)
FALSE_POSITIVES = [
    # Beth Israel Deaconess Medical Center (not part of MGB)
    ("KESHAVAN_M", "Keshavan, Matcheri", "Beth Israel Deaconess Medical Center"),
    ("KESHAVAN_MS", "Keshavan, Matcheri S", "Beth Israel Deaconess Medical Center"),
    ("TOROUS_J", "Torous, John", "Beth Israel Deaconess Medical Center"),

    # Psychiatry and Neurology joint departments (Neurology, not Psychiatry)
    ("DICKERSON_B", "Dickerson, Bradford", "Neurology - Psychiatry and Neurology dept"),
    ("DICKERSON_BC", "Dickerson, Bradford C", "Neurology - Psychiatry and Neurology dept"),

    # Boston Children's Hospital (separate from adult psychiatry)
    ("GLAHN_D", "Glahn, David", "Boston Children's Hospital"),
    ("GLAHN_DC", "Glahn, David C", "Boston Children's Hospital"),

    # Non-Psychiatry departments at MGH
    ("EDLOW_A", "Edlow, Andrea", "OB/GYN at MGH, not Psychiatry"),
    ("EDLOW_AG", "Edlow, Andrea G", "OB/GYN at MGH, not Psychiatry"),

    # Other institutions
    ("WINHUSEN_J", "Winhusen, John", "University of Cincinnati"),
    ("WINHUSEN_T", "Winhusen, T", "University of Cincinnati"),
    ("WINHUSEN_TJ", "Winhusen, T John", "University of Cincinnati"),
    ("HENNINGER_N", "Henninger, Nils", "University of Massachusetts"),
    ("PRESSMAN_P", "Pressman, Peter", "University of Maine"),
    ("BOLTE_S", "Bolte, Sven", "No clear MGB affiliation"),
]


def cleanup_false_positives(dry_run: bool = True):
    """
    Remove known false positive investigators.
    """
    print("=" * 80)
    print(f"REMOVING FALSE POSITIVE INVESTIGATORS - {'DRY RUN' if dry_run else 'LIVE MODE'}")
    print("=" * 80)

    db = SessionLocal()

    try:
        removed_count = 0
        not_found_count = 0

        for inv_id, inv_name, reason in FALSE_POSITIVES:
            # Check if investigator exists
            result = db.execute(text("""
                SELECT id, name
                FROM investigator_stats_v3
                WHERE id = :inv_id
            """), {"inv_id": inv_id})
            row = result.fetchone()

            if row:
                print(f"\n{'✗ WOULD REMOVE' if dry_run else '✓ REMOVING'}: {row[1]} ({row[0]})")
                print(f"  Reason: {reason}")

                if not dry_run:
                    # Remove investigator
                    db.execute(text("""
                        DELETE FROM investigator_stats_v3
                        WHERE id = :inv_id
                    """), {"inv_id": inv_id})

                    # Remove associated grants
                    db.execute(text("""
                        DELETE FROM grant_details_v3
                        WHERE investigator_id = :inv_id
                    """), {"inv_id": inv_id})

                removed_count += 1
            else:
                print(f"\n⊘ NOT FOUND: {inv_name} ({inv_id})")
                print(f"  (May already be removed or never existed)")
                not_found_count += 1

        if not dry_run:
            db.commit()

        # Summary
        print(f"\n{'=' * 80}")
        print("SUMMARY")
        print("=" * 80)
        print(f"{'Would remove' if dry_run else 'Removed'}: {removed_count} investigators")
        print(f"Not found in database: {not_found_count}")

        if dry_run:
            print(f"\n⚠️  DRY RUN MODE - No changes made to database")
            print(f"    Run without --dry-run to actually remove these investigators")
        else:
            print(f"\n✅ Successfully removed {removed_count} false positive investigators")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        db.rollback()
        raise

    finally:
        db.close()

    print(f"\n{'=' * 80}")
    print("CLEANUP COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove known false positive investigators")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without removing")
    args = parser.parse_args()

    cleanup_false_positives(dry_run=args.dry_run)
