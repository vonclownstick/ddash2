#!/usr/bin/env python3
"""
Cleanup script to remove investigators who no longer meet affiliation criteria.

Usage:
    python cleanup_database.py --dry-run    # Preview what would be removed
    python cleanup_database.py              # Actually remove them
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
from Bio import Entrez
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./local_storage.db")
Entrez.email = os.getenv("ENTREZ_EMAIL", "test@example.com")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Setup database
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def determine_hospital(affiliation_text: str, department: str = "psychiatry"):
    """
    Updated version with exclusions - same as in ddash2.py
    """
    text = affiliation_text.lower()
    dept_keywords = {
        "psychiatry": ['psychiatry', 'psychiatric'],
    }

    # EXCLUSIONS
    exclusions = [
        'beth israel',
        'children\'s hospital',
        'childrens hospital',
        'boston children',
        'university of massachusetts',
        'umass'
    ]

    for exclusion in exclusions:
        if exclusion in text:
            return None

    if department not in dept_keywords:
        return None

    if not any(keyword in text for keyword in dept_keywords[department]):
        return None

    # Exclude "Psychiatry and Neurology" joint departments
    if 'psychiatry and neurology' in text or 'psychiatry & neurology' in text:
        return None

    # Check which hospital
    if 'brigham' in text and 'women' in text:
        return 'BWH'
    if 'mass' in text and 'general' in text and 'hosp' in text:
        return 'MGH'
    if 'massachusetts general hospital' in text:
        return 'MGH'
    if 'mclean' in text:
        return 'McLean'
    if 'mgb' in text or 'mass general brigham' in text:
        return 'MGB'

    return None


def check_investigator_validity(inv_id: str, inv_name: str, department: str = "psychiatry") -> tuple[bool, str]:
    """
    Check if an investigator should be kept in the database.

    Returns:
        (is_valid, reason)
    """
    # Parse name for PubMed search
    parts = inv_name.replace(",", "").split()
    if len(parts) < 2:
        return False, "Invalid name format"

    last_name = parts[0]
    first_name = parts[1] if len(parts) > 1 else ""

    # Search for recent papers (past 12 months)
    one_year_ago = datetime.now() - timedelta(days=365)
    query = (
        f'{last_name} {first_name}[Author] AND '
        f'"{one_year_ago.strftime("%Y/%m/%d")}"[Date - Publication] : "{datetime.now().strftime("%Y/%m/%d")}"[Date - Publication]'
    )

    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=10)
        record = Entrez.read(handle)
        handle.close()
        pmids = record["IdList"]

        if not pmids:
            return False, "No recent papers found"

        # Fetch paper details
        fetch_handle = Entrez.efetch(db="pubmed", id=pmids[:5], retmode="xml")
        papers = Entrez.read(fetch_handle)
        fetch_handle.close()

        valid_affiliation_count = 0
        total_papers_checked = 0

        for article in papers['PubmedArticle']:
            medline = article['MedlineCitation']
            art = medline['Article']

            if 'AuthorList' not in art:
                continue

            # Find this author on the paper
            for auth in art['AuthorList']:
                auth_last = auth.get('LastName', '').upper()
                if auth_last != last_name.upper():
                    continue

                total_papers_checked += 1

                # Check affiliations
                if 'AffiliationInfo' in auth:
                    for affil_info in auth['AffiliationInfo']:
                        affiliation = affil_info.get('Affiliation', '')
                        hosp = determine_hospital(affiliation, department)
                        if hosp:
                            valid_affiliation_count += 1
                            break
                break

        if total_papers_checked == 0:
            return False, "Name not found in recent papers"

        if valid_affiliation_count == 0:
            return False, f"No valid MGB {department} affiliations in {total_papers_checked} recent papers"

        # Keep if at least 1 valid affiliation found
        return True, f"Valid: {valid_affiliation_count}/{total_papers_checked} papers with MGB affiliation"

    except Exception as e:
        return False, f"Error checking PubMed: {e}"


def cleanup_database(dry_run: bool = True, department: str = "psychiatry"):
    """
    Remove investigators who no longer meet affiliation criteria.
    """
    print("=" * 80)
    print(f"DATABASE CLEANUP - {'DRY RUN' if dry_run else 'LIVE MODE'}")
    print(f"Department: {department}")
    print("=" * 80)

    db = SessionLocal()

    try:
        # Get all investigators
        result = db.execute("""
            SELECT id, name
            FROM investigator_stats_v3
            ORDER BY name
        """)
        investigators = result.fetchall()

        print(f"\nFound {len(investigators)} investigators in database")
        print(f"\nChecking each investigator's recent publications...\n")

        to_remove = []
        to_keep = []

        for i, (inv_id, inv_name) in enumerate(investigators, 1):
            print(f"[{i}/{len(investigators)}] Checking: {inv_name} ({inv_id})")

            is_valid, reason = check_investigator_validity(inv_id, inv_name, department)

            if is_valid:
                to_keep.append((inv_id, inv_name, reason))
                print(f"  ✓ KEEP: {reason}")
            else:
                to_remove.append((inv_id, inv_name, reason))
                print(f"  ✗ REMOVE: {reason}")

            # Rate limiting
            if i % 10 == 0:
                import time
                time.sleep(1)

        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total investigators: {len(investigators)}")
        print(f"To keep: {len(to_keep)}")
        print(f"To remove: {len(to_remove)}")

        if to_remove:
            print(f"\n{'=' * 80}")
            print("INVESTIGATORS TO REMOVE:")
            print("=" * 80)
            for inv_id, inv_name, reason in to_remove:
                print(f"  • {inv_name} ({inv_id})")
                print(f"    Reason: {reason}")

            if not dry_run:
                print(f"\n{'=' * 80}")
                print("REMOVING FROM DATABASE...")
                print("=" * 80)

                for inv_id, inv_name, reason in to_remove:
                    # Remove investigator
                    db.execute("""
                        DELETE FROM investigator_stats_v3
                        WHERE id = :inv_id
                    """, {"inv_id": inv_id})

                    # Remove associated grants
                    db.execute("""
                        DELETE FROM grant_details_v3
                        WHERE investigator_id = :inv_id
                    """, {"inv_id": inv_id})

                    print(f"  ✓ Removed: {inv_name}")

                db.commit()
                print(f"\n✅ Successfully removed {len(to_remove)} investigators")
            else:
                print(f"\n⚠️  DRY RUN MODE - No changes made to database")
                print(f"    Run without --dry-run to actually remove these investigators")

        else:
            print("\n✅ No investigators need to be removed!")

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
    parser = argparse.ArgumentParser(description="Cleanup database by removing invalid investigators")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without removing")
    parser.add_argument("--department", default="psychiatry", help="Department to check (default: psychiatry)")
    args = parser.parse_args()

    cleanup_database(dry_run=args.dry_run, department=args.department)
