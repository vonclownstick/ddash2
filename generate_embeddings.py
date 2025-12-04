#!/usr/bin/env python3
"""
Command-line tool to generate embeddings for all investigators in the database.

Usage:
    python generate_embeddings.py

This script:
1. Loads all investigators from the database
2. For each investigator, builds a profile_text from:
   - Research themes
   - Primary populations
   - Recent publication titles (last 2 years)
   - Recent grant titles (last 2 years)
3. Computes an impact_score from RCR and funding
4. Generates an embedding using OpenAI text-embedding-3-large
5. Saves profile_text, embedding_json, and impact_score to the database

Environment variables needed:
- OPENAI_API_KEY
- DATABASE_URL (optional, defaults to sqlite)
"""

import os
import sys
import json
import math
import logging
from datetime import datetime, timedelta
from typing import List, Dict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Import from main app
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import openai

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./local_storage.db")

if not OPENAI_API_KEY:
    logger.error("❌ OPENAI_API_KEY not set!")
    sys.exit(1)

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Setup database
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
client = openai.OpenAI(api_key=OPENAI_API_KEY)

logger.info("✅ Connected to database and OpenAI")

# AIDEV-NOTE: Run database migration to add embedding columns if they don't exist
def check_and_migrate_db():
    """Add embedding-related columns if they don't exist"""
    if "sqlite" not in DATABASE_URL:
        logger.warning("⚠️  Auto-migration only works with SQLite. For PostgreSQL, run ddash2.py first.")
        return

    table_name = "investigator_stats_v3"
    with engine.connect() as conn:
        try:
            result = conn.execute(text(f"PRAGMA table_info({table_name})"))
            existing_cols = [row[1] for row in result.fetchall()]

            migrations_needed = []
            # Legacy columns (backward compatibility)
            if "profile_text" not in existing_cols:
                migrations_needed.append("profile_text TEXT DEFAULT ''")
            if "embedding_json" not in existing_cols:
                migrations_needed.append("embedding_json TEXT DEFAULT 'null'")
            if "impact_score" not in existing_cols:
                migrations_needed.append("impact_score REAL DEFAULT 0.0")

            # New dual-embedding columns
            if "embedding_themes_pops" not in existing_cols:
                migrations_needed.append("embedding_themes_pops TEXT DEFAULT 'null'")
            if "embedding_titles" not in existing_cols:
                migrations_needed.append("embedding_titles TEXT DEFAULT 'null'")

            if migrations_needed:
                logger.info(f"🔧 Running database migrations...")
                for migration in migrations_needed:
                    col_name = migration.split()[0]
                    logger.info(f"   Adding column: {col_name}")
                    conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {migration}"))
                conn.commit()
                logger.info(f"✅ Database migrations complete")
            else:
                logger.info("✅ Database schema up to date")

        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            logger.error(f"   Please run ddash2.py first to initialize the database")
            raise

# Run migration before proceeding
check_and_migrate_db()


def build_themes_pops_text(inv_data: Dict) -> str:
    """
    Build profile text from themes and populations only.
    This embedding will be supplementary (20%) in search ranking.

    Args:
        inv_data: Dict with keys: themes, populations

    Returns:
        String profile suitable for embedding
    """
    parts = []

    # Add themes (technologies/methods)
    themes = inv_data.get("themes", [])
    if themes:
        parts.append(f"Research technologies and methods: {', '.join(themes)}")

    # Add populations
    pops = inv_data.get("populations", [])
    if pops:
        parts.append(f"Study populations and cohorts: {', '.join(pops)}")

    return "\n\n".join(parts) if parts else "No themes or populations available"


def build_titles_text(inv_data: Dict) -> str:
    """
    Build profile text from publication and grant titles.
    This embedding will be prioritized (80%) in search ranking.

    Args:
        inv_data: Dict with keys: publications, grants

    Returns:
        String profile suitable for embedding
    """
    parts = []

    # Add publication titles (top 20 by RCR)
    pubs = inv_data.get("publications", [])
    if pubs:
        # Sort by RCR descending to get most impactful papers
        pubs_sorted = sorted(pubs, key=lambda x: x.get("rcr", 0), reverse=True)
        top_pubs = pubs_sorted[:20]
        pub_titles = [p.get("title", "") for p in top_pubs if p.get("title")]
        if pub_titles:
            parts.append(f"Publication titles: {' | '.join(pub_titles)}")

    # Add grant titles (last 2 years)
    grants = inv_data.get("grants", [])
    if grants:
        grant_titles = [g.get("title", "") for g in grants if g.get("title")]
        if grant_titles:
            parts.append(f"Grant titles: {' | '.join(grant_titles)}")

    return "\n\n".join(parts) if parts else "No publication or grant titles available"


def load_investigator_data(db, inv_id: str) -> Dict:
    """Load all data for one investigator"""
    # Get investigator
    inv_result = db.execute(text("""
        SELECT id, name, themes_json, tech_json, population_json, pmids_p2_json
        FROM investigator_stats_v3
        WHERE id = :inv_id
    """), {"inv_id": inv_id}).fetchone()

    if not inv_result:
        return None

    inv_id, name, themes_json, tech_json, population_json, pmids_p2_json = inv_result

    themes = json.loads(tech_json) if tech_json else []
    populations = json.loads(population_json) if population_json else []
    pmids = json.loads(pmids_p2_json) if pmids_p2_json else []

    # Get publications from last 2 years
    two_years_ago = datetime.now() - timedelta(days=730)
    publications = []

    if pmids:
        # AIDEV-NOTE: Build IN clause manually to avoid SQLAlchemy parameter binding issues
        pmid_placeholders = ','.join([f"'{pmid}'" for pmid in pmids])
        query = f"""
            SELECT pmid, title, pub_date, rcr
            FROM paper_details_v3
            WHERE pmid IN ({pmid_placeholders})
            AND pub_date >= :cutoff_date
        """
        pub_results = db.execute(text(query), {"cutoff_date": two_years_ago}).fetchall()

        for pmid, title, pub_date, rcr in pub_results:
            publications.append({
                "pmid": pmid,
                "title": title,
                "pub_date": pub_date,
                "rcr": rcr or 0.0
            })

    # Get grants from last 2 years
    grants = []
    grant_results = db.execute(text("""
        SELECT project_num, project_title, award_amount, fiscal_year
        FROM grant_details_v3
        WHERE investigator_id = :inv_id
        AND fiscal_year >= :min_year
    """), {"inv_id": inv_id, "min_year": datetime.now().year - 2}).fetchall()

    for project_num, project_title, award_amount, fiscal_year in grant_results:
        grants.append({
            "project_num": project_num,
            "title": project_title,
            "award_amount": award_amount or 0.0,
            "fiscal_year": fiscal_year
        })

    return {
        "id": inv_id,
        "name": name,
        "themes": themes,
        "populations": populations,
        "publications": publications,
        "grants": grants
    }


def generate_embedding(profile_text: str) -> List[float]:
    """Generate embedding using OpenAI"""
    if not profile_text.strip():
        logger.warning("Empty profile text, returning zero vector")
        return [0.0] * 3072  # text-embedding-3-large dimension

    try:
        response = client.embeddings.create(
            input=profile_text,
            model="text-embedding-3-large"
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None


def process_investigator(db, inv_id: str) -> bool:
    """
    Process one investigator: generate two separate embeddings.

    AIDEV-NOTE: Dual-embedding approach for better semantic search
    - Embedding 1 (themes_pops): Research themes + study populations (20% weight)
    - Embedding 2 (titles): Publication + grant titles (80% weight)
    """
    try:
        # Load data
        inv_data = load_investigator_data(db, inv_id)
        if not inv_data:
            logger.warning(f"No data found for investigator {inv_id}")
            return False

        # Build profile texts for both embeddings
        themes_pops_text = build_themes_pops_text(inv_data)
        titles_text = build_titles_text(inv_data)

        logger.info(f"Built profiles for {inv_data['name']} (themes/pops: {len(themes_pops_text)} chars, titles: {len(titles_text)} chars)")

        # Generate both embeddings
        embedding_themes_pops = generate_embedding(themes_pops_text)
        if embedding_themes_pops is None:
            logger.error(f"Failed to generate themes/pops embedding for {inv_id}")
            return False

        embedding_titles = generate_embedding(titles_text)
        if embedding_titles is None:
            logger.error(f"Failed to generate titles embedding for {inv_id}")
            return False

        # Serialize embeddings
        embedding_themes_pops_json = json.dumps(embedding_themes_pops)
        embedding_titles_json = json.dumps(embedding_titles)

        # Update database with both embeddings
        db.execute(text("""
            UPDATE investigator_stats_v3
            SET embedding_themes_pops = :embedding_themes_pops,
                embedding_titles = :embedding_titles
            WHERE id = :inv_id
        """), {
            "embedding_themes_pops": embedding_themes_pops_json,
            "embedding_titles": embedding_titles_json,
            "inv_id": inv_id
        })
        db.commit()

        logger.info(f"✅ {inv_data['name']} - dual embeddings generated")
        return True

    except Exception as e:
        logger.error(f"Error processing {inv_id}: {e}")
        db.rollback()
        return False


def main():
    """Main entry point"""
    logger.info("🚀 Starting embedding generation")

    with SessionLocal() as db:
        # AIDEV-NOTE: Filter out low-activity investigators (>$1M funding but ≤2 current year papers)
        # Get current year for filtering
        current_year = datetime.now().year

        # Get all investigators, excluding those with high funding but low recent output
        result = db.execute(text("""
            SELECT id, funding_past_5y, n_p2
            FROM investigator_stats_v3
        """)).fetchall()

        inv_ids = []
        skipped = 0

        for inv_id, funding_5y, n_p2 in result:
            # Skip if funding > $1M but only 0-2 papers in last 12 months
            if funding_5y > 1000000 and n_p2 <= 2:
                logger.info(f"Skipping {inv_id}: High funding (${funding_5y:,.0f}) but low recent output ({n_p2} papers)")
                skipped += 1
                continue
            inv_ids.append(inv_id)

        logger.info(f"Found {len(inv_ids)} investigators to process ({skipped} skipped)")

        success_count = 0
        fail_count = 0

        for i, inv_id in enumerate(inv_ids, 1):
            logger.info(f"Processing {i}/{len(inv_ids)}: {inv_id}")

            if process_investigator(db, inv_id):
                success_count += 1
            else:
                fail_count += 1

            # Rate limiting: small delay every 10 investigators
            if i % 10 == 0:
                import time
                time.sleep(1)

    logger.info(f"✅ Complete! Success: {success_count}, Failed: {fail_count}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
