#!/usr/bin/env python3
"""
Command-line tool to generate descriptive names for UMAP clusters.

Usage:
    python generate_cluster_names.py

This script:
1. Loads all investigators with cluster assignments
2. For each cluster, aggregates publication titles and grant titles
3. Uses GPT-5-mini to analyze actual research content and generate specific cluster names (1-3 words)
4. Saves cluster names to database as JSON

AIDEV-NOTE: Uses actual publication/grant titles (favoring publications) rather than
AI-generated summaries for more accurate cluster naming.

Requirements:
- OPENAI_API_KEY environment variable set
- Investigators must have cluster_id assignments (run generate_umap.py first)
"""

import os
import sys
import json
import logging
from collections import Counter

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
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./local_storage.db")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    logger.error("❌ OPENAI_API_KEY not set! Please set it first:")
    logger.error("   export OPENAI_API_KEY='sk-...'")
    sys.exit(1)

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Setup database
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Setup OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

logger.info("✅ Connected to database and OpenAI")


def load_cluster_data(db):
    """Load investigators grouped by cluster with their publication and grant titles."""
    result = db.execute(text("""
        SELECT cluster_id, name, pmids_p2_json, id
        FROM investigator_stats_v3
        WHERE cluster_id IS NOT NULL
    """)).fetchall()

    # Group by cluster
    clusters = {}
    for cluster_id, name, pmids_json, inv_id in result:
        if cluster_id not in clusters:
            clusters[cluster_id] = {
                "investigators": [],
                "publication_titles": [],
                "grant_titles": []
            }

        clusters[cluster_id]["investigators"].append(name)

        # Get publication titles (top 10 by RCR from last 2 years)
        if pmids_json:
            try:
                pmids = json.loads(pmids_json)
                if pmids:
                    pmid_list = ', '.join([f"'{p}'" for p in pmids[:50]])  # Limit to avoid huge query
                    pub_result = db.execute(text(f"""
                        SELECT title, rcr
                        FROM paper_details_v3
                        WHERE pmid IN ({pmid_list})
                        AND pub_date >= date('now', '-2 years')
                        ORDER BY rcr DESC
                        LIMIT 10
                    """)).fetchall()

                    for title, rcr in pub_result:
                        if title:
                            clusters[cluster_id]["publication_titles"].append(title)
            except:
                pass

        # Get grant titles (last 2 years)
        try:
            grant_result = db.execute(text("""
                SELECT project_title
                FROM grant_details_v3
                WHERE investigator_id = :inv_id
                AND fiscal_year >= :min_year
                LIMIT 5
            """), {"inv_id": inv_id, "min_year": 2023}).fetchall()

            for (title,) in grant_result:
                if title:
                    clusters[cluster_id]["grant_titles"].append(title)
        except:
            pass

    return clusters


def generate_cluster_name(cluster_id, cluster_data):
    """Use GPT to generate a descriptive name for the cluster based on actual research titles."""
    # Combine publication titles (prioritized) and grant titles
    pub_titles = cluster_data.get("publication_titles", [])
    grant_titles = cluster_data.get("grant_titles", [])

    # Favor publication titles - take up to 30 pub titles, then fill with grants
    all_titles = pub_titles[:30] + grant_titles[:10]

    if not all_titles:
        logger.warning(f"⚠️  No titles found for Cluster {cluster_id}, using generic name")
        return f"Cluster {cluster_id}"

    # Truncate very long titles
    truncated_titles = [t[:150] for t in all_titles[:25]]
    titles_text = "\n- ".join(truncated_titles)

    # Build improved prompt
    prompt = f"""You are analyzing a cluster of {len(cluster_data['investigators'])} psychiatry researchers at Massachusetts General Hospital/Brigham and Women's Hospital/McLean Hospital.

Below are recent publication titles and grant titles from researchers in this cluster:

- {titles_text}

TASK: Generate a concise, descriptive name (1-3 words) that captures the PRIMARY research focus of this cluster.

REQUIREMENTS:
- Be SPECIFIC about the research domain (e.g., "Mood Disorders", not "Mental Health")
- Use psychiatric/neurological terminology when relevant (e.g., "Psychosis", "Addiction", "PTSD")
- Mention methods ONLY if they're the defining feature (e.g., "Neuroimaging Studies")
- Mention populations ONLY if highly specific (e.g., "Pediatric Anxiety", not "Clinical Studies")
- Avoid generic terms like "Research", "Studies", "Analysis"

GOOD EXAMPLES:
- "Psychosis Treatment"
- "Addiction Neuroscience"
- "Mood Disorders"
- "Child Psychiatry"
- "Neuroimaging"
- "PTSD & Trauma"

BAD EXAMPLES:
- "Mental Health Research" (too generic)
- "Clinical Studies" (too vague)
- "Biomedical Research" (not specific)

Respond with ONLY the cluster name (1-3 words), nothing else."""

    logger.info(f"🤖 Generating name for Cluster {cluster_id} ({len(all_titles)} titles)...")

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "You are an expert in psychiatry research classification. Generate highly specific, accurate cluster names based on actual research content."},
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=20,
        temperature=0.3  # Lower temperature for more consistent, focused names
    )

    cluster_name = response.choices[0].message.content.strip().strip('"').strip("'")

    logger.info(f"   ✅ Cluster {cluster_id}: '{cluster_name}'")
    logger.info(f"      ({len(cluster_data['investigators'])} investigators, {len(pub_titles)} pubs, {len(grant_titles)} grants)")

    return cluster_name


def save_cluster_names(db, cluster_names):
    """Save cluster names to database as JSON in a config table."""
    # Create config table if it doesn't exist
    db.execute(text("""
        CREATE TABLE IF NOT EXISTS config (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """))

    # Save cluster names as JSON
    cluster_names_json = json.dumps(cluster_names)

    db.execute(text("""
        INSERT OR REPLACE INTO config (key, value)
        VALUES ('cluster_names', :value)
    """), {"value": cluster_names_json})

    db.commit()

    logger.info("💾 Saved cluster names to database")


def main():
    """Main entry point"""
    logger.info("🚀 Starting cluster name generation")

    with SessionLocal() as db:
        # Load cluster data
        logger.info("📊 Loading cluster data...")
        clusters = load_cluster_data(db)

        if not clusters:
            logger.error("❌ No clusters found! Run 'python generate_umap.py' first")
            sys.exit(1)

        logger.info(f"   Found {len(clusters)} clusters")

        # Generate names for each cluster
        cluster_names = {}
        for cluster_id in sorted(clusters.keys()):
            cluster_name = generate_cluster_name(cluster_id, clusters[cluster_id])
            cluster_names[str(cluster_id)] = cluster_name

        # Save to database
        save_cluster_names(db, cluster_names)

    logger.info("✅ Cluster name generation complete!")
    logger.info("\nCluster Names:")
    for cluster_id, name in sorted(cluster_names.items(), key=lambda x: int(x[0])):
        logger.info(f"   Cluster {cluster_id}: {name}")

    logger.info("\n👉 Restart the app to see cluster names in the 3D visualization")


if __name__ == "__main__":
    main()
