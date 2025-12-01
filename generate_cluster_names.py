#!/usr/bin/env python3
"""
Command-line tool to generate descriptive names for UMAP clusters.

Usage:
    python generate_cluster_names.py

This script:
1. Loads all investigators with cluster assignments
2. For each cluster, aggregates top research themes and populations
3. Uses GPT-4o-mini to generate a concise, descriptive cluster name (1-3 words)
4. Saves cluster names to database as JSON

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
    """Load investigators grouped by cluster with their themes and populations."""
    result = db.execute(text("""
        SELECT cluster_id, name, tech_json, population_json
        FROM investigator_stats_v3
        WHERE cluster_id IS NOT NULL
        AND tech_json IS NOT NULL
        AND population_json IS NOT NULL
    """)).fetchall()

    # Group by cluster
    clusters = {}
    for cluster_id, name, tech_json, pop_json in result:
        if cluster_id not in clusters:
            clusters[cluster_id] = {
                "investigators": [],
                "technologies": [],
                "populations": []
            }

        clusters[cluster_id]["investigators"].append(name)

        # Parse technologies
        try:
            techs = json.loads(tech_json)
            clusters[cluster_id]["technologies"].extend(techs)
        except:
            pass

        # Parse populations
        try:
            pops = json.loads(pop_json)
            clusters[cluster_id]["populations"].extend(pops)
        except:
            pass

    return clusters


def generate_cluster_name(cluster_id, cluster_data):
    """Use GPT to generate a descriptive name for the cluster."""
    # Get top themes and populations
    tech_counter = Counter(cluster_data["technologies"])
    pop_counter = Counter(cluster_data["populations"])

    top_techs = [item for item, count in tech_counter.most_common(10)]
    top_pops = [item for item, count in pop_counter.most_common(10)]

    # Build prompt
    prompt = f"""You are analyzing a cluster of {len(cluster_data['investigators'])} biomedical researchers.

Top research technologies/methods:
{', '.join(top_techs[:8])}

Top study populations:
{', '.join(top_pops[:8])}

Generate a concise, descriptive name (1-3 words) for this research cluster that captures the main theme.
Examples of good cluster names:
- "Neuroimaging"
- "Cancer Genomics"
- "Pediatric Mental Health"
- "Cardiovascular Epidemiology"

Respond with ONLY the cluster name, nothing else."""

    logger.info(f"🤖 Generating name for Cluster {cluster_id}...")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a research classification expert. Generate concise, accurate cluster names."},
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=50
    )

    cluster_name = response.choices[0].message.content.strip().strip('"').strip("'")

    logger.info(f"   ✅ Cluster {cluster_id}: '{cluster_name}'")
    logger.info(f"      ({len(cluster_data['investigators'])} investigators)")

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
