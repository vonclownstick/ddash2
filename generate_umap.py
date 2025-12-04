#!/usr/bin/env python3
"""
Command-line tool to generate 3D UMAP visualization coordinates and clusters.

Usage:
    python generate_umap.py

This script:
1. Loads all investigators with dual embeddings from the database
2. Combines embeddings (75% themes_pops + 25% titles) into single vector
3. Computes 3D UMAP dimensionality reduction
4. Performs clustering (K-means)
5. Saves UMAP coordinates (x, y, z) and cluster_id to database

Requirements:
- umap-learn
- scikit-learn
- numpy

Install with: pip install umap-learn scikit-learn numpy
"""

import os
import sys
import json
import logging
import numpy as np
from typing import List, Tuple

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

logger.info("✅ Connected to database")

# Check dependencies
try:
    import umap
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    logger.error(f"❌ Missing required package: {e}")
    logger.error("Install with: pip install umap-learn scikit-learn numpy")
    sys.exit(1)


def load_embeddings_from_db() -> Tuple[List[str], np.ndarray]:
    """
    Load all investigators with dual embeddings from database.

    Returns:
        Tuple of (investigator_ids, combined_embeddings_matrix)
    """
    with SessionLocal() as db:
        # Load investigators with both embeddings
        result = db.execute(text("""
            SELECT id, embedding_themes_pops, embedding_titles
            FROM investigator_stats_v3
            WHERE embedding_themes_pops != 'null'
            AND embedding_themes_pops IS NOT NULL
            AND embedding_titles != 'null'
            AND embedding_titles IS NOT NULL
        """)).fetchall()

        if not result:
            logger.error("❌ No investigators with dual embeddings found!")
            logger.error("   Run 'python generate_embeddings.py' first")
            sys.exit(1)

        logger.info(f"📊 Loaded {len(result)} investigators with embeddings")

        investigator_ids = []
        embeddings_list = []

        for inv_id, themes_pops_json, titles_json in result:
            try:
                # Parse embeddings
                themes_pops_emb = np.array(json.loads(themes_pops_json))
                titles_emb = np.array(json.loads(titles_json))

                # Combine with same weighting as search: 20% themes/pops + 80% titles
                combined = 0.2 * themes_pops_emb + 0.8 * titles_emb

                investigator_ids.append(inv_id)
                embeddings_list.append(combined)

            except Exception as e:
                logger.warning(f"Skipping {inv_id}: {e}")
                continue

        if not embeddings_list:
            logger.error("❌ Failed to load any valid embeddings")
            sys.exit(1)

        # Convert to numpy array
        embeddings_matrix = np.array(embeddings_list)
        logger.info(f"✅ Embeddings matrix shape: {embeddings_matrix.shape}")

        return investigator_ids, embeddings_matrix


def compute_umap_3d(embeddings: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    """
    Compute 3D UMAP projection of embeddings.

    Args:
        embeddings: (n_investigators, embedding_dim) array
        n_neighbors: UMAP hyperparameter (controls local vs global structure)
        min_dist: UMAP hyperparameter (controls point spacing)

    Returns:
        (n_investigators, 3) array of 3D coordinates
    """
    logger.info(f"🔧 Computing 3D UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")

    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric='cosine',
        random_state=42,
        verbose=True
    )

    umap_coords = reducer.fit_transform(embeddings)

    logger.info(f"✅ UMAP computed, shape: {umap_coords.shape}")
    return umap_coords


def cluster_investigators(embeddings: np.ndarray, n_clusters: int = 8) -> np.ndarray:
    """
    Cluster investigators using K-means.

    Args:
        embeddings: (n_investigators, embedding_dim) array
        n_clusters: Number of clusters to create

    Returns:
        (n_investigators,) array of cluster IDs
    """
    logger.info(f"🔧 Clustering investigators (k={n_clusters})...")

    # Standardize features before clustering
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(embeddings_scaled)

    logger.info(f"✅ Clustering complete")

    # Show cluster distribution
    unique, counts = np.unique(cluster_ids, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        logger.info(f"   Cluster {cluster_id}: {count} investigators")

    return cluster_ids


def save_to_database(investigator_ids: List[str], umap_coords: np.ndarray, cluster_ids: np.ndarray):
    """Save UMAP coordinates and cluster IDs to database"""
    logger.info(f"💾 Saving UMAP data to database...")

    with SessionLocal() as db:
        for i, inv_id in enumerate(investigator_ids):
            x, y, z = float(umap_coords[i, 0]), float(umap_coords[i, 1]), float(umap_coords[i, 2])
            cluster_id = int(cluster_ids[i])

            db.execute(text("""
                UPDATE investigator_stats_v3
                SET umap_x = :x,
                    umap_y = :y,
                    umap_z = :z,
                    cluster_id = :cluster_id
                WHERE id = :inv_id
            """), {
                "x": x,
                "y": y,
                "z": z,
                "cluster_id": cluster_id,
                "inv_id": inv_id
            })

            if (i + 1) % 100 == 0:
                logger.info(f"   Saved {i + 1}/{len(investigator_ids)} investigators")

        db.commit()
        logger.info(f"✅ Saved {len(investigator_ids)} investigators to database")


def main():
    """Main entry point"""
    logger.info("🚀 Starting UMAP generation")

    # 1. Load embeddings
    investigator_ids, embeddings = load_embeddings_from_db()

    # 2. Compute 3D UMAP
    umap_coords = compute_umap_3d(embeddings)

    # 3. Cluster investigators
    cluster_ids = cluster_investigators(embeddings)

    # 4. Save to database
    save_to_database(investigator_ids, umap_coords, cluster_ids)

    logger.info("✅ UMAP generation complete!")
    logger.info("   Use the airplane icon in the dashboard to visualize")


if __name__ == "__main__":
    main()
