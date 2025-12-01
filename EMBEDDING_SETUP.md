# Smart Search Setup Guide

## Overview

The Smart Search feature uses OpenAI embeddings to find investigators based on semantic similarity to a natural language query. This allows users to search for expertise like "AI for stroke in older adults using claims data" and get relevant matches.

## How It Works (Dual-Embedding Approach)

1. **Offline Preprocessing**: For each investigator, we build TWO separate profile embeddings:

   **Embedding 1 - Themes & Populations (75% weight)**:
   - Research themes/technologies (from AI profiling)
   - Study populations (from AI profiling)

   **Embedding 2 - Titles (25% weight)**:
   - Recent publication titles (top 20 by RCR)
   - Recent grant titles (last 2 years)

2. **Embedding Generation**: Both profiles are embedded using OpenAI's `text-embedding-3-large` model (3072 dimensions each)

3. **Query Time**: When a user searches:
   - Query is normalized using GPT-4o-mini to extract key concepts
   - Normalized query is embedded using the same model
   - Cosine similarity is computed against BOTH investigator embeddings
   - **Weighted ranking**: 75% themes/populations + 25% titles
   - Results are sorted by final similarity score (best to worst)
   - Top 10 matches are returned

**Key Improvement**: The dual-embedding approach prioritizes semantic matches based on research focus (themes + populations) while using publication/grant titles as supplementary context. This yields more accurate matches than the previous single-embedding approach.

## Setup Instructions

### 1. Prerequisites

Make sure you have:
- Populated database with investigators, publications, and grants
- AI profiling completed (themes and populations)
- OpenAI API key set in environment

### 2. Generate Embeddings

Run the embedding generation script:

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="sk-..."

# Run the generation script
python generate_embeddings.py
```

**Important**: The script will automatically add the required database columns (`embedding_themes_pops`, `embedding_titles`) if they don't exist. You'll see:

```
2025-01-28 10:15:30 [INFO] ✅ Connected to database and OpenAI
2025-01-28 10:15:30 [INFO] 🔧 Running database migrations...
2025-01-28 10:15:30 [INFO]    Adding column: embedding_themes_pops
2025-01-28 10:15:30 [INFO]    Adding column: embedding_titles
2025-01-28 10:15:30 [INFO] ✅ Database migrations complete
```

Or if the schema is already up to date:

```
2025-01-28 10:15:30 [INFO] ✅ Database schema up to date
```

The script will then:
- Load all investigators from the database
- Build two separate profile texts for each investigator
  - Themes + populations profile
  - Publication + grant titles profile
- Generate TWO embeddings using OpenAI API (one for each profile)
- Save both embeddings to the database

**Expected runtime**: ~4-8 seconds per investigator (2 API calls per investigator)
- For 400 investigators: ~30-60 minutes

### 3. Monitor Progress

The script logs progress for each investigator:

```
2025-01-28 10:15:32 [INFO] Built profiles for Smith, John (themes/pops: 847 chars, titles: 2150 chars)
2025-01-28 10:15:35 [INFO] ✅ Smith, John - dual embeddings generated
2025-01-28 10:15:38 [INFO] Built profiles for Doe, Jane (themes/pops: 523 chars, titles: 1893 chars)
2025-01-28 10:15:41 [INFO] ✅ Doe, Jane - dual embeddings generated
```

### 4. Verify Setup

After running the script, check that embeddings were generated:

```python
# Quick verification script
from sqlalchemy import create_engine, text
engine = create_engine("sqlite:///./local_storage.db")

with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT COUNT(*) as total,
               SUM(CASE WHEN embedding_themes_pops != 'null' THEN 1 ELSE 0 END) as with_themes_pops,
               SUM(CASE WHEN embedding_titles != 'null' THEN 1 ELSE 0 END) as with_titles
        FROM investigator_stats_v3
    """))
    row = result.fetchone()
    print(f"Total investigators: {row[0]}")
    print(f"With themes/pops embeddings: {row[1]}")
    print(f"With titles embeddings: {row[2]}")
```

## Using Smart Search

### From the Web Interface

1. Click the lightning bolt button (⚡) next to the search box
2. Enter a natural language query describing the research expertise you're looking for
3. Click "Search"
4. View top 10 matching investigators with their:
   - Match percentage
   - Technologies/methods
   - Study populations

### Example Queries

- "AI for stroke in older adults using claims data"
- "Depression treatment in adolescents"
- "Machine learning for predicting suicide risk"
- "Neuroimaging studies of schizophrenia"
- "Clinical trials for bipolar disorder"
- "GWAS studies of autism"

## Maintenance

### When to Re-generate Embeddings

Re-run `generate_embeddings.py` after:
- Adding new investigators (from publication refresh)
- Running AI profiling updates
- Significant publication/grant data updates

### Partial Updates

To update only specific investigators (coming soon):

```python
# Update only new investigators
python generate_embeddings.py --only-new

# Update specific investigator
python generate_embeddings.py --investigator-id "SMITH_JOHN"
```

## Troubleshooting

### "No investigators with embeddings found"

This means embeddings haven't been generated yet. Run `generate_embeddings.py`.

### "no such column: profile_text" error

This means the database columns weren't added. The script now auto-migrates, but if you see this error:

1. Make sure you're running the latest version of `generate_embeddings.py`
2. The auto-migration runs at startup - check the logs for migration messages
3. For PostgreSQL databases, you may need to run the main app first: `python ddash2.py` (which will trigger migrations), then run the embedding script

### OpenAI API Errors

- **Rate limit**: Script includes automatic delays. For large databases, you may need to increase delays.
- **API key**: Verify `OPENAI_API_KEY` is set correctly
- **Model access**: Ensure your API key has access to `text-embedding-3-large`

### Poor Search Results

If results seem off:
1. Verify AI profiling has completed (themes and populations are populated)
2. Check that publication and grant data is recent
3. Consider re-running AI profiling with updated parameters

## Cost Estimation

OpenAI Pricing (as of 2025):
- `text-embedding-3-large`: $0.13 per 1M tokens

Typical costs (dual-embedding approach):
- Average themes/pops profile: ~500 tokens
- Average titles profile: ~1000 tokens
- **Total per investigator**: ~1500 tokens × 2 API calls
- 400 investigators: ~1.2M tokens ≈ **$0.15**
- Plus query embeddings: ~100 tokens per search ≈ **$0.00001 per search**

Still very affordable, with improved search accuracy from dual embeddings.

## Technical Details

### Database Schema

New dual-embedding fields added to `investigator_stats_v3`:

```sql
-- Dual-embedding approach (v2)
ALTER TABLE investigator_stats_v3 ADD COLUMN embedding_themes_pops TEXT DEFAULT 'null';
ALTER TABLE investigator_stats_v3 ADD COLUMN embedding_titles TEXT DEFAULT 'null';

-- Legacy fields (deprecated but kept for backward compatibility)
ALTER TABLE investigator_stats_v3 ADD COLUMN profile_text TEXT DEFAULT '';
ALTER TABLE investigator_stats_v3 ADD COLUMN embedding_json TEXT DEFAULT 'null';
ALTER TABLE investigator_stats_v3 ADD COLUMN impact_score REAL DEFAULT 0.0;
```

### Embedding Storage

Each investigator has TWO embeddings stored as JSON-encoded arrays:
- **embedding_themes_pops**: Research themes + study populations
- **embedding_titles**: Publication + grant titles
- Dimension: 3072 each (text-embedding-3-large)
- Format: `"[0.123, -0.456, 0.789, ...]"`

### Similarity Computation

Cosine similarity formula (applied to both embeddings):

```
similarity = (A · B) / (||A|| × ||B||)
```

Where:
- A = query embedding
- B = investigator embedding
- Result range: [-1, 1], higher is more similar

**Weighted Blending**:

```
final_score = (0.75 × similarity_themes_pops) + (0.25 × similarity_titles)
```

This prioritizes semantic matches based on research focus (themes + populations) at 75% weight, with publication/grant titles providing supplementary context at 25% weight.

**Impact Score Removed**: Previous versions included impact score in ranking, but the current version uses pure semantic similarity for more accurate matches.

## Future Enhancements

Potential improvements:
1. **Hybrid search**: Combine with BM25 lexical search
2. **Re-ranking**: Use LLM to re-rank top 20 results
3. **Filters**: Add filters for funding level, publication count, etc.
4. **Explanation**: Show why each investigator matched
5. **Caching**: Cache query embeddings for common searches
6. **Incremental updates**: Only re-embed investigators with changed data
