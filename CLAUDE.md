# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**MGB Psychiatry Dashboard (DDash v0.62)** - A secure, production-ready FastAPI web application that aggregates and displays research metrics for psychiatry investigators at Massachusetts General Brigham (MGB) affiliated hospitals (MGH, BWH, McLean). The dashboard tracks:
- PubMed publications with iCite RCR scores
- NIH grant funding via RePORTER API
- AI-generated research profiles using OpenAI
- **Smart Search** - Embedding-based semantic search for finding investigators by research expertise

## Running the Application

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment template and configure
cp .env.example .env
# Edit .env with your actual credentials

# Run the server
python ddash2.py

# Access the dashboard at http://localhost:8000
```

### Deployment to Render.com

1. Create a new Web Service
2. Connect your Git repository
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `python ddash2.py`
5. Configure environment variables:
   - `USER_PASSWORD` - Password for regular users (required)
   - `ADMIN_PASSWORD` - Password for admin access (required)
   - `PI123_PASSWORD` - Password for search-focused view (required)
   - `SESSION_SECRET` - Random secret for session encryption (required)
   - `OPENAI_API_KEY` - OpenAI API key for AI profiling
   - `ENTREZ_EMAIL` - Your email for PubMed API
   - `DATABASE_URL` - Render will auto-configure if PostgreSQL is added

## Architecture Overview

### File Structure

```
ddash2/
├── ddash2.py              # Main application (FastAPI backend)
├── templates/             # Jinja2 HTML templates
│   ├── index.html         # Main dashboard
│   ├── login.html         # Login page
│   └── maintenance.html   # Error/maintenance page
├── static/                # Static assets
│   ├── app.js             # Client-side JavaScript
│   └── styles.css         # Custom CSS
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variable template
├── .gitignore             # Git ignore rules
└── CLAUDE.md              # This file
```

### Application Components

1. **Authentication System (Lines 35-250)**
   - Three-tier authentication system:
     - `ADMIN_PASSWORD` - Full access: view all data + trigger refresh jobs
     - `USER_PASSWORD` - Read-only access: view all data, no refresh buttons
     - `PI123_PASSWORD` - Search-focused view: simplified interface for finding investigators
   - Session-based auth using `SessionMiddleware` with encrypted cookies
   - User type stored in session as `user_type` ("admin", "user", or "pi123")
   - `require_auth()` - Validates any authenticated user, returns user_type
   - `require_admin()` - Validates admin-level access only
   - Context-sensitive Help modal shows different instructions based on user_type

2. **Database Layer (Lines 57-187)**
   - SQLAlchemy models with graceful degradation (`DB_AVAILABLE` flag)
   - Models: `SystemState`, `InvestigatorStat`, `PaperDetail`, `GrantDetail`
   - SQLite with WAL mode by default (PostgreSQL compatible via `DATABASE_URL` env var)
   - Auto-migration logic for schema updates
   - **IMPORTANT**: All DB operations check `DB_AVAILABLE` first and fail gracefully

3. **Data Collection Jobs (Lines 308-750)**
   - `perform_publication_scrape()` - Queries PubMed, fetches iCite RCR scores
   - `perform_grant_scrape()` - Fetches NIH grants from RePORTER API with fuzzy name matching
   - `perform_ai_profiling()` - Parallel processing (ThreadPoolExecutor) to generate research profiles via OpenAI
   - All jobs run in background threads and update `SystemState` table

4. **Web Routes (Lines 753-968)**
   - `/login` (GET/POST) - Authentication
   - `/logout` (POST) - Clear session
   - `/` - Main dashboard (requires auth)
   - `/status_fragment` - HTMX polling for job status
   - `/refresh_pubs`, `/refresh_grants`, `/refresh_ai` - Admin-only data refresh triggers
   - `/investigator/{id}` - Modal detail view

5. **Frontend (templates/ and static/)**
   - HTMX-driven dynamic updates
   - Tailwind CSS styling
   - Client-side table sorting/filtering
   - Hamburger menu with About modal (v0.62)

## Critical Design Patterns

**Graceful Degradation**: The app checks `DB_AVAILABLE` throughout. If database connection fails, users see a maintenance page instead of errors. All background jobs also check availability before running.

**Author Identity Normalization** (Line 264):
```python
normalize_author_identity(last_name, fore_name, initials)
# Returns: "LASTNAME_FIRSTNAME" (e.g., "SMITH_JOHN")
```
This canonical ID is used across all tables to link investigators.

**Grant Matching Strategy** (Lines 531-574):
- First tries exact match: `LASTNAME_FIRSTNAME`
- Falls back to initial: `LASTNAME_F`
- Uses `canonical_map` to handle name variations

**AI Job Parallelization** (Lines 733-741):
- Uses `ThreadPoolExecutor` with 5 workers
- Each worker gets its own DB session (critical for SQLite)
- Retry logic with 3 attempts per investigator

## Database Schema

**investigator_stats_v3** (primary table):
- `id` (PK): Normalized author ID
- Publication metrics: `total_papers`, `total_icite`, `n_p2` (12-month count)
- Deltas: `delta_n_val`, `delta_n_str`, `delta_i_val`, `delta_i_str`
- Funding: `funding_current`, `funding_past_5y`
- AI fields: `themes_json`, `tech_json`, `population_json`

**grant_details_v3** (composite PK):
- `project_num` + `investigator_id`
- Links grants to investigators via normalized ID
- `is_active`: Binary flag if `end_date > now()`

**paper_details_v3**:
- Stores PubMed metadata and RCR scores
- Linked to investigators via JSON array in `pmids_p2_json`

## Security Features

1. **Password Authentication**: Three-tier system (admin/user/pi123) with role-based access control
2. **Environment Variables**: All secrets stored in env vars, never hardcoded
3. **Session Security**: Encrypted sessions via `SESSION_SECRET`
4. **Input Validation**: FastAPI automatic validation on all routes
5. **No API Keys in Code**: `OPENAI_API_KEY`, `SESSION_SECRET` loaded from environment
6. **Graceful Failure**: Database errors show maintenance page, not stack traces
7. **Admin-Only Actions**: Data refresh operations require admin password
8. **Context-Sensitive Help**: Help system adapts to user role for security awareness

## Environment Variables (Required)

- `USER_PASSWORD` - Password for read-only users (default: "user123" - INSECURE)
- `ADMIN_PASSWORD` - Password for admin users (default: "admin123" - INSECURE)
- `PI123_PASSWORD` - Password for search-focused view (default: "pi123" - INSECURE)
- `SESSION_SECRET` - Secret key for session encryption (auto-generated if not set)
- `OPENAI_API_KEY` - OpenAI API key (optional, AI profiling will be skipped if missing)
- `ENTREZ_EMAIL` - Email for PubMed API compliance (defaults to "user@example.com")
- `DATABASE_URL` - Database connection string (defaults to SQLite)
- `PORT` - Server port (defaults to 8000)

## Key Gotchas

1. **DB Session Management**: Background jobs open/close their own sessions. AI workers each need `SessionLocal()` (Line 628).

2. **State Preservation**: When refreshing publications, grants and AI data are backed up first (Lines 404-413).

3. **Hospital Affiliation Logic** (Line 255): Only counts publications where affiliation text contains "psychiatry" AND a hospital name.

4. **No Duplicate Grants**: Uses `unique_grant_inserts` dict with `(project_num, investigator_id)` tuple key (Line 563).

5. **Modal Tab Switching**: JavaScript functions in static/app.js, not backend logic.

6. **Hamburger Menu**: Positioned absolute, closes on outside click (static/app.js).

7. **Deployment Port**: Uses `PORT` env var for cloud deployment compatibility (Line 967).

## AIDEV Notes

**AIDEV-NOTE**: All sensitive configuration is now loaded from environment variables (Line 34). Never hardcode secrets in the codebase.

**AIDEV-NOTE**: The `determine_hospital()` function (Line 255) filters ONLY psychiatry affiliations. If expanding to other departments, this logic must change.

**AIDEV-NOTE**: Auto-migration only adds columns, never removes them (Lines 144-160). For breaking changes, you must manually drop/recreate tables.

**AIDEV-NOTE**: Database connection with error handling for deployment (Line 62). The `DB_AVAILABLE` flag is critical for graceful degradation.

**AIDEV-NOTE**: Graceful failure handling shows maintenance page if DB unavailable (Line 792). This ensures users never see raw error messages.

**AIDEV-TODO**: Add proper logging for all file I/O operations (currently only HTTP/API calls are logged).

## Deployment Checklist

Before deploying to production:

1. ✅ Set strong passwords for `USER_PASSWORD`, `ADMIN_PASSWORD`, and `PI123_PASSWORD`
2. ✅ Generate random `SESSION_SECRET` (use `python -c "import secrets; print(secrets.token_urlsafe(32))"`)
3. ✅ Configure `OPENAI_API_KEY` if using AI profiling
4. ✅ Set `ENTREZ_EMAIL` to your actual email
5. ✅ For production databases, set `DATABASE_URL` to PostgreSQL
6. ✅ Ensure `PORT` env var is set (Render auto-configures this)
7. ✅ Test login with all three user types (admin, user, pi123)
8. ✅ Verify database graceful degradation (temporarily break DB connection)
9. ✅ Check that non-admin users cannot see refresh buttons
10. ✅ Verify Help modal shows context-sensitive content for each user type

## Smart Search Feature

### Overview

Smart Search uses OpenAI embeddings to find investigators based on semantic similarity to natural language queries.

### Setup

1. **Generate Embeddings** (one-time):
   ```bash
   export OPENAI_API_KEY="sk-..."
   python generate_embeddings.py
   ```

2. **Database Fields**:
   - `profile_text` - Combined text from themes, populations, publications, grants
   - `embedding_json` - JSON-encoded vector (3072 dimensions from text-embedding-3-large)
   - `impact_score` - Computed from RCR and grant funding

### Architecture

**Offline (generate_embeddings.py)**:
1. Load investigator data (themes, populations, recent pubs/grants)
2. Build profile text for each investigator
3. Compute impact score: `sum(log(1 + rcr)) + sum(log(1 + funding/100k))`
4. Generate embeddings using text-embedding-3-large
5. Save to database

**Online (ddash2.py: /smart_search route)**:
1. Normalize query using GPT-4o-mini to extract key concepts
2. Embed normalized query using text-embedding-3-large
3. Load all investigators with embeddings
4. Compute cosine similarity for each
5. Blend with impact score: `final_score = similarity + (0.01 × impact_score)`
6. Return top 10 matches

### Key Files

- `generate_embeddings.py` - Command-line tool to populate embeddings
- `ddash2.py:976-1060` - Smart search route and cosine similarity function
- `templates/index.html:63-68, 170-222` - Lightning bolt button and modal
- `static/app.js:124-238` - Smart search JavaScript
- `EMBEDDING_SETUP.md` - Complete setup and usage guide

### Investigator Filtering

The system filters out low-activity investigators from:
- Main dashboard display
- Smart search results
- Embedding generation (saves API costs)

**Filter criteria**: Excludes investigators with **both**:
- 5-year funding > $1M AND
- Papers in last 12 months ≤ 2

See `FILTERING_LOGIC.md` for details and configuration.

### Maintenance

Re-run `generate_embeddings.py` after:
- Adding new investigators
- Running AI profiling updates
- Significant data changes

## Version History

- v0.62 (2025) - Production-ready with authentication, security hardening, deployment support, and Smart Search
