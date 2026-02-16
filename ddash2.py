import time
import os
import json
import requests
import socket
import sys
import logging
import threading
import math
import sqlite3
import concurrent.futures
import secrets
import unicodedata
from typing import List, Dict, Set, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, Request, BackgroundTasks, Depends, HTTPException, Form, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from Bio import Entrez
import openai
from rank_bm25 import BM25Okapi

# Optional imports for UMAP visualization (installed separately)
try:
    import numpy as np
    import umap
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    np = None  # type: ignore

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- DATABASE RETRY DECORATOR ---
# AIDEV-NOTE: Decorator to retry database operations on connection failures
from functools import wraps
from sqlalchemy.exc import OperationalError, DBAPIError

def retry_on_db_error(max_retries=3, delay=1.0):
    """
    Decorator to retry database operations when they fail due to connection issues.
    Common on cloud platforms like Render where connections can drop.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (OperationalError, DBAPIError) as e:
                    last_exception = e
                    error_msg = str(e).lower()
                    # Check if it's a recoverable connection error
                    is_connection_error = any(phrase in error_msg for phrase in [
                        'ssl connection',
                        'closed unexpectedly',
                        'could not translate',
                        'connection reset',
                        'connection refused',
                        'timeout',
                        'broken pipe'
                    ])

                    if is_connection_error and attempt < max_retries - 1:
                        logger.warning(f"Database connection error (attempt {attempt + 1}/{max_retries}): {e}")
                        time.sleep(delay * (attempt + 1))  # Exponential backoff
                        continue
                    else:
                        # Not a connection error or max retries reached
                        raise
            raise last_exception
        return wrapper
    return decorator

# --- CONFIGURATION ---
# AIDEV-NOTE: All sensitive configuration is now loaded from environment variables
USER_PASSWORD = os.getenv("USER_PASSWORD", "")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
PI123_PASSWORD = os.getenv("PI123_PASSWORD", "")
SESSION_SECRET = os.getenv("SESSION_SECRET", secrets.token_urlsafe(32))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ENTREZ_EMAIL = os.getenv("ENTREZ_EMAIL", "user@example.com")

# AIDEV-NOTE: Set defaults only for missing passwords (don't overwrite ones that are set!)
missing_passwords = []
if not USER_PASSWORD:
    USER_PASSWORD = "user123"
    missing_passwords.append("USER_PASSWORD")
if not ADMIN_PASSWORD:
    ADMIN_PASSWORD = "admin123"
    missing_passwords.append("ADMIN_PASSWORD")
if not PI123_PASSWORD:
    PI123_PASSWORD = "pi123"
    missing_passwords.append("PI123_PASSWORD")

if missing_passwords:
    logger.warning(f"⚠️  {', '.join(missing_passwords)} not set! Using defaults (INSECURE)")
else:
    logger.info("✅ All password environment variables are set")

if not OPENAI_API_KEY:
    logger.warning("⚠️  OPENAI_API_KEY not set! AI profiling will fail.")

# --- DATABASE SETUP ---
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, event, text
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.engine import Engine

# 0. Set Global Timeout
socket.setdefaulttimeout(15)

# 1. Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./local_storage.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# AIDEV-NOTE: Database connection with error handling for deployment
# AIDEV-NOTE: pool_pre_ping=True tests connections before use (critical for cloud DB)
# AIDEV-NOTE: pool_recycle=3600 recycles connections every hour to prevent stale connections
try:
    if "sqlite" in DATABASE_URL:
        # SQLite configuration
        engine = create_engine(
            DATABASE_URL,
            connect_args={"check_same_thread": False},
            pool_size=20,
            max_overflow=0
        )
    else:
        # PostgreSQL configuration for cloud deployment
        engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,           # Test connections before using them
            pool_recycle=3600,             # Recycle connections after 1 hour
            pool_size=5,                   # Smaller pool for free/hobby tier
            max_overflow=10,               # Allow overflow for concurrent requests
            connect_args={
                "connect_timeout": 10,     # 10 second connection timeout
                "options": "-c statement_timeout=30000"  # 30 second query timeout
            }
        )

    @event.listens_for(Engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        if "sqlite" in DATABASE_URL:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.close()

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()

    DB_AVAILABLE = True
    logger.info("✅ Database connection established")

except Exception as e:
    logger.error(f"❌ Database connection failed: {e}")
    DB_AVAILABLE = False
    # Create dummy session for graceful degradation
    SessionLocal = None
    Base = None

# 2. Models
if DB_AVAILABLE:
    class SystemState(Base):
        __tablename__ = "system_state_v3"
        id = Column(Integer, primary_key=True)
        status = Column(String, default="Ready")
        last_updated = Column(String, default="Never")

    class InvestigatorStat(Base):
        __tablename__ = "investigator_stats_v3"
        id = Column(String, primary_key=True)
        name = Column(String)
        affiliation = Column(String)
        total_papers = Column(Integer, default=0)
        total_icite = Column(Float, default=0.0)
        n_p2 = Column(Integer, default=0)
        delta_n_val = Column(Float, default=0.0)
        delta_n_str = Column(String, default="0%")
        delta_i_val = Column(Float, default=0.0)
        delta_i_str = Column(String, default="0%")
        pmids_p2_json = Column(Text)
        funding_current = Column(Float, default=0.0)
        funding_past_5y = Column(Float, default=0.0)
        grants_json = Column(Text)

        # AI FIELDS
        themes_json = Column(Text, default="[]")
        tech_json = Column(Text, default="[]")
        population_json = Column(Text, default="[]")

        # SMART SEARCH FIELDS (v2 - dual embeddings)
        # AIDEV-NOTE: Separate embeddings for better semantic matching
        # - themes_pops: Technologies/methods + study populations (prioritized 75%)
        # - titles: Publication + grant titles (supplementary 25%)
        embedding_themes_pops = Column(Text, default="null")  # JSON-encoded vector
        embedding_titles = Column(Text, default="null")  # JSON-encoded vector

        # UMAP VISUALIZATION FIELDS
        # AIDEV-NOTE: 3D UMAP coordinates for interactive visualization
        umap_x = Column(Float, default=0.0)
        umap_y = Column(Float, default=0.0)
        umap_z = Column(Float, default=0.0)
        cluster_id = Column(Integer, default=0)

        # MULTI-DEPARTMENT & QUALITY CONTROL FIELDS
        # AIDEV-NOTE: Support for multiple departments and misclassification detection
        department = Column(String, default="psychiatry")  # Department affiliation
        probable_misclassification = Column(Integer, default=0)  # Boolean: 1 if <10% MGB affiliation
        mgb_affiliation_percentage = Column(Float, default=100.0)  # % of p2 papers with MGB affiliation

        # Deprecated fields (kept for backward compatibility)
        profile_text = Column(Text, default="")
        embedding_json = Column(Text, default="null")
        impact_score = Column(Float, default=0.0)

    class PaperDetail(Base):
        __tablename__ = "paper_details_v3"
        pmid = Column(String, primary_key=True)
        title = Column(Text)
        journal = Column(String)
        pub_date = Column(DateTime)
        rcr = Column(Float)

    class GrantDetail(Base):
        __tablename__ = "grant_details_v3"
        project_num = Column(String, primary_key=True)
        investigator_id = Column(String, primary_key=True, index=True)
        core_project_num = Column(String, index=True)
        fiscal_year = Column(Integer)
        project_title = Column(Text)
        agency = Column(String)
        award_amount = Column(Float, default=0.0)
        start_date = Column(DateTime, nullable=True)
        end_date = Column(DateTime, nullable=True)
        is_active = Column(Integer, default=0)

    # --- AUTO-MIGRATION LOGIC ---
    def check_and_migrate_db():
        if "sqlite" not in DATABASE_URL: return
        table_name = InvestigatorStat.__tablename__
        with engine.connect() as conn:
            try:
                result = conn.execute(text(f"PRAGMA table_info({table_name})"))
                existing_cols = [row[1] for row in result.fetchall()]

                if "themes_json" not in existing_cols:
                    conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN themes_json TEXT DEFAULT '[]'"))
                if "tech_json" not in existing_cols:
                    conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN tech_json TEXT DEFAULT '[]'"))
                if "population_json" not in existing_cols:
                    conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN population_json TEXT DEFAULT '[]'"))
                # Legacy smart search fields
                if "profile_text" not in existing_cols:
                    conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN profile_text TEXT DEFAULT ''"))
                if "embedding_json" not in existing_cols:
                    conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN embedding_json TEXT DEFAULT 'null'"))
                if "impact_score" not in existing_cols:
                    conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN impact_score REAL DEFAULT 0.0"))

                # New dual-embedding smart search fields
                if "embedding_themes_pops" not in existing_cols:
                    conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN embedding_themes_pops TEXT DEFAULT 'null'"))
                    logger.info("Added column: embedding_themes_pops")
                if "embedding_titles" not in existing_cols:
                    conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN embedding_titles TEXT DEFAULT 'null'"))
                    logger.info("Added column: embedding_titles")

                # UMAP visualization fields
                if "umap_x" not in existing_cols:
                    conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN umap_x REAL DEFAULT 0.0"))
                    logger.info("Added column: umap_x")
                if "umap_y" not in existing_cols:
                    conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN umap_y REAL DEFAULT 0.0"))
                    logger.info("Added column: umap_y")
                if "umap_z" not in existing_cols:
                    conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN umap_z REAL DEFAULT 0.0"))
                    logger.info("Added column: umap_z")
                if "cluster_id" not in existing_cols:
                    conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN cluster_id INTEGER DEFAULT 0"))
                    logger.info("Added column: cluster_id")

                # Multi-department and quality control fields
                if "department" not in existing_cols:
                    conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN department TEXT DEFAULT 'psychiatry'"))
                    logger.info("Added column: department")
                if "probable_misclassification" not in existing_cols:
                    conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN probable_misclassification INTEGER DEFAULT 0"))
                    logger.info("Added column: probable_misclassification")
                if "mgb_affiliation_percentage" not in existing_cols:
                    conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN mgb_affiliation_percentage REAL DEFAULT 100.0"))
                    logger.info("Added column: mgb_affiliation_percentage")

                conn.commit()
            except Exception as e:
                logger.error(f"Migration failed: {e}")

    try:
        Base.metadata.create_all(bind=engine)
        check_and_migrate_db()
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        DB_AVAILABLE = False

    # --- SELF-HEALING STARTUP ---
    if DB_AVAILABLE:
        try:
            with SessionLocal() as db:
                state = db.query(SystemState).filter_by(id=1).first()
                if state:
                    state.status = "Ready"
                    db.commit()
        except Exception as e:
            logger.warning(f"Startup DB Check failed: {e}")

# AIDEV-NOTE: get_db must be defined at module level, not inside if DB_AVAILABLE block
# This ensures routes can import it even if initial DB connection fails
def get_db():
    """
    AIDEV-NOTE: Database session dependency with connection validation.
    Tests the connection before yielding to catch stale connections early.
    """
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database unavailable")
    db = SessionLocal()
    try:
        # Test the connection before yielding
        # This forces pool_pre_ping to validate the connection
        db.execute(text("SELECT 1"))
        yield db
    except Exception as e:
        db.close()
        logger.error(f"Database session error: {e}")
        raise HTTPException(status_code=503, detail="Database connection failed")
    finally:
        db.close()

# --- EXTERNAL API CONFIGURATION ---
Entrez.email = ENTREZ_EMAIL
Entrez.tool = "MGB_Impact_Tracker"
ICITE_API_URL = "https://icite.od.nih.gov/api/pubs"
ICITE_CHUNK_SIZE = 200
REPORTER_API_URL = "https://api.reporter.nih.gov/v2/projects/search"

# *** OPENAI CONFIGURATION ***
if OPENAI_API_KEY:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None

SEARCH_TERMS = [
    '"Massachusetts General Hospital"[Affiliation]',
    '"Brigham and Women\'s Hospital"[Affiliation]',
    '"Brigham and Womens Hospital"[Affiliation]',  # Without apostrophe for inconsistent PubMed data
    '"McLean Hospital"[Affiliation]',
    '"Mass General Brigham"[Affiliation]'
]

# --- FASTAPI APP ---
app = FastAPI(title="MGB Psychiatry Dashboard")
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET)

# Mount static files and templates
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# --- AUTHENTICATION DEPENDENCIES ---
def require_auth(request: Request):
    """
    Require any authenticated user.
    Returns user_type for context-sensitive UI.
    AIDEV-NOTE: Backward compatibility - if user_type not set, infer from is_admin
    """
    if not request.session.get("authenticated"):
        raise HTTPException(status_code=401, detail="Not authenticated")

    # Get user_type, with backward compatibility
    user_type = request.session.get("user_type")
    if not user_type:
        # Infer from is_admin for old sessions
        user_type = "admin" if request.session.get("is_admin", False) else "user"
        request.session["user_type"] = user_type

    return user_type

def require_admin(request: Request):
    """Require admin authentication"""
    if not request.session.get("authenticated") or not request.session.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
    return True

# --- LOGIC & HELPERS ---

def set_system_status(db: Session, status: str, update_time: bool = False):
    try:
        state = db.query(SystemState).filter_by(id=1).first()
        if not state:
            state = SystemState(id=1)
            db.add(state)
        state.status = status
        if update_time:
            state.last_updated = datetime.now().strftime("%H:%M:%S")
        db.commit()
    except Exception as e:
        logger.error(f"DB Error setting status: {e}")
        db.rollback()

def normalize_text(text: str) -> str:
    """
    Normalize text by removing accents and converting to uppercase.

    AIDEV-NOTE: Handles names with accented characters (e.g., "Öngür" → "ONGUR")
    This ensures name matching works regardless of Unicode encoding or accent usage.

    Examples:
        "Öngür" → "ONGUR"
        "José" → "JOSE"
        "François" → "FRANCOIS"
    """
    if not text:
        return ""
    # Decompose Unicode characters (e.g., "ö" → "o" + combining diaeresis)
    nfd = unicodedata.normalize('NFD', text)
    # Filter out combining diacritical marks
    without_accents = ''.join(char for char in nfd if unicodedata.category(char) != 'Mn')
    return without_accents.upper()

def get_date_windows():
    now = datetime.now()
    p2_end = now
    p2_start = now - timedelta(days=365)
    p1_end = p2_start
    p1_start = p1_end - timedelta(days=365)
    return p1_start, p1_end, p2_start, p2_end

def determine_hospital(affiliation_text: str, department: str = "psychiatry") -> Optional[str]:
    """
    Determine if affiliation matches one of the MGB hospitals for the specified department.

    AIDEV-NOTE: Multi-department support - department filter is applied here.
    Currently only 'psychiatry' is implemented, but can be extended.

    AIDEV-NOTE: Exclusion logic to prevent false positives:
    - Excludes Beth Israel Deaconess Medical Center (not part of MGB Psychiatry)
    - Excludes Boston Children's Hospital (separate from adult psychiatry)
    - Excludes "Psychiatry and Neurology" joint departments (these are Neurology)
    - Requires specific hospital names, not just "Massachusetts"
    """
    text = affiliation_text.lower()
    dept_keywords = {
        "psychiatry": ['psychiatry', 'psychiatric'],
        # Add more departments here as needed
    }

    # EXCLUSIONS: Institutions that should NOT match
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
            return None  # Explicitly excluded institution

    # Check if affiliation matches the department
    if department not in dept_keywords:
        return None  # Unknown department

    if not any(keyword in text for keyword in dept_keywords[department]):
        return None  # Doesn't match department

    # AIDEV-NOTE: Exclude "Psychiatry and Neurology" joint departments
    # These are neurology departments that happen to include psychiatry in the name
    if 'psychiatry and neurology' in text or 'psychiatry & neurology' in text:
        return None  # Joint department - exclude

    # Check which hospital (must be specific, not just "massachusetts")
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

def normalize_author_identity(last_name: str, fore_name: str, initials: str) -> str:
    if not last_name: return ""
    clean_first = ""
    if fore_name and len(fore_name) > 1:
        clean_first = fore_name.split(" ")[0]
    elif initials:
        clean_first = initials[0]
    if not clean_first: return ""
    return f"{last_name.upper()}_{clean_first.upper()}"

def fetch_icite_scores(pmids: List[str]) -> Dict[str, float]:
    scores = {}
    if not pmids: return scores
    chunks = [pmids[i:i + 100] for i in range(0, len(pmids), 100)]
    for idx, chunk in enumerate(chunks):
        pmid_str = ",".join(chunk)
        logger.info(f"Fetching iCite chunk {idx+1}/{len(chunks)}")
        try:
            r = requests.get(f"{ICITE_API_URL}?pmids={pmid_str}", timeout=20)
            if r.status_code == 200:
                data = r.json()
                if 'data' in data:
                    for paper in data['data']:
                        rcr = paper.get('relative_citation_ratio')
                        scores[str(paper.get('pmid'))] = float(rcr) if rcr is not None else 0.0
        except Exception as e:
            logger.error(f"iCite Error on chunk {idx}: {e}")
    return scores

def parse_pubmed_date(article_data: Dict, medline_citation: Dict) -> datetime:
    """
    Parse publication date from PubMed data.

    AIDEV-NOTE: Prioritize ArticleDate (e-pub) over journal publication date
    to avoid ahead-of-print papers being counted in future years.
    """
    try:
        # FIRST: Try ArticleDate (electronic publication date)
        article_dates = article_data.get('ArticleDate', [])
        if article_dates:
            art_date = article_dates[0]  # Use first ArticleDate
            year = art_date.get('Year')
            month = art_date.get('Month', '1')
            day = art_date.get('Day', '1')
            if year:
                return datetime(int(year), int(month), int(day))

        # SECOND: Fall back to Journal PubDate
        pub_date = article_data.get('Journal', {}).get('JournalIssue', {}).get('PubDate', {})
        year = pub_date.get('Year') or medline_citation.get('DateCompleted', {}).get('Year')
        if not year: return datetime.now()
        month_str = pub_date.get('Month', '1')
        day_str = pub_date.get('Day', '1')
        try:
            month = int(month_str) if month_str.isdigit() else datetime.strptime(month_str[:3], '%b').month
        except: month = 1
        return datetime(int(year), month, int(day_str) if day_str.isdigit() else 1)
    except: return datetime.now()

# --- JOB 1: PUBLICATIONS ONLY ---
# AIDEV-NOTE: New logic for master faculty list and misclassification detection

def perform_publication_scrape(department: str = "psychiatry"):
    """
    Two-phase publication scraping with misclassification detection:

    PHASE 1: Build master faculty list
    - Query PubMed with hospital+department affiliation for past 24 months
    - Extract all unique authors -> this is the master list

    PHASE 2: Expand publication retrieval
    - For each author in master list, query ALL their publications (by name, not affiliation)
    - Track which papers have MGB affiliation vs not
    - Calculate MGB affiliation % for past 12 months
    - Set probable_misclassification flag if < 10% MGB affiliation
    """
    logger.info(f"STARTING PUBLICATION SCRAPE FOR DEPARTMENT: {department}...")
    if not DB_AVAILABLE:
        logger.error("Database not available, skipping publication scrape")
        return

    db = SessionLocal()
    try:
        p1_start, p1_end, p2_start, p2_end = get_date_windows()

        # ====================
        # PHASE 1: BUILD MASTER FACULTY LIST
        # ====================
        set_system_status(db, f"Phase 1: Building master faculty list ({department})...")
        hospital_query = " OR ".join(SEARCH_TERMS)

        # Query for MGB-affiliated papers in past 24 months
        query_phase1 = (
            f'(("{department.capitalize()}"[Affiliation]) AND ({hospital_query})) AND '
            f'("{p1_start.strftime("%Y/%m/%d")}"[Date - Publication] : "{p2_end.strftime("%Y/%m/%d")}"[Date - Publication])'
        )

        logger.info(f"Phase 1 Query: {query_phase1}")
        handle = Entrez.esearch(db="pubmed", term=query_phase1, retmax=5000)
        record = Entrez.read(handle)
        handle.close()
        phase1_pmids = record["IdList"]

        logger.info(f"Phase 1: Found {len(phase1_pmids)} MGB-affiliated papers")
        set_system_status(db, f"Phase 1: Processing {len(phase1_pmids)} papers...")

        # Extract master faculty list from these papers
        master_faculty = {}  # aid -> {name, hospitals}
        batch_size = 20

        for i in range(0, len(phase1_pmids), batch_size):
            batch_ids = phase1_pmids[i:i+batch_size]
            try:
                socket.setdefaulttimeout(20)
                fetch_handle = Entrez.efetch(db="pubmed", id=batch_ids, retmode="xml")
                papers = Entrez.read(fetch_handle)
                fetch_handle.close()

                for article in papers['PubmedArticle']:
                    medline = article['MedlineCitation']
                    art = medline['Article']

                    if 'AuthorList' in art:
                        for author in art['AuthorList']:
                            if 'AffiliationInfo' in author and 'LastName' in author:
                                valid_hospitals = set()
                                for affil in author['AffiliationInfo']:
                                    hosp = determine_hospital(affil['Affiliation'], department)
                                    if hosp: valid_hospitals.add(hosp)

                                if valid_hospitals:
                                    last = author['LastName']
                                    first = author.get('ForeName', '')
                                    initials = author.get('Initials', '')

                                    # AIDEV-NOTE: Smart deduplication using prefix matching
                                    # Examples: "Perlis R" and "Perlis RH" merge to PERLIS_RH (same person)
                                    #           "Rauch SL" and "Rauch SD" stay separate (different people)
                                    # Normalize to handle accented characters (e.g., "Öngür" → "ONGUR")
                                    last_clean = normalize_text(last)

                                    # Prefer using the Initials field (e.g., "SL", "BC", "RH")
                                    if initials and len(initials) > 0:
                                        all_initials = initials.replace(' ', '').replace('.', '').upper()
                                    elif first and len(first) > 0:
                                        # Extract initials from full first name (may include middle name)
                                        name_parts = first.split()
                                        all_initials = ''.join([part[0].upper() for part in name_parts if part])
                                    else:
                                        continue  # Skip if no name available

                                    # Build best display name (prefer full first name)
                                    if first:
                                        disp_name = f"{last}, {first}"
                                    elif initials:
                                        disp_name = f"{last}, {initials}"
                                    else:
                                        disp_name = f"{last}, {all_initials}"

                                    # Smart merge: check if this matches an existing entry (prefix logic)
                                    canonical_id = f"{last_clean}_{all_initials}"
                                    matched_id = None

                                    for existing_id in master_faculty.keys():
                                        existing_parts = existing_id.split('_')
                                        if len(existing_parts) == 2:
                                            existing_last, existing_initials = existing_parts
                                            if existing_last == last_clean:
                                                # Same last name - check if initials are compatible
                                                # Compatible if one is a prefix of the other
                                                if existing_initials.startswith(all_initials) or all_initials.startswith(existing_initials):
                                                    # Merge: keep the LONGER/more specific initial
                                                    if len(all_initials) > len(existing_initials):
                                                        matched_id = existing_id  # Will update this entry
                                                        canonical_id = f"{last_clean}_{all_initials}"
                                                    else:
                                                        matched_id = existing_id
                                                        canonical_id = existing_id
                                                    break

                                    if matched_id and matched_id != canonical_id:
                                        # Update existing entry with more specific ID
                                        master_faculty[canonical_id] = master_faculty[matched_id]
                                        del master_faculty[matched_id]

                                    # Add or update entry
                                    if canonical_id not in master_faculty:
                                        master_faculty[canonical_id] = {"name": disp_name, "hospitals": set()}
                                    else:
                                        # Keep the longer/more detailed name
                                        if len(disp_name) > len(master_faculty[canonical_id]["name"]):
                                            master_faculty[canonical_id]["name"] = disp_name

                                    master_faculty[canonical_id]["hospitals"].update(valid_hospitals)
            except Exception as e:
                logger.error(f"Phase 1 Batch Error: {e}")
                continue

        logger.info(f"Phase 1 Complete: Master faculty list has {len(master_faculty)} investigators")

        # ====================
        # PHASE 2: EXPAND PUBLICATION RETRIEVAL
        # ====================
        set_system_status(db, f"Phase 2: Retrieving all publications for {len(master_faculty)} faculty...")

        temp_papers = {}  # pmid -> {title, journal, pub_date, rcr, has_mgb_affiliation}
        inv_data = {}  # aid -> {name, hospitals, pmids_all, pmids_mgb, pmids_p1, pmids_p2, pmids_mgb_p2}

        faculty_count = 0
        total_faculty = len(master_faculty)

        for aid, faculty in master_faculty.items():
            faculty_count += 1
            if faculty_count % 10 == 0:
                set_system_status(db, f"Phase 2: Processing faculty {faculty_count}/{total_faculty}...")
                logger.info(f"Phase 2: Processing faculty {faculty_count}/{total_faculty}")

            # Parse author name for querying
            parts = aid.split('_')
            if len(parts) != 2: continue
            last_name, all_initials = parts

            # AIDEV-NOTE: Now matching using ALL initials (e.g., "SL", "BC") to avoid false matches
            # e.g., RAUCH_SL will only match "Scott L Rauch", not "Steven D Rauch"

            # Query ALL publications by this author in past 24 months (not filtered by affiliation)
            author_query = (
                f'({last_name}[Author]) AND '
                f'("{p1_start.strftime("%Y/%m/%d")}"[Date - Publication] : "{p2_end.strftime("%Y/%m/%d")}"[Date - Publication])'
            )

            try:
                socket.setdefaulttimeout(15)
                handle = Entrez.esearch(db="pubmed", term=author_query, retmax=200)
                record = Entrez.read(handle)
                handle.close()
                author_pmids = record["IdList"]

                if not author_pmids:
                    continue

                # Fetch full details to check affiliations
                fetch_handle = Entrez.efetch(db="pubmed", id=author_pmids, retmode="xml")
                papers = Entrez.read(fetch_handle)
                fetch_handle.close()

                pmids_all = set()
                pmids_mgb = set()
                pmids_p1 = set()
                pmids_p2 = set()
                pmids_mgb_p2 = set()

                for article in papers['PubmedArticle']:
                    medline = article['MedlineCitation']
                    pmid = str(medline['PMID'])
                    art = medline['Article']
                    dt = parse_pubmed_date(art, medline)

                    # Check if this author is on this paper (match by name)
                    author_on_paper = False
                    has_mgb_affiliation = False

                    if 'AuthorList' in art:
                        for auth in art['AuthorList']:
                            if 'LastName' in auth:
                                # Normalize to handle accented characters (e.g., "Öngür" → "ONGUR")
                                auth_last = normalize_text(auth['LastName'])
                                auth_first = auth.get('ForeName', '')
                                auth_initials_raw = auth.get('Initials', '')

                                # AIDEV-NOTE: Extract ALL initials to match canonical ID
                                # Prefer Initials field, fall back to ForeName
                                if auth_initials_raw:
                                    auth_all_initials = auth_initials_raw.replace(' ', '').replace('.', '').upper()
                                elif auth_first:
                                    # Extract initials from full first name (may include middle name)
                                    name_parts = auth_first.split()
                                    auth_all_initials = ''.join([part[0].upper() for part in name_parts if part])
                                else:
                                    auth_all_initials = ''

                                # AIDEV-NOTE: Fuzzy match using prefix logic
                                # Match if last name matches AND initials are compatible (prefix match)
                                # Examples: paper "Perlis R" matches canonical "PERLIS_RH"
                                #           paper "Rauch SL" matches canonical "RAUCH_SL"
                                #           paper "Rauch S" matches canonical "RAUCH_SL" but NOT "RAUCH_SD"
                                initials_match = False
                                if auth_last == normalize_text(last_name) and auth_all_initials:
                                    # Exact match or prefix match
                                    if auth_all_initials == all_initials:
                                        initials_match = True
                                    elif auth_all_initials.startswith(all_initials) or all_initials.startswith(auth_all_initials):
                                        initials_match = True

                                if initials_match:
                                    author_on_paper = True

                                    # Check if THIS author has MGB affiliation on THIS paper
                                    if 'AffiliationInfo' in auth:
                                        for affil in auth['AffiliationInfo']:
                                            if determine_hospital(affil['Affiliation'], department):
                                                has_mgb_affiliation = True
                                                break
                                    break

                    if not author_on_paper:
                        continue  # Skip papers where name match is weak

                    # Store paper details
                    if pmid not in temp_papers:
                        temp_papers[pmid] = {
                            "pmid": pmid,
                            "title": art.get('ArticleTitle', 'No Title'),
                            "journal": art.get('Journal', {}).get('Title', 'No Journal'),
                            "pub_date": dt,
                            "rcr": 0.0,
                            "has_mgb_affiliation": has_mgb_affiliation
                        }

                    pmids_all.add(pmid)
                    if has_mgb_affiliation:
                        pmids_mgb.add(pmid)

                    # Categorize by time period
                    if p1_start <= dt < p1_end:
                        pmids_p1.add(pmid)
                    elif p2_start <= dt < p2_end:
                        pmids_p2.add(pmid)
                        if has_mgb_affiliation:
                            pmids_mgb_p2.add(pmid)

                inv_data[aid] = {
                    "name": faculty["name"],
                    "hospitals": faculty["hospitals"],
                    "pmids_all": pmids_all,
                    "pmids_mgb": pmids_mgb,
                    "pmids_p1": pmids_p1,
                    "pmids_p2": pmids_p2,
                    "pmids_mgb_p2": pmids_mgb_p2
                }

            except Exception as e:
                logger.error(f"Phase 2 Error for {aid}: {e}")
                continue

            # AIDEV-NOTE: Batch save every 100 faculty to prevent OOM on 512MB instances
            # Save papers to DB, fetch iCite scores, then clear memory
            if faculty_count % 100 == 0 or faculty_count == total_faculty:
                logger.info(f"Batch save at {faculty_count}/{total_faculty} faculty ({len(temp_papers)} papers)...")

                # Fetch iCite scores for this batch
                batch_pmids = list(temp_papers.keys())
                if batch_pmids:
                    rcr_scores = fetch_icite_scores(batch_pmids)
                    for pid, score in rcr_scores.items():
                        if pid in temp_papers:
                            temp_papers[pid]["rcr"] = score

                # Save papers to database
                for pid, p_data in temp_papers.items():
                    db.merge(PaperDetail(
                        pmid=pid,
                        title=p_data['title'],
                        journal=p_data['journal'],
                        pub_date=p_data['pub_date'],
                        rcr=p_data['rcr']
                    ))

                db.commit()
                db.expunge_all()  # Clear session
                temp_papers.clear()  # Clear paper cache

                import gc
                gc.collect()  # Force garbage collection
                logger.info(f"Batch saved and memory freed")

        # AIDEV-NOTE: Papers are now saved in batches above
        # This section handles any remaining papers if total_faculty is not divisible by 100
        if temp_papers:
            logger.info(f"Saving final batch of {len(temp_papers)} papers...")
            batch_pmids = list(temp_papers.keys())
            rcr_scores = fetch_icite_scores(batch_pmids)
            for pid, score in rcr_scores.items():
                if pid in temp_papers:
                    temp_papers[pid]["rcr"] = score

            for pid, p_data in temp_papers.items():
                db.merge(PaperDetail(
                    pmid=pid,
                    title=p_data['title'],
                    journal=p_data['journal'],
                    pub_date=p_data['pub_date'],
                    rcr=p_data['rcr']
                ))
            db.commit()
            temp_papers.clear()

        # ====================
        # PHASE 3: CREATE INVESTIGATOR RECORDS WITH MISCLASSIFICATION DETECTION
        # ====================
        set_system_status(db, "Creating investigator records...")

        # Preserve funding AND AI info
        existing_invs = db.query(InvestigatorStat).all()
        backup_data = {
            inv.id: {
                "funding_current": inv.funding_current,
                "funding_past_5y": inv.funding_past_5y,
                "themes_json": inv.themes_json,
                "tech_json": inv.tech_json,
                "population_json": inv.population_json
            } for inv in existing_invs
        }

        db.query(InvestigatorStat).delete()

        # AIDEV-NOTE: aid is the canonical_id created in Phase 1 (LASTNAME_FIRSTINITIAL)
        # This ensures all name variations (e.g., "Perlis Roy", "Perlis R", "Perlis RH")
        # are deduplicated into a single database record
        for aid, data in inv_data.items():
            hospitals = data["hospitals"].copy()
            if any(h in hospitals for h in ['MGH', 'BWH', 'McLean']) and 'MGB' in hospitals:
                hospitals.remove('MGB')
            affiliation_str = ", ".join(sorted(hospitals))

            # Calculate metrics
            set_p1 = list(data["pmids_p1"])
            set_p2 = list(data["pmids_p2"])

            # AIDEV-NOTE: Query database for RCR since temp_papers is cleared in batches
            rcr_p1 = 0.0
            rcr_p2 = 0.0
            if set_p1:
                papers_p1 = db.query(PaperDetail).filter(PaperDetail.pmid.in_(set_p1)).all()
                rcr_p1 = sum(p.rcr for p in papers_p1)
            if set_p2:
                papers_p2 = db.query(PaperDetail).filter(PaperDetail.pmid.in_(set_p2)).all()
                rcr_p2 = sum(p.rcr for p in papers_p2)

            n_p1, n_p2 = len(set_p1), len(set_p2)

            def get_delta(prev, curr):
                if prev > 0:
                    return ((curr - prev)/prev)*100, f"{abs(int(((curr-prev)/prev)*100))}%"
                return (100.0, "New") if curr > 0 else (0.0, "-")

            dn_val, dn_str = get_delta(n_p1, n_p2)
            di_val, di_str = get_delta(rcr_p1, rcr_p2)

            # Calculate MGB affiliation percentage for p2 papers
            if n_p2 > 0:
                mgb_affiliation_pct = (len(data["pmids_mgb_p2"]) / n_p2) * 100.0
            else:
                mgb_affiliation_pct = 100.0  # Default to 100% if no papers

            # Determine misclassification flag
            probable_misclassification = 1 if mgb_affiliation_pct < 10.0 else 0

            backup = backup_data.get(aid, {})

            db.add(InvestigatorStat(
                id=aid,
                name=data["name"],
                affiliation=affiliation_str,
                total_papers=n_p1+n_p2,
                total_icite=rcr_p1+rcr_p2,
                n_p2=n_p2,
                delta_n_val=dn_val,
                delta_n_str=dn_str,
                delta_i_val=di_val,
                delta_i_str=di_str,
                pmids_p2_json=json.dumps(set_p2),
                funding_current=backup.get("funding_current", 0.0),
                funding_past_5y=backup.get("funding_past_5y", 0.0),
                themes_json=backup.get("themes_json", "[]"),
                tech_json=backup.get("tech_json", "[]"),
                population_json=backup.get("population_json", "[]"),
                department=department,
                probable_misclassification=probable_misclassification,
                mgb_affiliation_percentage=mgb_affiliation_pct
            ))

        set_system_status(db, "Ready", update_time=True)
        logger.info(f"Publication Job Complete. Processed {len(inv_data)} investigators.")
        db.commit()

    except Exception as e:
        logger.error(f"PUB JOB FAILURE: {e}")
        db.rollback()
        set_system_status(db, f"Error: {str(e)[:50]}")
    finally:
        db.close()

# --- JOB 2: GRANTS ONLY (RELAXED) ---

def perform_grant_scrape():
    logger.info("STARTING GRANT SCRAPE (RELAXED MODE)...")
    if not DB_AVAILABLE:
        logger.error("Database not available, skipping grant scrape")
        return

    db = SessionLocal()
    try:
        investigators = db.query(InvestigatorStat).all()
        if not investigators:
            set_system_status(db, "Error: No investigators found. Run Pubs first.")
            return

        query_names = []
        canonical_map = {}

        # Build Map
        for inv in investigators:
            parts = inv.id.split('_')
            if len(parts) == 2:
                last, first_part = parts
                # AIDEV-NOTE: Use normalize_text() for accented characters (e.g., "Öngür")
                key = f"{normalize_text(last)}_{normalize_text(first_part)}"
                if key not in canonical_map: canonical_map[key] = []
                canonical_map[key].append(inv.id)

                # AIDEV-NOTE: NIH Reporter query uses only first initial
                # Query with "R" will match "Roy", "Roy H", "Robert", etc.
                # Then fuzzy matching filters to correct investigator
                first_initial = first_part[0] if first_part else ""
                # Normalize last name for query to handle accents
                query_names.append({"last_name": normalize_text(last), "first_name": first_initial})

        batch_size = 20
        current_year = datetime.now().year
        years = [current_year - i for i in range(6)]
        unique_names = [dict(t) for t in {tuple(d.items()) for d in query_names}]

        grants_buffer = []
        total_batches = (len(unique_names) // batch_size) + 1

        for i in range(0, len(unique_names), batch_size):
            batch_num = (i // batch_size) + 1
            status = f"Fetching Grants: Batch {batch_num}/{total_batches}"
            set_system_status(db, status)
            logger.info(status)

            batch = unique_names[i:i+batch_size]
            payload = {
                "criteria": { "pi_names": batch, "fiscal_years": years },
                "include_fields": ["ProjectNum", "ProjectSerialNum", "FiscalYear", "Organization", "AgencyIcAdmin", "ContactPiName", "PrincipalInvestigators", "ProjectStartDate", "ProjectEndDate", "AwardAmount", "ProjectTitle"],
                "offset": 0, "limit": 500
            }

            while True:
                try:
                    r = requests.post(REPORTER_API_URL, json=payload, timeout=25)
                    if r.status_code != 200: break
                    data = r.json()
                    results = data.get("results", [])
                    grants_buffer.extend(results)
                    if len(results) < 500 or data.get('meta', {}).get('total', 0) <= (payload['offset'] + 500): break
                    payload['offset'] += 500
                    time.sleep(0.5)
                except Exception as e:
                    logger.error(f"RePORTER Batch Error: {e}")
                    break

        set_system_status(db, "Processing Grant Data...")

        tracked_ids = [i.id for i in investigators]
        db.query(GrantDetail).filter(GrantDetail.investigator_id.in_(tracked_ids)).delete(synchronize_session=False)

        unique_grant_inserts = {}

        for g in grants_buffer:
            pis = g.get('principal_investigators') or []

            for pi in pis:
                # AIDEV-NOTE: Normalize names to handle accented characters
                p_first = normalize_text(pi.get('first_name') or '')
                p_last = normalize_text(pi.get('last_name') or '')

                if not p_first or not p_last: continue

                # AIDEV-NOTE: Extract initials from full name (e.g., "ROY HERBERT" → "RH")
                p_first_clean = p_first.split(" ")[0]
                name_parts = p_first.split()
                p_initials = ''.join([part[0] for part in name_parts if part])

                matched_ids = []

                # AIDEV-NOTE: Fuzzy matching with prefix logic (same as publications)
                # Try matching: exact initials, full name, or prefix match
                # Examples: "PERLIS_RH" matches "Roy H", "Roy", "R", etc.
                for canonical_key in canonical_map.keys():
                    canonical_last, canonical_initials = canonical_key.split('_')

                    if canonical_last != p_last:
                        continue

                    # Prefix matching: either direction
                    # "RH" matches "R", "R" matches "RH"
                    if (canonical_initials.startswith(p_initials) or
                        p_initials.startswith(canonical_initials)):
                        matched_ids.extend(canonical_map[canonical_key])
                        break

                if matched_ids:
                    for inv_id in matched_ids:
                        start_d = None
                        end_d = None
                        if g.get('project_start_date'): start_d = datetime.strptime(g['project_start_date'].split('T')[0], '%Y-%m-%d')
                        if g.get('project_end_date'): end_d = datetime.strptime(g['project_end_date'].split('T')[0], '%Y-%m-%d')
                        is_active = 1 if end_d and end_d > datetime.now() else 0

                        proj_num = g.get('project_num')

                        unique_grant_inserts[(proj_num, inv_id)] = GrantDetail(
                            project_num = proj_num,
                            core_project_num = g.get('project_serial_num'),
                            investigator_id = inv_id,
                            fiscal_year = g.get('fiscal_year'),
                            project_title = g.get('project_title'),
                            agency = g.get('agency_ic_admin', {}).get('code', 'NIH'),
                            award_amount = float(g.get('award_amount') or 0),
                            start_date = start_d,
                            end_date = end_d,
                            is_active = is_active
                        )

        for grant_obj in unique_grant_inserts.values():
            db.merge(grant_obj)
        db.commit()

        set_system_status(db, "Recalculating Totals...")

        grants_all = db.query(GrantDetail).all()
        inv_grant_buckets = {}
        for g in grants_all:
            if g.investigator_id not in inv_grant_buckets: inv_grant_buckets[g.investigator_id] = []
            inv_grant_buckets[g.investigator_id].append(g)

        funding_map = {}
        for aid, grants in inv_grant_buckets.items():
            total_5y = sum(g.award_amount for g in grants)
            latest_projects = {}
            for g in grants:
                core = g.core_project_num
                if core not in latest_projects or g.fiscal_year > latest_projects[core].fiscal_year:
                    latest_projects[core] = g
            current_sum = sum(g.award_amount for g in latest_projects.values() if g.is_active)
            funding_map[aid] = (current_sum, total_5y)

        for inv in investigators:
            f_curr, f_5y = funding_map.get(inv.id, (0.0, 0.0))
            inv.funding_current = f_curr
            inv.funding_past_5y = f_5y

        set_system_status(db, "Ready", update_time=True)
        logger.info("Grant Job Complete.")
        db.commit()

    except Exception as e:
        logger.error(f"GRANT JOB FAILURE: {e}")
        db.rollback()
        set_system_status(db, f"Error: {str(e)[:50]}")
    finally:
        db.close()

# --- JOB 3: PARALLEL AI PROFILING ---

def process_investigator_ai(inv_id):
    """Worker function for parallel AI processing"""
    if not client:
        logger.error("OpenAI client not available")
        return

    if not DB_AVAILABLE:
        logger.error("Database not available")
        return

    # Create local DB session for this thread
    db = SessionLocal()
    try:
        inv = db.query(InvestigatorStat).filter_by(id=inv_id).first()
        if not inv: return

        # 1. Get Top Recent Papers (up to 10)
        recent_titles = []
        impact_titles = []

        if inv.pmids_p2_json:
            pmids = json.loads(inv.pmids_p2_json)
            papers = db.query(PaperDetail).filter(PaperDetail.pmid.in_(pmids)).all()
            papers.sort(key=lambda x: x.pub_date, reverse=True)
            recent_titles = [p.title for p in papers[:10]]

            papers.sort(key=lambda x: x.rcr, reverse=True)
            impact_titles = [p.title for p in papers[:10]]

        # 2. Get Grants
        grants = db.query(GrantDetail).filter_by(investigator_id=inv.id).all()
        grant_titles = [g.project_title for g in grants]

        if not recent_titles and not grant_titles:
            return

        # 3. Construct Prompt (Updated)
        prompt_content = f"""
        Investigator: {inv.name}

        Recent Publications:
        {json.dumps(recent_titles)}

        High Impact Publications:
        {json.dumps(impact_titles)}

        Grants (Last 5 Years):
        {json.dumps(grant_titles)}

        Task:
        1. Identify 3 main technologies or methodologies used (e.g., fMRI, GWAS, CBT). This is MANDATORY.
        2. Identify the primary population studied (e.g., Pediatric, Geriatric, Bipolar Disorder Patients, General Population). Return 1-2 distinct phrases. This is optional; return empty list if unclear.

        JSON Response Format:
        {{
            "technologies": ["Tech 1", "Tech 2", "Tech 3"],
            "populations": ["Pop 1", "Pop 2"]
        }}
        """

        # 4. Retry Logic
        max_retries = 3

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant providing concise scientific summaries."},
                        {"role": "user", "content": prompt_content}
                    ],
                    max_completion_tokens=5000,
                    response_format={"type": "json_object"}
                )

                result_text = response.choices[0].message.content
                data = json.loads(result_text)

                techs = data.get("technologies", [])
                pops = data.get("populations", [])

                if not techs:
                    raise ValueError("Empty technologies received")

                inv.tech_json = json.dumps(techs)
                inv.population_json = json.dumps(pops)
                db.commit()
                logger.info(f"AI Success for {inv.name}")
                return

            except Exception as e:
                logger.warning(f"AI Attempt {attempt+1} failed for {inv.name}: {e}")
                time.sleep(1.0)

    except Exception as e:
        logger.error(f"Worker Error for {inv_id}: {e}")
    finally:
        db.close()

def perform_ai_profiling():
    logger.info("STARTING PARALLEL AI PROFILE JOB...")
    if not DB_AVAILABLE:
        logger.error("Database not available, skipping AI profiling")
        return

    db = SessionLocal()
    try:
        # Get all IDs first
        investigators = db.query(InvestigatorStat).all()
        inv_ids = [i.id for i in investigators]
        db.close() # Close main session, workers open their own

        total = len(inv_ids)
        completed = 0

        # Parallel Execution with ThreadPool
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_id = {executor.submit(process_investigator_ai, iid): iid for iid in inv_ids}

            for future in concurrent.futures.as_completed(future_to_id):
                completed += 1
                # Update status every few items
                if completed % 2 == 0:
                    with SessionLocal() as status_db:
                        set_system_status(status_db, f"AI Processing: {completed}/{total}")

        with SessionLocal() as status_db:
            set_system_status(status_db, "Ready", update_time=True)
        logger.info("AI Job Complete.")

        # AIDEV-NOTE: Trigger embedding generation after AI profiling completes
        logger.info("Triggering embedding generation...")
        threading.Thread(target=perform_embedding_generation).start()

    except Exception as e:
        logger.error(f"AI JOB MAIN FAILURE: {e}")
        with SessionLocal() as status_db:
             set_system_status(status_db, f"Error: {str(e)[:50]}")


# --- JOB 4: EMBEDDING GENERATION ---

def perform_embedding_generation():
    """
    Generate dual embeddings for all investigators.
    AIDEV-NOTE: Runs automatically after AI profiling or when needed for UMAP.
    """
    logger.info("STARTING EMBEDDING GENERATION JOB...")
    if not DB_AVAILABLE:
        logger.error("Database not available, skipping embedding generation")
        return

    db = SessionLocal()
    try:
        set_system_status(db, "Generating Embeddings...")

        # Get investigators that need embeddings
        investigators = db.query(InvestigatorStat).filter(
            (InvestigatorStat.embedding_themes_pops == None) |
            (InvestigatorStat.embedding_themes_pops == 'null') |
            (InvestigatorStat.embedding_titles == None) |
            (InvestigatorStat.embedding_titles == 'null')
        ).all()

        logger.info(f"Found {len(investigators)} investigators needing embeddings")

        success_count = 0
        two_years_ago = datetime.now() - timedelta(days=730)

        for i, inv in enumerate(investigators, 1):
            try:
                # Build themes/pops text
                themes = json.loads(inv.tech_json) if inv.tech_json else []
                pops = json.loads(inv.population_json) if inv.population_json else []

                themes_pops_parts = []
                if themes:
                    themes_pops_parts.append(f"Research technologies and methods: {', '.join(themes)}")
                if pops:
                    themes_pops_parts.append(f"Study populations and cohorts: {', '.join(pops)}")
                themes_pops_text = "\n\n".join(themes_pops_parts) if themes_pops_parts else "No themes or populations available"

                # Build titles text
                pmids = json.loads(inv.pmids_p2_json) if inv.pmids_p2_json else []
                papers = db.query(PaperDetail).filter(
                    PaperDetail.pmid.in_(pmids),
                    PaperDetail.pub_date >= two_years_ago
                ).all() if pmids else []

                # Sort by RCR and get top 20
                papers_sorted = sorted(papers, key=lambda x: x.rcr or 0, reverse=True)[:20]
                pub_titles = [p.title for p in papers_sorted if p.title]

                grants = db.query(GrantDetail).filter(
                    GrantDetail.investigator_id == inv.id,
                    GrantDetail.fiscal_year >= datetime.now().year - 2
                ).all()
                grant_titles = [g.project_title for g in grants if g.project_title]

                titles_parts = []
                if pub_titles:
                    titles_parts.append(f"Publication titles: {' | '.join(pub_titles)}")
                if grant_titles:
                    titles_parts.append(f"Grant titles: {' | '.join(grant_titles)}")
                titles_text = "\n\n".join(titles_parts) if titles_parts else "No publication or grant titles available"

                # Generate embeddings
                if themes_pops_text.strip():
                    response1 = client.embeddings.create(
                        input=themes_pops_text,
                        model="text-embedding-3-large"
                    )
                    embedding_themes_pops = response1.data[0].embedding
                else:
                    embedding_themes_pops = [0.0] * 3072

                if titles_text.strip():
                    response2 = client.embeddings.create(
                        input=titles_text,
                        model="text-embedding-3-large"
                    )
                    embedding_titles = response2.data[0].embedding
                else:
                    embedding_titles = [0.0] * 3072

                # Save to database
                inv.embedding_themes_pops = json.dumps(embedding_themes_pops)
                inv.embedding_titles = json.dumps(embedding_titles)
                db.commit()

                success_count += 1

                if i % 10 == 0:
                    set_system_status(db, f"Embeddings: {i}/{len(investigators)}")
                    logger.info(f"Generated embeddings for {i}/{len(investigators)} investigators")
                    time.sleep(0.5)  # Rate limiting

            except Exception as e:
                logger.error(f"Error generating embedding for {inv.id}: {e}")
                db.rollback()
                continue

        set_system_status(db, "Ready", update_time=True)
        logger.info(f"Embedding generation complete! Success: {success_count}/{len(investigators)}")

        # Automatically trigger UMAP generation after embeddings
        if success_count > 0:
            logger.info("Triggering UMAP generation...")
            threading.Thread(target=perform_umap_generation).start()

    except Exception as e:
        logger.error(f"EMBEDDING JOB FAILURE: {e}")
        set_system_status(db, f"Error: {str(e)[:50]}")
    finally:
        db.close()


# --- JOB 5: UMAP GENERATION ---

def perform_umap_generation():
    """
    Generate 3D UMAP coordinates and clusters from embeddings.
    AIDEV-NOTE: Automatically triggered after embedding generation.
    """
    logger.info("STARTING UMAP GENERATION JOB...")

    if not DB_AVAILABLE:
        logger.error("Database not available, skipping UMAP generation")
        return

    if not UMAP_AVAILABLE:
        logger.error("UMAP libraries not available! Install with: pip install umap-learn scikit-learn numpy")
        return

    db = SessionLocal()
    try:
        set_system_status(db, "Generating UMAP Visualization...")

        # Load investigators with dual embeddings
        investigators = db.query(InvestigatorStat).filter(
            InvestigatorStat.embedding_themes_pops != None,
            InvestigatorStat.embedding_themes_pops != 'null',
            InvestigatorStat.embedding_titles != None,
            InvestigatorStat.embedding_titles != 'null'
        ).all()

        if not investigators:
            logger.error("No investigators with embeddings found! Run embedding generation first.")
            set_system_status(db, "Error: No embeddings")
            return

        logger.info(f"Loaded {len(investigators)} investigators with embeddings")

        # Parse embeddings and combine (20% themes/pops + 80% titles)
        investigator_ids = []
        embeddings_list = []

        for inv in investigators:
            try:
                themes_pops_emb = np.array(json.loads(inv.embedding_themes_pops))
                titles_emb = np.array(json.loads(inv.embedding_titles))
                combined = 0.2 * themes_pops_emb + 0.8 * titles_emb

                investigator_ids.append(inv.id)
                embeddings_list.append(combined)
            except Exception as e:
                logger.warning(f"Skipping {inv.id}: {e}")
                continue

        if not embeddings_list:
            logger.error("Failed to load any valid embeddings")
            set_system_status(db, "Error: Invalid embeddings")
            return

        embeddings_matrix = np.array(embeddings_list)
        logger.info(f"Embeddings matrix shape: {embeddings_matrix.shape}")

        # Compute 3D UMAP
        set_system_status(db, "Computing 3D UMAP...")
        logger.info("Computing 3D UMAP...")

        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )
        umap_coords = reducer.fit_transform(embeddings_matrix)
        logger.info(f"UMAP computed, shape: {umap_coords.shape}")

        # Cluster investigators
        set_system_status(db, "Clustering investigators...")
        logger.info("Clustering investigators...")

        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings_matrix)

        kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
        cluster_ids = kmeans.fit_predict(embeddings_scaled)

        # Show cluster distribution
        unique, counts = np.unique(cluster_ids, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            logger.info(f"   Cluster {cluster_id}: {count} investigators")

        # Save to database
        set_system_status(db, "Saving UMAP data...")
        logger.info("Saving UMAP data to database...")

        for i, inv_id in enumerate(investigator_ids):
            x, y, z = float(umap_coords[i, 0]), float(umap_coords[i, 1]), float(umap_coords[i, 2])
            cluster_id = int(cluster_ids[i])

            inv = db.query(InvestigatorStat).filter_by(id=inv_id).first()
            if inv:
                inv.umap_x = x
                inv.umap_y = y
                inv.umap_z = z
                inv.cluster_id = cluster_id

            if (i + 1) % 100 == 0:
                db.commit()
                logger.info(f"   Saved {i + 1}/{len(investigator_ids)} investigators")

        db.commit()
        set_system_status(db, "Ready", update_time=True)
        logger.info(f"UMAP generation complete! Saved {len(investigator_ids)} investigators")

    except Exception as e:
        logger.error(f"UMAP JOB FAILURE: {e}")
        set_system_status(db, f"Error: {str(e)[:50]}")
    finally:
        db.close()


# --- ROUTES ---

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, error: str = None):
    return templates.TemplateResponse("login.html", {"request": request, "error": error})

@app.post("/login")
def login(request: Request, password: str = Form(...)):
    """
    AIDEV-NOTE: Three-tier authentication system
    - admin: Full access (refresh buttons, all features)
    - user: Read-only access (all features except refresh buttons)
    - pi123: Simplified search-focused view (name, technologies, populations only)
    """
    user_type = None

    if password == ADMIN_PASSWORD:
        user_type = "admin"
    elif password == USER_PASSWORD:
        user_type = "user"
    elif password == PI123_PASSWORD:
        user_type = "pi123"

    if user_type:
        request.session["authenticated"] = True
        request.session["user_type"] = user_type
        request.session["is_admin"] = (user_type == "admin")
        return RedirectResponse(url="/", status_code=303)
    else:
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "Invalid password"
        })

@app.post("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)

@app.get("/", response_class=HTMLResponse)
def home(request: Request, db: Session = Depends(get_db)):
    """
    AIDEV-NOTE: Main dashboard with three-tier user support
    - admin/user: See full table with grants and publications
    - pi123: Sees simplified search-focused view (handled in template)
    """
    # Check authentication
    if not request.session.get("authenticated"):
        return RedirectResponse(url="/login", status_code=303)

    # Get user_type with backward compatibility
    user_type = request.session.get("user_type")
    if not user_type:
        user_type = "admin" if request.session.get("is_admin", False) else "user"
        request.session["user_type"] = user_type

    # AIDEV-NOTE: Graceful failure handling - show maintenance page if DB unavailable
    if not DB_AVAILABLE:
        return templates.TemplateResponse("maintenance.html", {"request": request})

    try:
        # Retry database queries with exponential backoff
        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                state = db.query(SystemState).filter_by(id=1).first()
                status_txt = state.status if state else "Ready"
                last_upd = state.last_updated if state else "Never"

                # AIDEV-NOTE: Filter investigators
                # Only exclude probable misclassifications (<10% MGB affiliation in past 12 months)
                # The misclassification filter is more accurate than arbitrary thresholds
                all_investigators = db.query(InvestigatorStat).all()
                inv_list = [
                    inv for inv in all_investigators
                    if (inv.probable_misclassification == 0 or inv.probable_misclassification is None)
                ]
                # Sort by current funding
                inv_list.sort(key=lambda x: x.funding_current, reverse=True)

                return templates.TemplateResponse("index.html", {
                    "request": request,
                    "investigators": inv_list,
                    "status": status_txt,
                    "last_updated": last_upd,
                    "is_admin": request.session.get("is_admin", False),
                    "user_type": user_type
                })
            except (OperationalError, DBAPIError) as e:
                last_error = e
                error_msg = str(e).lower()
                is_connection_error = any(phrase in error_msg for phrase in [
                    'ssl connection', 'closed unexpectedly', 'could not translate',
                    'connection reset', 'timeout', 'broken pipe'
                ])

                if is_connection_error and attempt < max_retries - 1:
                    logger.warning(f"Dashboard DB error (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(1.0 * (attempt + 1))  # Exponential backoff
                    # Force new connection
                    db.close()
                    db = SessionLocal()
                    continue
                else:
                    raise

        # If we exhausted retries, raise the last error
        if last_error:
            raise last_error

    except Exception as e:
        logger.error(f"Error loading dashboard: {e}", exc_info=True)
        return templates.TemplateResponse("maintenance.html", {"request": request})

@app.get("/status_fragment", response_class=HTMLResponse)
def status_fragment(request: Request, db: Session = Depends(get_db)):
    require_auth(request)

    if not DB_AVAILABLE:
        return HTMLResponse("<span class='px-2 py-1 rounded bg-red-600 font-medium'>DB Unavailable</span>")

    try:
        state = db.query(SystemState).filter_by(id=1).first()
        status_txt = state.status if state else "Unknown"
        content = f"""<span class="px-2 py-1 rounded bg-amber-500 font-medium animate-pulse" hx-get="/status_fragment" hx-trigger="load delay:1s" hx-swap="outerHTML">{status_txt}</span>"""
        if status_txt == "Ready" or "Error" in status_txt:
             return HTMLResponse("<div id='status-container' hx-trigger='load' hx-get='/' hx-target='body'>Reloading...</div>", headers={"Cache-Control": "no-store"})
        return HTMLResponse(content, headers={"Cache-Control": "no-store"})
    except Exception as e:
        logger.error(f"Error in status_fragment: {e}")
        return HTMLResponse("<span class='px-2 py-1 rounded bg-red-600 font-medium'>DB Error</span>", headers={"Cache-Control": "no-store"})

@app.post("/refresh_pubs")
def trigger_pubs(request: Request, db: Session = Depends(get_db)):
    require_admin(request)

    if not DB_AVAILABLE:
        return HTMLResponse("<span>DB Unavailable</span>")

    state = db.query(SystemState).filter_by(id=1).first()
    if state and state.status != "Ready" and "Error" not in state.status: return HTMLResponse("<span>Busy...</span>")
    set_system_status(db, "Initializing Pubs...")
    thread = threading.Thread(target=perform_publication_scrape)
    thread.start()
    return HTMLResponse("""<div id="status-container" class="text-xs flex items-center gap-3"><span class="px-2 py-1 rounded bg-amber-500 font-medium animate-pulse" hx-get="/status_fragment" hx-trigger="load delay:1s" hx-swap="outerHTML">Starting Pubs...</span></div>""", headers={"Cache-Control": "no-store"})

@app.post("/refresh_grants")
def trigger_grants(request: Request, db: Session = Depends(get_db)):
    require_admin(request)

    if not DB_AVAILABLE:
        return HTMLResponse("<span>DB Unavailable</span>")

    state = db.query(SystemState).filter_by(id=1).first()
    if state and state.status != "Ready" and "Error" not in state.status: return HTMLResponse("<span>Busy...</span>")
    set_system_status(db, "Initializing Grants...")
    thread = threading.Thread(target=perform_grant_scrape)
    thread.start()
    return HTMLResponse("""<div id="status-container" class="text-xs flex items-center gap-3"><span class="px-2 py-1 rounded bg-amber-500 font-medium animate-pulse" hx-get="/status_fragment" hx-trigger="load delay:1s" hx-swap="outerHTML">Starting Grants...</span></div>""", headers={"Cache-Control": "no-store"})

@app.post("/refresh_ai")
def trigger_ai(request: Request, db: Session = Depends(get_db)):
    require_admin(request)

    if not DB_AVAILABLE:
        return HTMLResponse("<span>DB Unavailable</span>")

    state = db.query(SystemState).filter_by(id=1).first()
    if state and state.status != "Ready" and "Error" not in state.status: return HTMLResponse("<span>Busy...</span>")
    set_system_status(db, "Initializing AI Profile...")
    thread = threading.Thread(target=perform_ai_profiling)
    thread.start()
    return HTMLResponse("""<div id="status-container" class="text-xs flex items-center gap-3"><span class="px-2 py-1 rounded bg-purple-500 text-white font-medium animate-pulse" hx-get="/status_fragment" hx-trigger="load delay:1s" hx-swap="outerHTML">Running AI...</span></div>""", headers={"Cache-Control": "no-store"})

@app.get("/investigator/{auth_id}", response_class=HTMLResponse)
def get_modal(request: Request, auth_id: str, db: Session = Depends(get_db)):
    require_auth(request)

    if not DB_AVAILABLE:
        return "<div>Database unavailable</div>"

    inv = db.query(InvestigatorStat).filter_by(id=auth_id).first()
    if not inv: return "<div>Not Found</div>"

    pmids = json.loads(inv.pmids_p2_json) if inv.pmids_p2_json else []
    papers = db.query(PaperDetail).filter(PaperDetail.pmid.in_(pmids)).all()
    papers.sort(key=lambda x: x.pub_date, reverse=True)
    grants = db.query(GrantDetail).filter_by(investigator_id=auth_id).order_by(GrantDetail.fiscal_year.desc()).all()

    # LOAD AI DATA
    techs = json.loads(inv.tech_json) if inv.tech_json else []
    pops = json.loads(inv.population_json) if inv.population_json else []

    html = f"""
    <div class="flex border-b border-gray-200">
        <button id="btn-profile" onclick="switchTab('profile')" class="tab-btn active px-4 py-2 text-sm focus:outline-none">Research Profile</button>
        <button id="btn-pubs" onclick="switchTab('pubs')" class="tab-btn px-4 py-2 text-sm focus:outline-none">Publications ({len(papers)})</button>
        <button id="btn-grants" onclick="switchTab('grants')" class="tab-btn px-4 py-2 text-sm focus:outline-none">NIH Grants ({len(grants)})</button>
    </div>

    <div id="tab-profile" class="tab-content p-6">
        <div class="space-y-6">
            <div>
                <h4 class="text-xs font-bold text-gray-500 uppercase tracking-wider mb-3">Key Technologies & Methods</h4>
                <div class="flex flex-wrap gap-2">
    """

    if techs:
        for t in techs:
            html += f'<span class="bg-indigo-100 text-indigo-700 border border-indigo-200 px-3 py-1 rounded-full text-sm font-medium shadow-sm">{t}</span>'
    else:
        html += '<span class="text-gray-400 italic text-sm">No technologies identified</span>'

    html += """
                </div>
            </div>
            <div>
                <h4 class="text-xs font-bold text-gray-500 uppercase tracking-wider mb-3">Primary Populations</h4>
                <div class="flex flex-wrap gap-2">
    """

    if pops:
        for p in pops:
            html += f'<span class="bg-emerald-100 text-emerald-800 border border-emerald-200 px-3 py-1 rounded-full text-sm font-medium shadow-sm">{p}</span>'
    else:
        html += '<span class="text-gray-400 italic text-sm">No specific population identified</span>'

    html += """
                </div>
            </div>
            <div class="mt-8 p-3 bg-gray-50 rounded border border-gray-100 text-center text-xs text-gray-400">
                Generated by AI based on analysis of recent publications and grants.
            </div>
        </div>
    </div>

    <div id="tab-pubs" class="tab-content hidden">
    """
    if not papers: html += "<div class='p-6 text-center text-gray-500'>No recent papers.</div>"
    for p in papers:
        date_str = p.pub_date.strftime('%b %Y') if p.pub_date else "N/A"
        html += f"""
        <div class='p-4 border-b hover:bg-slate-50 transition'>
            <div class='flex justify-between items-start gap-4'>
                <h4 class='font-semibold text-gray-900 leading-tight text-sm'>{p.title}</h4>
                <span class='shrink-0 bg-purple-100 text-purple-800 text-[10px] font-bold px-2 py-0.5 rounded'>RCR: {"%.2f" % p.rcr}</span>
            </div>
            <div class='mt-1 text-xs text-gray-600 flex items-center gap-2'>
                <span class='italic'>{p.journal}</span> <span class='text-gray-300'>&bull;</span> <span>{date_str}</span>
                <a href='https://pubmed.ncbi.nlm.nih.gov/{p.pmid}' target='_blank' class='ml-auto text-blue-500 hover:underline text-[10px] uppercase font-bold tracking-wider'>PubMed ↗</a>
            </div>
        </div>"""

    # TAB 3: GRANTS
    html += '</div><div id="tab-grants" class="tab-content hidden">'
    if not grants: html += "<div class='p-6 text-center text-gray-500'>No NIH grants found in the last 5 years.</div>"
    for g in grants:
        active_badge = "<span class='bg-green-100 text-green-800 text-[10px] px-2 py-0.5 rounded font-bold'>Active</span>" if g.is_active else "<span class='bg-gray-100 text-gray-500 text-[10px] px-2 py-0.5 rounded'>Past</span>"
        dates = f"{g.start_date.strftime('%Y')} - {g.end_date.strftime('%Y')}" if g.start_date and g.end_date else "Dates N/A"
        html += f"""
        <div class='p-4 border-b hover:bg-slate-50 transition'>
            <div class='flex justify-between items-start'>
                <div><h4 class='font-semibold text-gray-900 text-sm'>{g.project_title}</h4>
                <div class='text-xs text-gray-500 mt-1'>{g.project_num} <span class='mx-1'>|</span> {g.agency} <span class='mx-1'>|</span> {dates}</div></div>
                <div class='text-right'><div class='text-sm font-mono font-medium'>${"{:,.0f}".format(g.award_amount)}</div><div class='mt-1'>{active_badge}</div></div>
            </div>
        </div>"""
    html += "</div>"
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})

# --- SMART SEARCH ---

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors"""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot_product / (mag1 * mag2)

@app.post("/smart_search")
def smart_search(request: Request, query: str = Form(...), db: Session = Depends(get_db)):
    """
    Hybrid search combining BM25 lexical search with dual-embedding semantic search.

    AIDEV-NOTE: Hybrid ranking strategy
    - BM25 (30%): Lexical keyword matching for exact terms and proper nouns
    - Embeddings (70%): Semantic similarity
      - 75% weight: themes/populations embedding (primary semantic match)
      - 25% weight: publication/grant titles embedding (supplementary context)
    - Final score = 0.7 × semantic_score + 0.3 × bm25_score
    """
    require_auth(request)

    if not DB_AVAILABLE or not client:
        return {"error": "Search unavailable"}

    try:
        # 1. Normalize query with LLM
        logger.info(f"Smart search query: {query}")
        normalized_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Extract key research concepts from the user query. Focus on: research methods/technologies, populations studied, and medical conditions. Return 1-2 sentences."},
                {"role": "user", "content": query}
            ],
            max_completion_tokens=200
        )
        normalized_query = normalized_response.choices[0].message.content
        logger.info(f"Normalized query: {normalized_query}")

        # 2. Embed the normalized query
        embed_response = client.embeddings.create(
            input=normalized_query,
            model="text-embedding-3-large"
        )
        query_embedding = embed_response.data[0].embedding

        # 3. Load all investigators with dual embeddings
        all_investigators = db.query(InvestigatorStat).filter(
            InvestigatorStat.embedding_themes_pops != 'null',
            InvestigatorStat.embedding_themes_pops.isnot(None),
            InvestigatorStat.embedding_titles != 'null',
            InvestigatorStat.embedding_titles.isnot(None)
        ).all()

        # AIDEV-NOTE: Filter investigators
        # Only exclude probable misclassifications (<10% MGB affiliation in past 12 months)
        investigators = [
            inv for inv in all_investigators
            if (inv.probable_misclassification == 0 or inv.probable_misclassification is None)
        ]

        if not investigators:
            return {"results": [], "message": "No investigators with embeddings found. Please run 'python generate_embeddings.py' first."}

        # 4. Build BM25 index from investigator text profiles
        bm25_corpus = []
        inv_map = []  # Maps corpus index to investigator
        for inv in investigators:
            # Build text profile: name + themes + populations + titles
            text_parts = [inv.name.lower()]

            if inv.tech_json:
                try:
                    techs = json.loads(inv.tech_json)
                    text_parts.extend([t.lower() for t in techs])
                except:
                    pass

            if inv.population_json:
                try:
                    pops = json.loads(inv.population_json)
                    text_parts.extend([p.lower() for p in pops])
                except:
                    pass

            # Tokenize for BM25
            tokenized_doc = ' '.join(text_parts).split()
            bm25_corpus.append(tokenized_doc)
            inv_map.append(inv)

        # Initialize BM25
        bm25 = BM25Okapi(bm25_corpus)

        # Tokenize query
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)

        # Normalize BM25 scores to [0, 1]
        max_bm25 = max(bm25_scores) if bm25_scores.max() > 0 else 1.0
        bm25_scores_normalized = bm25_scores / max_bm25

        # 5. Compute semantic similarities and hybrid scores
        matches = []
        skipped = 0
        for idx, inv in enumerate(inv_map):
            try:
                # Load both embeddings
                if not inv.embedding_themes_pops or inv.embedding_themes_pops == 'null':
                    skipped += 1
                    continue
                if not inv.embedding_titles or inv.embedding_titles == 'null':
                    skipped += 1
                    continue

                themes_pops_embedding = json.loads(inv.embedding_themes_pops)
                titles_embedding = json.loads(inv.embedding_titles)

                # Compute separate semantic similarities
                similarity_themes_pops = cosine_similarity(query_embedding, themes_pops_embedding)
                similarity_titles = cosine_similarity(query_embedding, titles_embedding)

                # Weighted semantic blend: 20% themes/pops + 80% titles
                semantic_score = (0.2 * similarity_themes_pops) + (0.8 * similarity_titles)

                # Get BM25 score for this investigator
                bm25_score = float(bm25_scores_normalized[idx])

                # Hybrid final score: 70% semantic + 30% BM25
                final_score = (0.7 * semantic_score) + (0.3 * bm25_score)

                matches.append({
                    "id": inv.id,
                    "name": inv.name,
                    "technologies": json.loads(inv.tech_json) if inv.tech_json else [],
                    "populations": json.loads(inv.population_json) if inv.population_json else [],
                    "similarity_themes_pops": similarity_themes_pops,
                    "similarity_titles": similarity_titles,
                    "semantic_score": semantic_score,
                    "bm25_score": bm25_score,
                    "final_score": final_score
                })
            except Exception as e:
                logger.error(f"Error processing investigator {inv.id}: {e}")
                continue

        if skipped > 0:
            logger.warning(f"Skipped {skipped} investigators without dual embeddings. Run 'python generate_embeddings.py' to update.")

        # 6. Sort by final score DESCENDING (best to worst match)
        matches.sort(key=lambda x: x["final_score"], reverse=True)
        top_matches = matches[:10]

        # Convert scores to percentages for display
        for match in top_matches:
            match["match_percentage"] = round(match["final_score"] * 100, 1)

        # AIDEV-NOTE: Debug logging to terminal for match scores
        logger.info(f"🔍 Hybrid Search Results for query: '{query}'")
        logger.info(f"📊 Total matches found: {len(matches)}, returning top {len(top_matches)}")
        for i, match in enumerate(top_matches, 1):
            logger.info(f"  #{i}: {match['name']}")
            logger.info(f"      - Semantic score: {match.get('semantic_score', 0):.4f} (Themes: {match.get('similarity_themes_pops', 0):.4f}, Titles: {match.get('similarity_titles', 0):.4f})")
            logger.info(f"      - BM25 score: {match.get('bm25_score', 0):.4f}")
            logger.info(f"      - Final hybrid score: {match.get('final_score', 0):.4f}")
            logger.info(f"      - Match percentage: {match.get('match_percentage', 'MISSING')}%")

        return {"results": top_matches, "query": query, "normalized_query": normalized_query}

    except Exception as e:
        logger.error(f"Smart search error: {e}")
        return {"error": str(e), "results": []}

# --- 3D UMAP VISUALIZATION ---

@app.get("/umap_data")
def get_umap_data(request: Request, db: Session = Depends(get_db)):
    """
    Get 3D UMAP visualization data for all investigators.

    Returns JSON with:
    - nodes: Array of {id, name, x, y, z, cluster, funding, technologies, populations}
    - clusters: Cluster metadata
    """
    require_auth(request)

    if not DB_AVAILABLE:
        return {"error": "Database unavailable"}

    try:
        # AIDEV-NOTE: Check if UMAP data exists, automatically generate if missing
        # First check if any investigators have UMAP coordinates
        umap_count = db.query(InvestigatorStat).filter(
            InvestigatorStat.umap_x != 0.0,
            InvestigatorStat.umap_y != 0.0,
            InvestigatorStat.umap_z != 0.0
        ).count()

        if umap_count == 0:
            # Check if we have embeddings to generate UMAP from
            embedding_count = db.query(InvestigatorStat).filter(
                InvestigatorStat.embedding_themes_pops != None,
                InvestigatorStat.embedding_themes_pops != 'null',
                InvestigatorStat.embedding_titles != None,
                InvestigatorStat.embedding_titles != 'null'
            ).count()

            if embedding_count == 0:
                # No embeddings either - trigger embedding generation
                logger.info("No embeddings found, triggering embedding generation...")
                threading.Thread(target=perform_embedding_generation).start()
                return {"status": "generating", "message": "Generating AI embeddings and UMAP visualization. This takes 5-10 minutes. Please refresh in a few minutes."}
            else:
                # Have embeddings but no UMAP - trigger UMAP generation
                logger.info("Embeddings found but no UMAP, triggering UMAP generation...")
                threading.Thread(target=perform_umap_generation).start()
                return {"status": "generating", "message": "Generating UMAP visualization from existing embeddings. This takes 2-3 minutes. Please refresh shortly."}

        # Load all investigators with UMAP coordinates
        all_investigators = db.query(InvestigatorStat).filter(
            InvestigatorStat.umap_x != 0.0,
            InvestigatorStat.umap_y != 0.0,
            InvestigatorStat.umap_z != 0.0
        ).all()

        # AIDEV-NOTE: Filter investigators for visualization
        # 1. Exclude probable misclassifications (<10% MGB affiliation)
        # 2. Require embeddings and >2 papers for meaningful clustering
        investigators = [
            inv for inv in all_investigators
            if ((inv.probable_misclassification == 0 or inv.probable_misclassification is None) and
                inv.embedding_themes_pops and inv.embedding_themes_pops != 'null' and
                inv.embedding_titles and inv.embedding_titles != 'null' and
                (inv.n_p2 or 0) > 2)  # Need >2 papers for meaningful research profile
        ]

        if not investigators:
            return {"error": "No valid investigators for visualization after filtering."}

        # Build node data
        nodes = []
        for inv in investigators:
            nodes.append({
                "id": inv.id,
                "name": inv.name,
                "x": inv.umap_x,
                "y": inv.umap_y,
                "z": inv.umap_z,
                "cluster": inv.cluster_id,
                "funding": inv.funding_current or 0.0,
                "technologies": json.loads(inv.tech_json) if inv.tech_json else [],
                "populations": json.loads(inv.population_json) if inv.population_json else []
            })

        # Load cluster names from config table
        cluster_names = {}
        try:
            result = db.execute(text("SELECT value FROM config WHERE key = 'cluster_names'")).fetchone()
            if result:
                cluster_names = json.loads(result[0])
        except:
            logger.warning("⚠️  No cluster names found. Run 'python generate_cluster_names.py' to generate them.")

        # Get cluster statistics
        cluster_counts = {}
        for inv in investigators:
            cluster_id = inv.cluster_id
            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1

        clusters = [
            {
                "id": cluster_id,
                "count": count,
                "name": cluster_names.get(str(cluster_id), f"Cluster {cluster_id}")
            }
            for cluster_id, count in sorted(cluster_counts.items())
        ]

        logger.info(f"📊 Sending UMAP data: {len(nodes)} nodes, {len(clusters)} clusters")
        return {"nodes": nodes, "clusters": clusters, "cluster_names": cluster_names}

    except Exception as e:
        logger.error(f"UMAP data error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
