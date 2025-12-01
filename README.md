# DDash - MGB Psychiatry Research Dashboard

**Version 0.62** | © 2025 Roy Perlis and MGH CQH

A secure web dashboard for tracking research metrics across Massachusetts General Brigham (MGB) psychiatry departments.

## Features

- 📊 **Publication Metrics** - Automated tracking of PubMed publications with iCite RCR scores
- 💰 **Grant Funding** - NIH RePORTER integration for active and historical grants
- 🤖 **AI Research Profiles** - OpenAI-powered analysis of research focus areas
- ⚡ **Smart Search** - Embedding-based semantic search to find investigators by research expertise
- 🔐 **Secure Access** - Two-tier authentication (user/admin)
- ☁️ **Cloud-Ready** - Deploy to Render, Heroku, or any cloud platform
- 🛡️ **Graceful Degradation** - Maintenance mode if database is unavailable

## Quick Start

### Local Development

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment variables
cp .env.example .env
# Edit .env with your credentials

# 3. Run the application
python ddash2.py

# 4. Open http://localhost:8000
```

**Default login credentials** (change in production!):
- User: `user123`
- Admin: `admin123`
- PI123: `pi123`

## Deployment to Render.com

1. Fork/clone this repository
2. Create a new **Web Service** on Render
3. Connect your repository
4. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python ddash2.py`
5. Add environment variables:
   ```
   USER_PASSWORD=your-user-password
   ADMIN_PASSWORD=your-admin-password
   PI123_PASSWORD=your-pi123-password
   SESSION_SECRET=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
   OPENAI_API_KEY=sk-...
   ENTREZ_EMAIL=your@email.com
   ```
6. (Optional) Add PostgreSQL database for production use

## Configuration

All configuration is done via environment variables. See `.env.example` for a complete list.

### Required Variables

- `USER_PASSWORD` - Password for read-only access (view all data)
- `ADMIN_PASSWORD` - Password for admin access (can trigger data refreshes)
- `PI123_PASSWORD` - Password for search-focused view (simplified interface)
- `SESSION_SECRET` - Secret key for session encryption

### Optional Variables

- `OPENAI_API_KEY` - Enables AI research profiling (OpenAI GPT-4)
- `ENTREZ_EMAIL` - Your email for PubMed API (NCBI requirement)
- `DATABASE_URL` - Database connection (defaults to SQLite)
- `PORT` - Server port (defaults to 8000)

## User Roles

### Admin (Full Access)
- View investigator metrics
- Search and filter investigators (text search + smart search)
- View publication and grant details
- Access AI research profiles
- Refresh publication data from PubMed
- Refresh grant data from NIH RePORTER
- Trigger AI profiling updates

### User (Read-Only)
- View investigator metrics
- Search and filter investigators (text search + smart search)
- View publication and grant details
- Access AI research profiles

### PI123 (Search-Focused View)
- Simplified interface designed for finding investigators
- Smart search to find researchers by expertise
- View investigator names, technologies/methods, and study populations
- Text search to filter results by name

## Data Sources

- **Publications**: [PubMed/NCBI](https://pubmed.ncbi.nlm.nih.gov/)
- **Citation Metrics**: [iCite API](https://icite.od.nih.gov/)
- **Grant Data**: [NIH RePORTER](https://reporter.nih.gov/)
- **AI Analysis**: [OpenAI GPT-4](https://openai.com/)

## Technology Stack

- **Backend**: FastAPI, SQLAlchemy, Uvicorn
- **Frontend**: HTMX, Tailwind CSS, Jinja2
- **Database**: SQLite (local), PostgreSQL (production)
- **APIs**: BioPython (PubMed), OpenAI, NIH RePORTER

## Smart Search Setup

The Smart Search feature requires a one-time embedding generation step:

```bash
# Generate embeddings for all investigators
export OPENAI_API_KEY="sk-..."
python generate_embeddings.py
```

This analyzes each investigator's publications, grants, and research themes to enable semantic search. See [EMBEDDING_SETUP.md](EMBEDDING_SETUP.md) for detailed setup instructions.

**Example searches:**
- "AI for stroke in older adults using claims data"
- "Depression treatment in adolescents"
- "Machine learning for predicting suicide risk"

## Development

See [CLAUDE.md](CLAUDE.md) for detailed architecture documentation.

```bash
# Install dev dependencies
pip install -r requirements.txt

# Run with auto-reload
uvicorn ddash2:app --reload --host 0.0.0.0 --port 8000
```

## Security Notes

- ⚠️ **Change default passwords** before deploying
- 🔑 Never commit `.env` file to version control
- 🔒 Use strong, randomly generated `SESSION_SECRET`
- 🗄️ Use PostgreSQL for production (not SQLite)
- 🌐 Deploy behind HTTPS in production

## License

All rights reserved. © 2025 Roy Perlis and MGH Center for Quantitative Health (CQH).

## Support

For issues or questions, contact the development team.

---

**Built for Massachusetts General Brigham Psychiatry**
