"""
Arohi Backend — FastAPI (local dev, no AWS)

Run locally:
  python -m venv .venv && source .venv/bin/activate
  pip install fastapi uvicorn[standard] SQLAlchemy psycopg2-binary pydantic python-dotenv openai
  export $(grep -v '^#' .env | xargs)  # or set env vars manually
  uvicorn app:app --reload --port 8000

Suggested env (.env):
  DATABASE_URL=sqlite+pysqlite:///./arohi.db
  OPENAI_API_KEY=xxxx
  ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000
  ENVIRONMENT=dev
  AWS_S3_BUCKET=your-bucket-name
  AWS_S3_FOLDER_PREFIX=dev/
"""
from __future__ import annotations
import os, json, uuid, logging, time
from pathlib import Path
from typing import List, Optional
from contextlib import contextmanager
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Use Uvicorn's logger so messages show in the terminal
# (print-style logging often doesn't appear when running behind Uvicorn/Gunicorn)
LOGGER = logging.getLogger("uvicorn.error")

from fastapi import FastAPI, HTTPException, Body, Request, Response, BackgroundTasks, status
from collections import defaultdict, deque
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, String, Integer, Boolean, JSON, text, ForeignKey
from datetime import datetime, timedelta
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session

# ----- OpenAI (server-side only; never expose keys to frontend) -----
# We import the SDK and try to create a client from the OPENAI_API_KEY
# in your .env. If anything fails (no key, not installed, etc.), we
# set the client to None and the API will fall back to a safe default
# so local development keeps working.
try:
    from openai import OpenAI
except Exception as _import_err:
    LOGGER.exception("OpenAI SDK import failed")
    OpenAI = None

_openai_client = None
if OpenAI is not None:
    _openai_api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    try:
        if not _openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        LOGGER.info("OpenAI client init: attempting connection to api.openai.com")
        client = OpenAI(api_key=_openai_api_key)
        try:
            client.models.list()
        except Exception:
            LOGGER.exception("OpenAI models.list() failed during init")
            raise
        _openai_client = client
        LOGGER.info("OpenAI client init: success")
    except Exception as _e:
        LOGGER.exception("OpenAI client init failed (check OPENAI_API_KEY, network, package)")
        _openai_client = None

# ----- Config -----
# All configuration is read from environment variables (see .env / .env.example)
# so you can change behavior without touching the code. Reasonable defaults
# are provided so the project runs out-of-the-box with SQLite.
#
# Important: use an absolute path for the default SQLite DB so it stays
# consistent regardless of where you run `uvicorn` from (Windows/WSL, ngrok, etc.).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DB_PATH = _REPO_ROOT / "arohi.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite+pysqlite:///{_DEFAULT_DB_PATH}")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]
# Owner / brand config served to frontend
OWNER_PHONE = os.getenv("OWNER_PHONE", "+919999999999")
BRAND_NAME = os.getenv("BRAND_NAME", "Arohi's collection")
LOGO_URL = os.getenv("LOGO_URL", "")
# AI config
# Model and prompts used by the /ai/describe endpoint. You can tweak these
# in .env without changing code.
AI_MODEL = os.getenv("AI_MODEL", "gpt-4o")
AI_PROMPT_SYSTEM = os.getenv(
    "AI_PROMPT_SYSTEM",
    "You are a Product listing expert for a Shopify store that sells artificial jewelry in India. Identify materials and craftsmanship styles accurately.",
)
AI_PROMPT_USER = os.getenv(
    "AI_PROMPT_USER",
    (
        "Please analyze these image(s) of artificial jewelry to be listed on the Shopify store.\n"
        "Output must be STRICT JSON with keys: title, description, category, style_tag.\n"
        "A good listing uses a title and description that are relevant to the jewelry material and type, tailored for an Indian audience.\n"
        "Avoid any words implying precious metals or real gemstones (gold, silver, diamond, etc.). Neutral descriptors allowed: 'silver-tone', 'kundan', 'polki', 'oxidized', 'meenakari', 'pearl', 'american diamond (AD)', 'moissanite'.\n"
        "style_tag must capture the jewellery style / craftsmanship (e.g., Kundan, Oxidized Silver, American Diamond) using the closest match from this list: American Diamond, Antique Gold Plated, Beaded, German Silver, Kundan, Meenakari, Mirror Work, Moissanite, Oxidized Silver, Pearl, Polki, Rose Gold Plated, Temple, Terracotta, Thread Work. If unsure, respond with 'Not specified'.\n"
        "Category must be one of: Anklets, Bracelets, Brooches & Lapel Pins, Charms & Pendants, Earrings, Jewelry Sets, Necklaces, Rings.\n"
        "If a necklace is shown with matching earrings, choose 'Jewelry Sets'. If only a necklace is present, choose 'Necklaces'. If only earrings, choose 'Earrings'."
    ),
)
AI_DEBUG = os.getenv("AI_DEBUG", "0").lower() in {"1", "true", "yes"}
OBS_LOG_TIMING = os.getenv("OBS_LOG_TIMING", "1").lower() in {"1","true","yes"}

# Log key AI settings once on startup to help diagnose prod vs dev behaviour
try:
    LOGGER.info(
        "AI config: AI_DEBUG=%s DEBUG_AI=%s model=%s client_ready=%s",
        AI_DEBUG,
        os.getenv("DEBUG_AI"),
        os.getenv("AI_MODEL", ""),
        bool(_openai_client if '_openai_client' in globals() else None),
    )
except Exception:
    pass

JEWELRY_STYLE_OPTIONS = [
    "American Diamond",
    "Antique Gold Plated",
    "Beaded",
    "German Silver",
    "Kundan",
    "Meenakari",
    "Mirror Work",
    "Moissanite",
    "Oxidized Silver",
    "Pearl",
    "Polki",
    "Rose Gold Plated",
    "Temple",
    "Terracotta",
    "Thread Work",
]

_STYLE_LOOKUP = {val.lower(): val for val in JEWELRY_STYLE_OPTIONS}
_STYLE_SYNONYMS = {
    "ad": "American Diamond",
    "cz": "American Diamond",
    "cz stone": "American Diamond",
    "american diamond (ad)": "American Diamond",
    "oxidised": "Oxidized Silver",
    "oxidized": "Oxidized Silver",
    "oxidized silver": "Oxidized Silver",
    "oxidised silver": "Oxidized Silver",
    "kundan work": "Kundan",
    "polki work": "Polki",
    "meenakari work": "Meenakari",
    "mirror": "Mirror Work",
    "mirror work": "Mirror Work",
    "beads": "Beaded",
    "beaded": "Beaded",
    "thread": "Thread Work",
    "thread work": "Thread Work",
    "temple jewellery": "Temple",
    "temple jewelry": "Temple",
    "german silver": "German Silver",
    "antique gold": "Antique Gold Plated",
    "antique finish": "Antique Gold Plated",
    "rose gold": "Rose Gold Plated",
    "pearl work": "Pearl",
    "terracotta": "Terracotta",
    "moissanite": "Moissanite",
}


def _normalize_style_tag(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    key = raw.lower()
    if key in {"not specified", "unspecified", "n/a", "na", "none"}:
        return None
    if key in _STYLE_SYNONYMS:
        return _STYLE_SYNONYMS[key]
    if key in _STYLE_LOOKUP:
        return _STYLE_LOOKUP[key]
    simplified = key.replace("jewellery", "").replace("jewelry", "").strip()
    if simplified in _STYLE_SYNONYMS:
        return _STYLE_SYNONYMS[simplified]
    if simplified in _STYLE_LOOKUP:
        return _STYLE_LOOKUP[simplified]
    for option in JEWELRY_STYLE_OPTIONS:
        opt_lower = option.lower()
        if key in opt_lower or opt_lower in key or simplified in opt_lower or opt_lower in simplified:
            return option
    return raw.title()

# Robust env parsers (tolerate accidental inline comments like "700: warn for slow calls")
def _int_env(name: str, default: int) -> int:
    try:
        raw = str(os.getenv(name, str(default)))
        import re
        m = re.search(r"\d+", raw)
        return int(m.group(0)) if m else int(default)
    except Exception:
        return int(default)

def _float_env(name: str, default: float) -> float:
    try:
        raw = str(os.getenv(name, str(default)))
        # take first token before space/colon
        raw = raw.split()[0].split(":")[0]
        return float(raw)
    except Exception:
        return float(default)

OBS_SLOW_MS = _int_env("OBS_SLOW_MS", 700)
SENTRY_DSN = os.getenv("SENTRY_DSN", "")
SENTRY_TRACES = _float_env("SENTRY_TRACES", 0.0)
OBS_WINDOW_SEC = _int_env("OBS_WINDOW_SEC", 300)  # 5 minutes
RETAIN_EVENTS_DAYS = _int_env("RETAIN_EVENTS_DAYS", 30)
RETAIN_REQ_STATS_DAYS = _int_env("RETAIN_REQ_STATS_DAYS", 7)

# In-memory recent request timings per route (windowed)
_REQ_STATS: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

# Image storage provider (local base64 | cloudflare | s3 [future])
IMAGE_PROVIDER = os.getenv("IMAGE_PROVIDER", "local").lower()
# Cloudflare Images config (used when IMAGE_PROVIDER=cloudflare)
CF_IMAGES_ACCOUNT_ID = os.getenv("CF_IMAGES_ACCOUNT_ID", "")
CF_IMAGES_ACCOUNT_HASH = os.getenv("CF_IMAGES_ACCOUNT_HASH", "")  # for delivery URLs
CF_IMAGES_API_TOKEN = os.getenv("CF_IMAGES_API_TOKEN", "")
CF_IMAGES_VARIANT_FULL = os.getenv("CF_IMAGES_VARIANT_FULL", "public")
CF_IMAGES_VARIANT_THUMB = os.getenv("CF_IMAGES_VARIANT_THUMB", "thumb")
# S3 config (used when IMAGE_PROVIDER=s3). Use IAM role or access keys.
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET", "")
AWS_S3_REGION = os.getenv("AWS_S3_REGION", os.getenv("AWS_REGION", ""))
# Optional CDN domain (e.g., CloudFront) to prefix keys; otherwise uses S3 public URL
AWS_S3_CDN_BASE_URL = os.getenv("AWS_S3_CDN_BASE_URL", "")
# Environment-specific S3 folder prefix (dev/staging/prod)
ENVIRONMENT = os.getenv("ENVIRONMENT", "dev").lower()
AWS_S3_FOLDER_PREFIX = os.getenv("AWS_S3_FOLDER_PREFIX", f"{ENVIRONMENT}/")

engine = create_engine(DATABASE_URL, future=True)
# SessionLocal is a factory that gives us a new DB session for each request
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
# Base is the parent class for our ORM models (Product, Order, ...)
Base = declarative_base()


# ----- DB Models -----
class Product(Base):
    """Product catalog item stored in the database.

    Notes:
    - price is stored in paise (integer) to avoid floating point issues.
    - images is a JSON array of URLs/base64 strings.
    """
    __tablename__ = "products"
    id = Column(String, primary_key=True)  # e.g., Earrings-123456
    title = Column(String, nullable=False)
    description = Column(String, nullable=False)
    category = Column(String, index=True, nullable=False)
    style_tag = Column(String, index=True, nullable=True)
    price_in_paise = Column(Integer, nullable=False)
    cost_in_paise = Column(Integer, nullable=False, default=0)
    qty = Column(Integer, nullable=False, default=0)
    available = Column(Boolean, nullable=False, default=True)
    images = Column(JSON, nullable=False, default=list)  # [url, ...]
    images_small = Column(JSON, nullable=True, default=list)
    status = Column(String, nullable=False, default="ready")  # ready | processing | failed
    processing_error = Column(String, nullable=True)
    pending_images = Column(JSON, nullable=True, default=list)
    created_at = Column(String, server_default=text("CURRENT_TIMESTAMP"))
    updated_at = Column(String, server_default=text("CURRENT_TIMESTAMP"))

class Order(Base):
    """A customer shortlist or order.

    We model a simple status field and a 1..N relationship to OrderItem.
    """
    __tablename__ = "orders"
    id = Column(String, primary_key=True)  # e.g., ORD-xxx
    customer_phone = Column(String, nullable=False)
    status = Column(String, nullable=False, default="pending")  # pending | confirmed | cancelled
    created_at = Column(String, server_default=text("CURRENT_TIMESTAMP"))
    confirmed_at = Column(String, nullable=True)
    removed_items = Column(JSON, nullable=False, default=list)
    session_id = Column(String, nullable=True)
    items = relationship("OrderItem", back_populates="order", cascade="all, delete-orphan")

class OrderItem(Base):
    """Line items connecting Orders to Products."""
    __tablename__ = "order_items"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    order_id = Column(String, ForeignKey("orders.id"), nullable=False)
    product_id = Column(String, ForeignKey("products.id"), nullable=False)
    qty = Column(Integer, nullable=False, default=1)
    order = relationship("Order", back_populates="items")
    product = relationship("Product")

class Event(Base):
    __tablename__ = "events"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, index=True, nullable=True)
    kind = Column(String, index=True, nullable=False)  # add_to_cart | remove_from_cart | send_order | confirm_order
    payload = Column(JSON, nullable=True, default=dict)
    created_at = Column(String, server_default=text("CURRENT_TIMESTAMP"))

# Create tables automatically on startup for local dev (quick and simple).
# For production, prefer Alembic migrations instead.
Base.metadata.create_all(engine)

# Lightweight schema patch for SQLite: add columns if missing, ensure helpful indexes
_dialect = engine.dialect.name
try:
    with engine.begin() as conn:
        if _dialect == "sqlite":
            # SQLite lacks IF NOT EXISTS on ALTER TABLE, so inspect schema manually
            cols_products = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(products)").fetchall()}
            if "cost_in_paise" not in cols_products:
                conn.exec_driver_sql("ALTER TABLE products ADD COLUMN cost_in_paise INTEGER DEFAULT 0")
            if "images_small" not in cols_products:
                conn.exec_driver_sql("ALTER TABLE products ADD COLUMN images_small JSON")
            if "status" not in cols_products:
                conn.exec_driver_sql("ALTER TABLE products ADD COLUMN status TEXT DEFAULT 'ready'")
            if "processing_error" not in cols_products:
                conn.exec_driver_sql("ALTER TABLE products ADD COLUMN processing_error TEXT")
            if "pending_images" not in cols_products:
                conn.exec_driver_sql("ALTER TABLE products ADD COLUMN pending_images JSON")
            if "style_tag" not in cols_products:
                conn.exec_driver_sql("ALTER TABLE products ADD COLUMN style_tag TEXT")

            cols_orders = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(orders)").fetchall()}
            if "confirmed_at" not in cols_orders:
                conn.exec_driver_sql("ALTER TABLE orders ADD COLUMN confirmed_at TEXT")
            if "removed_items" not in cols_orders:
                conn.exec_driver_sql("ALTER TABLE orders ADD COLUMN removed_items JSON DEFAULT '[]'")
            if "session_id" not in cols_orders:
                conn.exec_driver_sql("ALTER TABLE orders ADD COLUMN session_id TEXT")
        else:
            # Engines like Postgres support IF NOT EXISTS directly
            conn.exec_driver_sql("ALTER TABLE products ADD COLUMN IF NOT EXISTS cost_in_paise INTEGER DEFAULT 0")
            conn.exec_driver_sql("ALTER TABLE products ADD COLUMN IF NOT EXISTS images_small JSON")
            conn.exec_driver_sql("ALTER TABLE products ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'ready'")
            conn.exec_driver_sql("ALTER TABLE products ADD COLUMN IF NOT EXISTS processing_error TEXT")
            conn.exec_driver_sql("ALTER TABLE products ADD COLUMN IF NOT EXISTS pending_images JSON")
            conn.exec_driver_sql("ALTER TABLE products ADD COLUMN IF NOT EXISTS style_tag TEXT")
            conn.exec_driver_sql("ALTER TABLE orders ADD COLUMN IF NOT EXISTS confirmed_at TEXT")
            conn.exec_driver_sql("ALTER TABLE orders ADD COLUMN IF NOT EXISTS removed_items JSON DEFAULT '[]'::json")
            conn.exec_driver_sql("ALTER TABLE orders ADD COLUMN IF NOT EXISTS session_id TEXT")

        # Helpful indexes (supported on both SQLite and Postgres)
        conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_products_created_at ON products(created_at)")
        conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_products_category ON products(category)")
        conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_products_available_qty ON products(available, qty)")
        conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_orders_session_created ON orders(session_id, created_at)")

        # Create events table if missing
        payload_type = "JSON"
        conn.exec_driver_sql(
            f"""
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                kind TEXT NOT NULL,
                payload {payload_type},
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id)")
except Exception as _e:
    LOGGER.warning("Schema patch skipped/failure: %s", _e)

# ---- Image utilities (optional Pillow) ----
def _make_thumbnail_dataurl(data_url: str, max_size=(640, 640), quality=70) -> str:
    try:
        if not (data_url or "").startswith("data:image"):
            return data_url
        import base64, io
        from PIL import Image
        head, b64 = data_url.split(",", 1)
        raw = base64.b64decode(b64)
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        im.thumbnail(max_size, Image.LANCZOS)
        out = io.BytesIO()
        im.save(out, format="JPEG", quality=quality, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode("ascii")
    except Exception:
        return data_url

def make_thumbnails(urls):
    try:
        return [_make_thumbnail_dataurl(u) for u in (urls or [])]
    except Exception:
        return urls or []

def admin_backfill_thumbnails(request: Request, token: Optional[str] = None, limit: Optional[int] = None, dry_run: bool = False):
    """Generate images_small for products that are missing them.

    Security: guarded by ADMIN_TOKEN env var. Provide via ?token=... or header X-Admin-Token.
    Only intended for local/UAT. For production, move to a proper admin auth.
    """
    admin_token = (os.getenv("ADMIN_TOKEN", "") or "").strip()
    provided = (token or "").strip()
    # Support Authorization: Bearer <token> and X-Admin-Token
    if not provided:
        auth = request.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            provided = auth.split(" ", 1)[1].strip()
    if not provided:
        provided = request.headers.get("X-Admin-Token", "").strip()
    # Enforce only if ADMIN_TOKEN is set
    if admin_token:
        if not provided or provided != admin_token:
            raise HTTPException(401, "Unauthorized: invalid or missing admin token")
    updated = 0
    scanned = 0
    with session_scope() as db:
        qry = db.query(Product).order_by(Product.created_at.desc())
        if limit:
            qry = qry.limit(int(limit))
        for p in qry.all():
            scanned += 1
            imgs = p.images or []
            small = p.images_small or []
            # decide whether to refresh: missing, length mismatch, or any small missing
            needs = (not small) or (len(small) != len(imgs))
            if not needs:
                continue
            thumbs = make_thumbnails(imgs)
            if dry_run:
                updated += 1
                continue
            p.images_small = thumbs
            updated += 1
        if not dry_run:
            db.flush()
    return {"scanned": scanned, "updated": updated, "dry_run": dry_run}

# ----- Cloudflare Images helper -----
def cf_images_upload_one(data_url_or_url: str):
    if not (CF_IMAGES_ACCOUNT_ID and CF_IMAGES_ACCOUNT_HASH and CF_IMAGES_API_TOKEN):
        raise RuntimeError("Cloudflare Images env not configured")
    try:
        import requests, base64
    except Exception as e:
        raise RuntimeError("requests not installed; pip install requests") from e
    endpoint = f"https://api.cloudflare.com/client/v4/accounts/{CF_IMAGES_ACCOUNT_ID}/images/v1"
    headers = {"Authorization": f"Bearer {CF_IMAGES_API_TOKEN}"}
    files = None
    data = {}
    val = data_url_or_url or ""
    if val.startswith("data:image"):
        head, b64 = val.split(",", 1)
        content_type = head.split(":",1)[1].split(";",1)[0]
        binary = base64.b64decode(b64)
        files = {"file": ("upload.jpg", binary, content_type or "image/jpeg")}
    else:
        data = {"url": val}
    r = requests.post(endpoint, headers=headers, data=data, files=files, timeout=30)
    r.raise_for_status()
    resp = r.json()
    if not resp.get("success"):
        raise RuntimeError(f"Cloudflare upload failed: {resp}")
    image_id = resp.get("result", {}).get("id")
    full = f"https://imagedelivery.net/{CF_IMAGES_ACCOUNT_HASH}/{image_id}/{CF_IMAGES_VARIANT_FULL}"
    thumb = f"https://imagedelivery.net/{CF_IMAGES_ACCOUNT_HASH}/{image_id}/{CF_IMAGES_VARIANT_THUMB}"
    return full, thumb

def _dataurl_to_bytes_mime(val: str):
    import base64
    head, b64 = val.split(",", 1)
    mime = head.split(":",1)[1].split(";",1)[0]
    return base64.b64decode(b64), (mime or "image/jpeg")

def _jpeg_resize_bytes(data: bytes, max_size=(1600,1600), quality=85) -> bytes:
    try:
        from PIL import Image
        import io
        im = Image.open(io.BytesIO(data)).convert("RGB")
        im.thumbnail(max_size, Image.LANCZOS)
        out = io.BytesIO()
        im.save(out, format="JPEG", quality=quality, optimize=True)
        return out.getvalue()
    except Exception:
        return data

def s3_upload_bytes(key: str, data: bytes, content_type: str, cache_control: str = "public, max-age=31536000") -> str:
    if not AWS_S3_BUCKET:
        raise RuntimeError("AWS_S3_BUCKET not set")
    try:
        import boto3
    except Exception as e:
        raise RuntimeError("boto3 not installed; pip install boto3") from e
    extra = {"ContentType": content_type}
    if cache_control:
        extra["CacheControl"] = cache_control
    s3 = boto3.client("s3", region_name=AWS_S3_REGION or None)
    t0 = time.perf_counter() if OBS_LOG_TIMING else None
    try:
        s3.put_object(Bucket=AWS_S3_BUCKET, Key=key, Body=data, **extra)
    except Exception as e:
        LOGGER.exception("S3 put_object failed (bucket=%s key=%s)", AWS_S3_BUCKET, key)
        raise
    finally:
        if t0 is not None:
            LOGGER.info(
                "S3 put_object key=%s size=%d latency=%.1fms",
                key,
                len(data or b""),
                (time.perf_counter() - t0) * 1000.0,
            )
    if AWS_S3_CDN_BASE_URL:
        return f"{AWS_S3_CDN_BASE_URL.rstrip('/')}/{key}"
    if AWS_S3_REGION:
        return f"https://{AWS_S3_BUCKET}.s3.{AWS_S3_REGION}.amazonaws.com/{key}"
    # regionless fallback
    return f"https://{AWS_S3_BUCKET}.s3.amazonaws.com/{key}"

def s3_upload_images(product_id: str, images: list[str]) -> tuple[list[str], list[str]]:
    import uuid
    LOGGER.info(
        "S3 upload start: product=%s images=%d prefix=%s",
        product_id,
        len(images or []),
        AWS_S3_FOLDER_PREFIX,
    )
    full_urls, thumb_urls = [], []
    for val in (images or []):
        name = uuid.uuid4().hex
        # Use environment-specific folder prefix
        key_full = f"{AWS_S3_FOLDER_PREFIX}products/{product_id}/{name}.jpg"
        key_thumb = f"{AWS_S3_FOLDER_PREFIX}products/{product_id}/{name}.thumb.jpg"
        if (val or "").startswith("data:image"):
            data, mime = _dataurl_to_bytes_mime(val)
        else:
            # fetch bytes
            try:
                import requests
                r = requests.get(val, timeout=20)
                r.raise_for_status()
                data = r.content
                mime = r.headers.get("Content-Type", "image/jpeg")
            except Exception:
                data, mime = b"", "image/jpeg"
        if not data:
            LOGGER.warning("Skipping upload for product %s (no data for image source)", product_id)
            full_urls.append(val)
            thumb_candidate = _guess_thumb_url_from_full(val) or make_thumbnails([val])[0]
            thumb_urls.append(thumb_candidate)
            continue
        # re-encode originals to jpeg at 85 to cap size
        t_resize = time.perf_counter() if OBS_LOG_TIMING else None
        full_jpeg = _jpeg_resize_bytes(data, max_size=(2000,2000), quality=85)
        thumb_jpeg = _jpeg_resize_bytes(data, max_size=(640,640), quality=75)
        if t_resize is not None:
            LOGGER.info(
                "Image reencode latency=%.1fms size_in=%d size_full=%d size_thumb=%d",
                (time.perf_counter() - t_resize) * 1000.0,
                len(data or b""),
                len(full_jpeg or b""),
                len(thumb_jpeg or b""),
            )
        try:
            full_url = s3_upload_bytes(key_full, full_jpeg, "image/jpeg")
            thumb_url = s3_upload_bytes(key_thumb, thumb_jpeg, "image/jpeg")
        except Exception:
            LOGGER.exception("S3 upload failed for product %s (key=%s)", product_id, key_full)
            raise
        full_urls.append(full_url)
        thumb_urls.append(thumb_url)
    LOGGER.info("S3 upload complete: product=%s uploaded=%d", product_id, len(full_urls))
    return full_urls, thumb_urls

def maybe_upload_to_cdn(images: list[str], product_id: str = "") -> tuple[list[str], list[str]]:
    imgs = images or []
    if IMAGE_PROVIDER == "cloudflare":
        full, thumbs = [], []
        for u in imgs:
            try:
                f, t = cf_images_upload_one(u)
            except Exception as e:
                LOGGER.error("CF upload failed; keeping original: %s", e)
                f, t = u, _make_thumbnail_dataurl(u)
            full.append(f)
            thumbs.append(t)
        return full, thumbs
    if IMAGE_PROVIDER == "s3":
        return s3_upload_images(product_id or "misc", imgs)
    # default local
    return imgs, make_thumbnails(imgs)


def _normalize_pending_images(raw: Optional[list]) -> list[dict[str, Optional[str]]]:
    items: list[dict[str, Optional[str]]] = []
    for item in list(raw or []):
        if isinstance(item, dict):
            src = item.get('src') or item.get('url') or item.get('image')
            small = item.get('small') or item.get('thumb')
        else:
            src = item
            small = None
        if not src:
            continue
        items.append({'src': src, 'small': small})
    return items


def _guess_thumb_url_from_full(url: str) -> Optional[str]:
    if not url:
        return None
    if url.endswith('.thumb.jpg') or url.endswith('.thumb.jpeg'):
        return url
    if url.startswith('http'):
        base, sep, query = url.partition('?')
        suffix = f'?{query}' if sep == '?' else ''
        if base.endswith('.jpg'):
            return f"{base[:-4]}.thumb.jpg{suffix}"
        if base.endswith('.jpeg'):
            return f"{base[:-5]}.thumb.jpeg{suffix}"
    return None


def _process_product_images_async(product_id: str, images: Optional[list] = None):
    try:
        with session_scope() as db:
            prod: Product = db.get(Product, product_id)
            if not prod:
                return
            pending_items = _normalize_pending_images(images if images is not None else prod.pending_images)
            existing_full = list(prod.images or [])
            existing_small = list(prod.images_small or [])
            existing_lookup: dict[str, list[Optional[str]]] = {}
            for idx, full_url in enumerate(existing_full):
                if not full_url:
                    continue
                if isinstance(full_url, str) and full_url.startswith('data:image'):
                    continue
                thumb_url = existing_small[idx] if idx < len(existing_small) else None
                existing_lookup.setdefault(full_url, []).append(thumb_url)
        if not pending_items:
            with session_scope() as db:
                prod: Product = db.get(Product, product_id)
                if not prod:
                    return
                prod.images = []
                prod.images_small = []
                prod.pending_images = []
                prod.status = 'ready'
                prod.processing_error = None
            return

        upload_sources: list[str] = []
        for item in pending_items:
            src = item['src']
            if src and not src.startswith('data:image') and existing_lookup.get(src):
                continue
            if src and src.startswith('data:image'):
                upload_sources.append(src)
            else:
                existing_lookup.setdefault(src, []).append(item.get('small'))

        uploaded_full: list[str] = []
        uploaded_thumbs: list[str] = []
        if upload_sources:
            t_img = time.perf_counter() if OBS_LOG_TIMING else None
            uploaded_full, uploaded_thumbs = maybe_upload_to_cdn(upload_sources, product_id)
            if t_img is not None:
                LOGGER.info(
                    'Products upload processed images=%d provider=%s latency=%.1fms',
                    len(upload_sources),
                    IMAGE_PROVIDER,
                    (time.perf_counter() - t_img) * 1000.0,
                )
        upload_iter = iter(zip(uploaded_full, uploaded_thumbs))

        final_full: list[str] = []
        final_small: list[str] = []
        for item in pending_items:
            src = item['src']
            small_hint = item.get('small')
            if existing_lookup.get(src):
                small_list = existing_lookup[src]
                small_val = small_list.pop(0) if small_list else None
                final_full.append(src)
                final_small.append(
                    small_val
                    or small_hint
                    or _guess_thumb_url_from_full(src)
                    or make_thumbnails([src])[0]
                )
            else:
                try:
                    full_val, thumb_val = next(upload_iter)
                except StopIteration:
                    full_val = src
                    thumb_val = make_thumbnails([src])[0]
                final_full.append(full_val)
                final_small.append(thumb_val)

        with session_scope() as db:
            prod: Product = db.get(Product, product_id)
            if not prod:
                return
            prod.images = final_full
            prod.images_small = final_small
            prod.pending_images = []
            prod.status = 'ready'
            prod.processing_error = None
    except Exception as exc:
        LOGGER.exception('Background image processing failed for %s', product_id)
        with session_scope() as db:
            prod: Product = db.get(Product, product_id)
            if prod:
                prod.processing_error = str(exc)[:400]
                prod.status = 'failed'
@contextmanager
def session_scope():
    """Small helper to get a DB session with automatic commit/rollback.

    Usage:
        with session_scope() as db:
            ... use db session ...
    """
    db: Session = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

# ----- Schemas -----
# Pydantic models validate input and shape output for the API.
# "In" models describe what the client sends, "Out" models what we return.
class ProductIn(BaseModel):
    id: str
    title: str
    description: str
    category: str
    style_tag: Optional[str] = None
    price: int = Field(..., ge=0, description="Price in INR whole rupees")
    cost: int = Field(0, ge=0, description="Optional cost in INR whole rupees")
    qty: int = Field(..., ge=0)
    available: bool = True
    images: List[str] = []

class ProductOut(BaseModel):
    id: str
    title: str
    description: str
    category: str
    style_tag: Optional[str] = None
    price: int
    cost: int
    qty: int
    available: bool
    images: List[str]
    status: str = "ready"
    processing_error: Optional[str] = None

class ProductListOut(BaseModel):
    items: List[ProductOut]
    total: Optional[int] = None
    next_offset: Optional[int] = None


def _product_to_out(m: Product, *, include_processing_error: bool = True) -> ProductOut:
    imgs = (m.images_small or []) or (m.images or [])
    return ProductOut(
        id=m.id,
        title=m.title,
        description=m.description,
        category=m.category,
        style_tag=_normalize_style_tag(getattr(m, 'style_tag', None)),
        price=m.price_in_paise // 100,
        cost=(getattr(m, 'cost_in_paise', 0) or 0) // 100,
        qty=m.qty,
        available=m.available,
        images=imgs,
        status=getattr(m, 'status', 'ready') or 'ready',
        processing_error=(m.processing_error if include_processing_error else None),
    )

class OrderCreate(BaseModel):
    items: List[str]  # product IDs; fixed quantity = 1 per item
    customer_phone: str
    session_id: Optional[str] = None

class OrderOut(BaseModel):
    id: str; status: str; customer_phone: str; items: List[str]
    created_at: Optional[str] = None
    confirmed_at: Optional[str] = None
    removed_items: Optional[List[str]] = []
    created_at: Optional[str] = None

class DescribeRequest(BaseModel):
    """Request body for /ai/describe.

    The frontend sends one or more image URLs (can be data: URLs)
    and the server asks the AI model to generate metadata.
    """
    image_urls: List[str] = Field(default_factory=list)

class DescribeResponse(BaseModel):
    """Response body from /ai/describe: strictly JSON fields used by the UI."""
    title: str
    description: str
    category: str
    style_tag: Optional[str] = None

# ----- FastAPI -----
app = FastAPI(title="Arohi Backend", version="0.1.0")
# CORS allows the browser-based frontend (different port) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOWED_ORIGINS == ["*"] else ALLOWED_ORIGINS,
    allow_credentials=ALLOWED_ORIGINS != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register admin endpoints defined above (after app is created)
app.post("/admin/backfill_thumbnails")(admin_backfill_thumbnails)

if OBS_LOG_TIMING:
    @app.middleware("http")
    async def _timing_mw(request: Request, call_next):
        import time
        t0 = time.perf_counter()
        resp = await call_next(request)
        dt_ms = int((time.perf_counter() - t0) * 1000)
        try:
            log_fn = LOGGER.warning if dt_ms >= OBS_SLOW_MS else LOGGER.info
            log_fn("%s %s -> %s %dms", request.method, request.url.path, getattr(resp, "status_code", "?"), dt_ms)
            # record to in-memory stats (window OBS_WINDOW_SEC)
            now = time.time()
            dq = _REQ_STATS[request.url.path]
            dq.append((now, dt_ms))
            # trim by time window
            while dq and (now - dq[0][0]) > OBS_WINDOW_SEC:
                dq.popleft()
        except Exception:
            pass
        return resp

# ---- Admin auth helper ----
def _extract_admin_token(request: Request, token_query: Optional[str] = None) -> str:
    provided = (token_query or "").strip()
    if not provided:
        auth = request.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            provided = auth.split(" ", 1)[1].strip()
    if not provided:
        provided = request.headers.get("X-Admin-Token", "").strip()
    if not provided:
        provided = request.query_params.get("token", "").strip()
    return provided

def require_admin(request: Request, token_query: Optional[str] = None):
    admin_token = (os.getenv("ADMIN_TOKEN", "") or "").strip()
    if not admin_token:
        return  # open in local/dev when no token configured
    provided = _extract_admin_token(request, token_query)
    if not provided or provided != admin_token:
        raise HTTPException(401, "Unauthorized: invalid or missing admin token")

# Optional Sentry SDK (errors + traces)
try:
    if SENTRY_DSN:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        sentry_sdk.init(dsn=SENTRY_DSN, integrations=[FastApiIntegration()], traces_sample_rate=SENTRY_TRACES)
        LOGGER.info("Sentry initialized (traces=%s)", SENTRY_TRACES)
except Exception as _e:
    LOGGER.warning("Sentry init failed: %s", _e)

@app.get("/health/obs")
def health_obs():
    # recent request stats
    req = []
    for path, dq in list(_REQ_STATS.items()):
        if not dq:
            continue
        vals = [ms for _, ms in dq]
        n = len(vals)
        avg = sum(vals) / n
        # p95
        sv = sorted(vals)
        import math
        p95 = sv[max(0, min(n-1, math.ceil(0.95*n)-1))]
        req.append({"path": path, "count": n, "avg_ms": round(avg, 1), "p95_ms": p95})
    req.sort(key=lambda x: x["path"])  # stable order

    # events in last 24h
    events = {"total": 0}
    try:
        with engine.connect() as conn:
            if _dialect == "sqlite":
                sql = "SELECT kind, COUNT(*) FROM events WHERE created_at >= datetime('now','-24 hours') GROUP BY kind"
            else:
                sql = "SELECT kind, COUNT(*) FROM events WHERE created_at::timestamp >= (CURRENT_TIMESTAMP - INTERVAL '24 hours') GROUP BY kind"
            rows = conn.exec_driver_sql(sql).fetchall()
            total = 0
            for kind, cnt in rows:
                events[str(kind)] = int(cnt)
                total += int(cnt)
            events["total"] = total
    except Exception as e:
        events = {"error": str(e)}

    return {"ok": True, "window_sec": OBS_WINDOW_SEC, "requests": req, "events_24h": events}


# ---- Long-term request stats (per-minute aggregates) ----
# SQLite instrumentation is disabled for non-SQLite engines.
if _dialect == "sqlite":
    try:
        with engine.connect() as conn:
            conn.exec_driver_sql(
                """
                CREATE TABLE IF NOT EXISTS req_stats (
                  ts_min TEXT NOT NULL,
                  path   TEXT NOT NULL,
                  count  INTEGER NOT NULL,
                  avg_ms REAL NOT NULL,
                  p95_ms REAL NOT NULL,
                  PRIMARY KEY (ts_min, path)
                )
                """
            )
    except Exception as _e:
        LOGGER.warning("Failed to ensure req_stats table: %s", _e)
else:
    LOGGER.info("SQLite req_stats table not created for dialect %s", _dialect)


async def _aggregate_stats_loop():
    """Every minute, write per-minute aggregates into req_stats.

    Aggregates are computed from the in-memory _REQ_STATS window for the past minute.
    """
    if _dialect != "sqlite":
        # Postgres/MySQL deployments rely on external APM; skip local persistence.
        while True:
            await asyncio.sleep(3600)
        return
    import time, math
    while True:
        try:
            now = time.time()
            # compute stats for the last full minute bucket
            # use current minute (floor) as bucket timestamp
            bucket_ts = int(now // 60 * 60)
            rows = []
            for path, dq in list(_REQ_STATS.items()):
                # values in the last minute
                vals = [ms for t, ms in dq if t >= bucket_ts]
                if not vals:
                    continue
                n = len(vals)
                avg = sum(vals) / n
                sv = sorted(vals)
                p95 = sv[max(0, min(n-1, math.ceil(0.95*n)-1))]
                rows.append((bucket_ts, path, n, round(avg,1), p95))
            if rows:
                # write into SQLite
                with engine.begin() as conn:
                    for ts, path, cnt, avg_ms, p95_ms in rows:
                        ts_min_iso = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:00')
                        conn.exec_driver_sql(
                            "INSERT OR REPLACE INTO req_stats (ts_min, path, count, avg_ms, p95_ms) VALUES (?,?,?,?,?)",
                            (ts_min_iso, path, cnt, avg_ms, p95_ms),
                        )
        except Exception as e:
            try:
                LOGGER.warning("stats loop error: %s", e)
            except Exception:
                pass
        # sleep until next minute boundary
        now2 = time.time()
        sleep_sec = max(5.0, 60.0 - (now2 % 60.0))
        await asyncio.sleep(sleep_sec)


@app.on_event("startup")
async def _start_stats_loop():
    try:
        asyncio.create_task(_aggregate_stats_loop())
        LOGGER.info("Started stats aggregation loop")
    except Exception as e:
        LOGGER.warning("Failed to start stats loop: %s", e)
    # retention loop (every 6 hours)
    async def _retention_loop():
        import time
        while True:
            try:
                with engine.begin() as conn:
                    if _dialect == "sqlite":
                        conn.exec_driver_sql(
                            "DELETE FROM req_stats WHERE ts_min < datetime('now', ?)",
                            (f"-{RETAIN_REQ_STATS_DAYS} days",),
                        )
                        conn.exec_driver_sql(
                            "DELETE FROM events WHERE created_at < datetime('now', ?)",
                            (f"-{RETAIN_EVENTS_DAYS} days",),
                        )
                    else:
                        conn.exec_driver_sql(
                            f"DELETE FROM events WHERE created_at::timestamp < (CURRENT_TIMESTAMP - INTERVAL '{RETAIN_EVENTS_DAYS} days')"
                        )
            except Exception as e:
                try: LOGGER.warning("retention error: %s", e)
                except Exception: pass
            await asyncio.sleep(6*60*60)
    try:
        asyncio.create_task(_retention_loop())
        LOGGER.info("Started retention loop")
    except Exception as e:
        LOGGER.warning("Failed to start retention loop: %s", e)


@app.get("/health/obs/history")
def health_obs_history(limit_minutes: int = 60):
    """Return last N minutes of per-minute aggregates from req_stats."""
    if _dialect != "sqlite":
        return {"ok": False, "error": "req_stats history available only with sqlite"}
    try:
        with engine.connect() as conn:
            rows = conn.exec_driver_sql(
                "SELECT ts_min, path, count, avg_ms, p95_ms FROM req_stats ORDER BY ts_min DESC, path ASC LIMIT ?",
                (limit_minutes * 20,),  # rough cap, multiple paths per minute
            ).fetchall()
            out = [
                {"ts_min": r[0], "path": r[1], "count": r[2], "avg_ms": r[3], "p95_ms": r[4]}
                for r in rows
            ]
            return {"ok": True, "rows": out}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/config")
def get_config():
    """Public config consumed by the frontend (no secrets).

    Returning this from the backend avoids hard-coding owner phone/brand
    inside the frontend bundle.
    """
    return {"owner_phone": OWNER_PHONE, "brand_name": BRAND_NAME, "logo_url": LOGO_URL}

# ---- S3 presign (PUT) ----
# ---- AI describe (server calls OpenAI Vision) ----

# ---- Analytics ----
class EventIn(BaseModel):
    session_id: Optional[str] = None
    kind: str
    payload: Optional[dict] = None

@app.post("/events")
def create_event(e: EventIn):
    # Simple validation
    if not e.kind:
        raise HTTPException(400, "kind required")
    with session_scope() as db:
        ev = Event(session_id=e.session_id, kind=e.kind, payload=e.payload or {})
        db.add(ev)
        db.flush()
        try:
            LOGGER.info("event: kind=%s session=%s", e.kind, (e.session_id or ""))
        except Exception:
            pass
        return {"ok": True, "id": ev.id}

# ---- Admin event debug ----
@app.get("/admin/events/recent")
def admin_events_recent(request: Request, limit: int = 50):
    require_admin(request)
    with engine.connect() as conn:
        rows = conn.exec_driver_sql(
            "SELECT created_at, session_id, kind, payload FROM events ORDER BY created_at DESC LIMIT :limit",
            {"limit": max(1, int(limit))},
        ).fetchall()
    items = [
        {"created_at": r[0], "session_id": r[1], "kind": r[2], "payload": r[3]}
        for r in rows
    ]
    return {"ok": True, "items": items}

@app.get("/admin/events/counts")
def admin_events_counts(request: Request, hours: int = 24):
    require_admin(request)
    lookback = max(1, int(hours))
    cutoff = datetime.utcnow() - timedelta(hours=lookback)
    with engine.connect() as conn:
        if _dialect == "sqlite":
            rows = conn.exec_driver_sql(
                "SELECT kind, COUNT(*) FROM events WHERE created_at >= datetime('now', ?) GROUP BY kind",
                (f"-{lookback} hours",),
            ).fetchall()
        else:
            rows = conn.exec_driver_sql(
                "SELECT kind, COUNT(*) FROM events WHERE created_at::timestamp >= :cutoff GROUP BY kind",
                {"cutoff": cutoff},
            ).fetchall()
    return {"ok": True, "hours": hours, "counts": {str(k): int(c) for k,c in rows}}

@app.post("/ai/describe", response_model=DescribeResponse)
def ai_describe(req: DescribeRequest):
    """Generate product title/description/category from image(s) using AI.

    - Reads model and prompts from env (see AI_* variables above)
    - Accepts image URLs (including base64 data URLs) from the frontend
    - Returns a small JSON that the UI uses to prefill the upload form
    """
    # If AI_DEBUG is enabled, or OpenAI client is unavailable, return a safe dummy
    if AI_DEBUG:
        LOGGER.info("AI describe fallback: AI_DEBUG=1 (images=%d)", len(req.image_urls))
        cat = "Jewelry Sets" if len(req.image_urls) != 1 else "Earrings"
        style_guess = _normalize_style_tag("Kundan" if cat == "Jewelry Sets" else "Oxidized Silver")
        title = "Elegant kundan jewelry set" if cat == "Jewelry Sets" else "Minimalist oxidized earrings"
        description = (
            "Developer mode (AI_DEBUG) is ON: sample metadata. Silver-tone detailing; suitable for festive and everyday wear."
        )
        return DescribeResponse(title=title, description=description, category=cat, style_tag=style_guess)
    if not _openai_client:
        LOGGER.warning("AI describe fallback: OpenAI client unavailable (images=%d)", len(req.image_urls))
        cat = "Jewelry Sets" if len(req.image_urls) != 1 else "Earrings"
        style_guess = _normalize_style_tag("Kundan" if cat == "Jewelry Sets" else "Oxidized Silver")
        title = "Elegant kundan jewelry set" if cat == "Jewelry Sets" else "Minimalist oxidized earrings"
        description = (
            "Developer mode (AI client missing) is ON: sample metadata. Silver-tone detailing; suitable for festive and everyday wear."
        )
        return DescribeResponse(title=title, description=description, category=cat, style_tag=style_guess)
    if not req.image_urls:
        raise HTTPException(400, "image_urls required")
    try:
        # Chat Completions with system + multimodal user content
        messages = [
            {"role": "system", "content": AI_PROMPT_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": AI_PROMPT_USER},
                    *[{"type": "image_url", "image_url": {"url": u}} for u in req.image_urls],
                ],
            },
        ]
        # Ask the model (gpt-4o by default) to generate JSON only
        t0 = time.perf_counter()
        resp = _openai_client.chat.completions.create(
            model=AI_MODEL,
            messages=messages,
            max_tokens=600,
            temperature=0.2,
        )
        if OBS_LOG_TIMING:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            LOGGER.info(
                "AI describe OpenAI latency=%.1fms images=%d model=%s",
                latency_ms,
                len(req.image_urls),
                AI_MODEL,
            )
        content = resp.choices[0].message.content or "{}"
        if os.getenv("DEBUG_AI"):
            LOGGER.info("AI describe raw content: %s", content)
        # Best-effort JSON extraction in case the model wraps code fences
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                data = json.loads(content[start:end+1])
            else:
                raise
        if "description" not in data and "description_" in data:
            data["description"] = data.get("description_")
        if "style_tag" in data:
            data["style_tag"] = _normalize_style_tag(data.get("style_tag"))
        return DescribeResponse(**data)
    except json.JSONDecodeError:
        LOGGER.error("OpenAI returned invalid JSON: %s", content)
        raise HTTPException(502, "Invalid response from OpenAI")
    except Exception as e:
        if 't0' in locals() and OBS_LOG_TIMING:
            LOGGER.warning(
                "AI describe OpenAI failure latency=%.1fms images=%d model=%s",
                (time.perf_counter() - t0) * 1000.0,
                len(req.image_urls),
                AI_MODEL,
            )
        LOGGER.exception("OpenAI describe failed (images=%d)", len(req.image_urls))
        raise HTTPException(500, f"OpenAI error: {e}")

# ---- Products ----
# CRUD endpoints used by both Customer and Owner apps
@app.get("/products", response_model=ProductListOut, response_model_exclude={"items": {"__all__": {"cost", "processing_error"}}})
def list_products(
    request: Request,
    response: Response,
    category: Optional[str] = None,
    q: Optional[str] = None,
    min_price: Optional[int] = None,
    max_price: Optional[int] = None,
    only_available: bool = False,
    style_tag: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    img_first: bool = False,
): 
    with session_scope() as db:
        qry = db.query(Product)
        qry = qry.filter(Product.status == "ready")
        if category:
            qry = qry.filter(Product.category == category)
        style_filter = _normalize_style_tag(style_tag)
        if style_filter:
            qry = qry.filter(Product.style_tag == style_filter)
        if q:
            like = f"%{q}%"
            qry = qry.filter((Product.title.ilike(like)) | (Product.description.ilike(like)))
        if min_price is not None:
            qry = qry.filter(Product.price_in_paise >= int(min_price) * 100)
        if max_price is not None:
            qry = qry.filter(Product.price_in_paise <= int(max_price) * 100)
        if only_available:
            qry = qry.filter(Product.available == True, Product.qty > 0)
        total = qry.count()
        page = (
            qry.order_by(Product.created_at.desc())
            .offset(max(0, int(offset)))
            .limit(max(0, int(limit)))
            .all()
        )
        items: List[ProductOut] = []
        for p in page:
            out = _product_to_out(p, include_processing_error=False)
            if img_first and isinstance(out.images, list) and len(out.images) > 1:
                out.images = [out.images[0]]
            items.append(out)
        next_offset = (offset + limit) if (offset + limit) < total else None
        # ETag + simple cache header
        import hashlib, json as _json
        sig_src = _json.dumps({
            'params': {'category': category, 'q': q, 'min': min_price, 'max': max_price,
                       'only': only_available, 'style': style_filter, 'limit': limit, 'offset': offset},
            'items': [{'id': p.id, 'qty': p.qty, 'avail': p.available, 'price': p.price_in_paise} for p in page]
        }, sort_keys=True).encode('utf-8')
        etag = 'W/"' + hashlib.sha1(sig_src).hexdigest() + '"'
        inm = request.headers.get('If-None-Match')
        if inm == etag:
            response.headers['ETag'] = etag
            response.headers['X-Cache'] = 'HIT'
            return Response(status_code=304)
        response.headers['ETag'] = etag
        response.headers['Cache-Control'] = 'public, max-age=60'
        response.headers['X-Cache'] = 'MISS'
        return {"items": items, "total": total, "next_offset": next_offset}

@app.get("/products/{pid}/images")
def get_product_images(pid: str):
    """Return all images for a product (used for lazy loading in Shop).

    Keeps list payloads smaller by allowing Shop to fetch full image arrays on demand.
    """
    with session_scope() as db:
        m: Product = db.get(Product, pid)
        if not m:
            raise HTTPException(404, "Product not found")
        return {"images": m.images or []}

@app.get("/owner/products", response_model=ProductListOut)
def list_products_owner(
    request: Request,
    response: Response,
    category: Optional[str] = None,
    q: Optional[str] = None,
    min_price: Optional[int] = None,
    max_price: Optional[int] = None,
    only_available: bool = False,
    style_tag: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
):
    require_admin(request)
    # Same as public listing but includes cost in response
    with session_scope() as db:
        qry = db.query(Product)
        if category:
            qry = qry.filter(Product.category == category)
        style_filter = _normalize_style_tag(style_tag)
        if style_filter:
            qry = qry.filter(Product.style_tag == style_filter)
        if q:
            like = f"%{q}%"
            qry = qry.filter((Product.title.ilike(like)) | (Product.description.ilike(like)))
        if min_price is not None:
            qry = qry.filter(Product.price_in_paise >= int(min_price) * 100)
        if max_price is not None:
            qry = qry.filter(Product.price_in_paise <= int(max_price) * 100)
        if only_available:
            qry = qry.filter(Product.available == True, Product.qty > 0)
        total = qry.count()
        page = (
            qry.order_by(Product.created_at.desc())
            .offset(max(0, int(offset)))
            .limit(max(0, int(limit)))
            .all()
        )
        items = [_product_to_out(p) for p in page]
        next_offset = (offset + limit) if (offset + limit) < total else None
        import hashlib, json as _json
        sig_src = _json.dumps({
            'params': {'category': category, 'q': q, 'min': min_price, 'max': max_price,
                       'only': only_available, 'style': style_filter, 'limit': limit, 'offset': offset},
            'items': [{'id': p.id, 'qty': p.qty, 'avail': p.available, 'price': p.price_in_paise} for p in page]
        }, sort_keys=True).encode('utf-8')
        etag = 'W/"' + hashlib.sha1(sig_src).hexdigest() + '"'
        inm = request.headers.get('If-None-Match')
        if inm == etag:
            response.headers['ETag'] = etag
            response.headers['X-Cache'] = 'HIT'
            return Response(status_code=304)
        response.headers['ETag'] = etag
        response.headers['Cache-Control'] = 'private, max-age=60'
        response.headers['X-Cache'] = 'MISS'
        return {"items": items, "total": total, "next_offset": next_offset}


@app.get("/owner/products/{pid}", response_model=ProductOut)
def get_product_owner(request: Request, pid: str):
    require_admin(request)
    with session_scope() as db:
        prod: Product = db.get(Product, pid)
        if not prod:
            raise HTTPException(404, "Product not found")
        return _product_to_out(prod).model_dump()


@app.post("/owner/products/{pid}/reprocess", response_model=ProductOut, status_code=status.HTTP_202_ACCEPTED)
def reprocess_product(request: Request, pid: str, background_tasks: BackgroundTasks):
    require_admin(request)
    with session_scope() as db:
        prod: Product = db.get(Product, pid)
        if not prod:
            raise HTTPException(404, "Product not found")
        pending_raw = prod.pending_images or []
        if pending_raw:
            images_payload = _normalize_pending_images(pending_raw)
        else:
            current_full = list(prod.images or [])
            current_small = list(prod.images_small or [])
            images_payload = []
            for idx, full in enumerate(current_full):
                if not full:
                    continue
                small_val = current_small[idx] if idx < len(current_small) else None
                images_payload.append({"src": full, "small": small_val})
        if not images_payload:
            raise HTTPException(400, "No images available to reprocess")
        prod.pending_images = images_payload
        prod.status = "processing"
        prod.processing_error = None
        prod.images = [item["src"] for item in images_payload]
        db.flush()
        out = _product_to_out(prod)
    background_tasks.add_task(_process_product_images_async, pid, images_payload)
    return out.model_dump()

@app.post("/products", response_model=ProductOut, status_code=status.HTTP_202_ACCEPTED)
def create_product(request: Request, p: ProductIn, background_tasks: BackgroundTasks):
    require_admin(request)
    raw_images = [img for img in list(p.images or []) if img]
    pending_payload = [{"src": img, "small": None} for img in raw_images]
    with session_scope() as db:
        LOGGER.info(
            "Create product: id=%s, title=%s, cat=%s, style_tag=%s, images=%d",
            p.id,
            p.title,
            p.category,
            _normalize_style_tag(getattr(p, "style_tag", None)),
            len(raw_images),
        )
        if db.get(Product, p.id):
            raise HTTPException(409, "Product ID already exists")
        status_val = "processing" if pending_payload else "ready"
        placeholder_images = [item["src"] for item in pending_payload] if status_val == "processing" else []
        m = Product(
            id=p.id,
            title=p.title,
            description=p.description,
            category=p.category,
            style_tag=_normalize_style_tag(getattr(p, "style_tag", None)),
            price_in_paise=p.price * 100,
            cost_in_paise=(p.cost or 0) * 100,
            qty=p.qty,
            available=p.available and p.qty > 0,
            images=placeholder_images,
            images_small=placeholder_images,
            status=status_val,
            processing_error=None,
            pending_images=pending_payload,
        )
        db.add(m)
        db.flush()
        out = _product_to_out(m)
    if pending_payload:
        background_tasks.add_task(_process_product_images_async, p.id, pending_payload)
    return out.model_dump()

@app.patch("/products/{pid}", response_model=ProductOut)
def update_product(request: Request, pid: str, p: ProductIn, background_tasks: BackgroundTasks):
    require_admin(request)
    pending_payload = None
    with session_scope() as db:
        LOGGER.info(
            "Update product: id=%s, title=%s, cat=%s, style_tag=%s, images=%d",
            pid,
            p.title,
            p.category,
            _normalize_style_tag(getattr(p, "style_tag", None)),
            len(p.images or []),
        )
        m: Product = db.get(Product, pid)
        if not m:
            raise HTTPException(404, "Product not found")
        m.title = p.title
        m.description = p.description
        m.category = p.category
        if "style_tag" in p.model_fields_set:
            m.style_tag = _normalize_style_tag(p.style_tag)
        m.price_in_paise = p.price * 100
        m.cost_in_paise = (p.cost or 0) * 100
        m.qty = p.qty
        m.available = p.available and p.qty > 0
        incoming_images = [img for img in list(p.images or []) if img]
        current_images = list(m.images or [])
        current_small = list(m.images_small or [])
        existing_preview: dict[str, list[Optional[str]]] = {}
        for idx, img_url in enumerate(current_images):
            thumb = current_small[idx] if idx < len(current_small) else None
            existing_preview.setdefault(img_url, []).append(thumb)
        images_changed = incoming_images != current_images

        if images_changed:
            if incoming_images:
                preview_smalls: list[str] = []
                payload: list[dict[str, Optional[str]]] = []
                for val in incoming_images:
                    small_list = existing_preview.get(val)
                    small_val = small_list.pop(0) if small_list else None
                    payload.append({"src": val, "small": small_val})
                    preview_smalls.append(small_val or val)
                pending_payload = payload
                m.pending_images = payload
                m.status = "processing"
                m.processing_error = None
                m.images = [item["src"] for item in payload]
                m.images_small = preview_smalls
            else:
                # Explicitly cleared images
                m.pending_images = []
                m.images = []
                m.images_small = []
                m.status = "ready"
                m.processing_error = None
        db.flush()
        out = _product_to_out(m)

    if images_changed and pending_payload:
        background_tasks.add_task(_process_product_images_async, pid, pending_payload)
    return out.model_dump()

@app.delete("/products/{pid}")
def delete_product(request: Request, pid: str):
    require_admin(request)
    with session_scope() as db:
        m = db.get(Product, pid)
        if not m:
            raise HTTPException(404, "Product not found")
        db.delete(m)
        return {"ok": True}

# ---- Orders ----
# A minimal flow: customers create a shortlist (an order with items),
# owner confirms it which atomically decrements inventory.
@app.post("/orders", response_model=OrderOut)
def create_order(req: OrderCreate):
    if not req.items:
        raise HTTPException(400, "No items")
    with session_scope() as db:
        # Create order with fixed qty=1 per item
        oid = f"ORD-{uuid.uuid4().hex[:8]}"
        order = Order(id=oid, customer_phone=req.customer_phone, status="pending", session_id=req.session_id)
        db.add(order)
        for pid in req.items:
            # validate product exists
            prod = db.get(Product, pid)
            if not prod:
                raise HTTPException(400, f"Product not found: {pid}")
            db.add(OrderItem(order=order, product_id=pid, qty=1))
        db.flush()
        return OrderOut(id=order.id, status=order.status, customer_phone=order.customer_phone, items=[i.product_id for i in order.items])

class OrdersBySessionOut(BaseModel):
    items: List[OrderOut]

@app.get("/orders/by_session", response_model=OrdersBySessionOut)
def orders_by_session(session_id: str, limit: int = 1):
    if not session_id:
        raise HTTPException(400, "session_id required")
    with session_scope() as db:
        qry = db.query(Order).filter(Order.session_id == session_id)
        page = qry.order_by(Order.created_at.desc()).limit(max(1, int(limit))).all()
        out: List[OrderOut] = []
        for o in page:
            out.append(OrderOut(id=o.id, status=o.status, customer_phone=o.customer_phone, items=[i.product_id for i in o.items], created_at=o.created_at, confirmed_at=o.confirmed_at, removed_items=o.removed_items))
        return {"items": out}

@app.patch("/orders/{oid}/confirm", response_model=OrderOut)
def confirm_order(request: Request, oid: str):
    require_admin(request)
    with session_scope() as db:
        order: Order = db.get(Order, oid)
        if not order:
            raise HTTPException(404, "Order not found")
        if order.status == "confirmed":
            return OrderOut(id=order.id, status=order.status, customer_phone=order.customer_phone, items=[i.product_id for i in order.items])
        # Atomic inventory update
        for item in order.items:
            prod: Product = db.get(Product, item.product_id)
            if not prod or not prod.available or prod.qty <= 0:
                reason = "not found" if not prod else ("not available" if not prod.available else "qty <= 0")
                raise HTTPException(409, f"Item not confirmable: {item.product_id} ({reason})")
            prod.qty = max(0, prod.qty - 1)
            prod.available = prod.qty > 0 and prod.available
        order.status = "confirmed"
        order.confirmed_at = datetime.utcnow().isoformat()
        db.flush()
        return OrderOut(id=order.id, status=order.status, customer_phone=order.customer_phone, items=[i.product_id for i in order.items], created_at=order.created_at, confirmed_at=order.confirmed_at, removed_items=order.removed_items)

class OrdersListOut(BaseModel):
    items: List[OrderOut]
    total: Optional[int] = None
    next_offset: Optional[int] = None

# Optional: list orders for owner inbox (with pagination)
@app.get("/orders", response_model=OrdersListOut)
def list_orders(request: Request, status: Optional[str] = None, limit: int = 100, offset: int = 0):
    require_admin(request)
    with session_scope() as db:
        qry = db.query(Order)
        if status:
            qry = qry.filter(Order.status == status)
        total = qry.count()
        page = qry.order_by(Order.created_at.desc()).offset(max(0,int(offset))).limit(max(0,int(limit))).all()
        out: List[OrderOut] = []
        for o in page:
            out.append(OrderOut(id=o.id, status=o.status, customer_phone=o.customer_phone, items=[i.product_id for i in o.items], created_at=o.created_at, confirmed_at=o.confirmed_at, removed_items=o.removed_items))
        next_offset = (offset + limit) if (offset + limit) < total else None
        return {"items": out, "total": total, "next_offset": next_offset}

@app.patch("/orders/{oid}/remove_item", response_model=OrderOut)
def remove_order_item(request: Request, oid: str, pid: str = Body(..., embed=True)):
    require_admin(request)
    with session_scope() as db:
        order: Order = db.get(Order, oid)
        if not order:
            raise HTTPException(404, "Order not found")
        # remove item association
        order.items = [i for i in order.items if i.product_id != pid]
        # track removed list
        removed = set(order.removed_items or [])
        removed.add(pid)
        order.removed_items = list(removed)
        db.flush()
        return OrderOut(id=order.id, status=order.status, customer_phone=order.customer_phone, items=[i.product_id for i in order.items], created_at=order.created_at, confirmed_at=order.confirmed_at, removed_items=order.removed_items)

@app.patch("/orders/{oid}/add_item", response_model=OrderOut)
def add_order_item(request: Request, oid: str, pid: str = Body(..., embed=True)):
    require_admin(request)
    with session_scope() as db:
        order: Order = db.get(Order, oid)
        if not order:
            raise HTTPException(404, "Order not found")
        # validate product exists
        prod: Product = db.get(Product, pid)
        if not prod:
            raise HTTPException(400, f"Product not found: {pid}")
        # add back only if not already present
        existing_ids = {i.product_id for i in order.items}
        if pid not in existing_ids:
            db.add(OrderItem(order=order, product_id=pid, qty=1))
        # remove from removed list if present
        removed = set(order.removed_items or [])
        if pid in removed:
            removed.remove(pid)
            order.removed_items = list(removed)
        db.flush()
        return OrderOut(id=order.id, status=order.status, customer_phone=order.customer_phone, items=[i.product_id for i in order.items], created_at=order.created_at, confirmed_at=order.confirmed_at, removed_items=order.removed_items)

# ----- Notes -----
# • For Lambda deploy, add:
#     pip install mangum
#     from mangum import Mangum
#     handler = Mangum(app)
# • For migrations, integrate Alembic later. This starter uses create_all for speed.
# • Replace sqlite fallback with real RDS in env before prod use.
