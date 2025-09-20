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
"""
from __future__ import annotations
import os, json, uuid, logging
from pathlib import Path
from typing import List, Optional
from contextlib import contextmanager
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Use Uvicorn's logger so messages show in the terminal
# (print-style logging often doesn't appear when running behind Uvicorn/Gunicorn)
LOGGER = logging.getLogger("uvicorn.error")

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, String, Integer, Boolean, JSON, text, ForeignKey
from datetime import datetime
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session

# ----- OpenAI (server-side only; never expose keys to frontend) -----
# We import the SDK and try to create a client from the OPENAI_API_KEY
# in your .env. If anything fails (no key, not installed, etc.), we
# set the client to None and the API will fall back to a safe default
# so local development keeps working.
try:
    from openai import OpenAI
    _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception:
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
# AI config
# Model and prompts used by the /ai/describe endpoint. You can tweak these
# in .env without changing code.
AI_MODEL = os.getenv("AI_MODEL", "gpt-4o")
AI_PROMPT_SYSTEM = os.getenv(
    "AI_PROMPT_SYSTEM",
    "You are a Product listing expert for a Shopify store that sells artificial jewelry in India.",
)
AI_PROMPT_USER = os.getenv(
    "AI_PROMPT_USER",
    (
        "Please analyze these image(s) of artificial jewelry to be listed on the Shopify store.\n"
        "Output must be STRICT JSON with keys: title, description, category.\n"
        "A good listing uses a title and description that are relevant to the jewelry material and type, tailored for an Indian audience.\n"
        "Avoid any words implying precious metals or real gemstones (gold, silver, diamond, etc.). Neutral descriptors allowed: 'silver-tone', 'kundan', 'polki', 'oxidized', 'meenakari', 'pearl', 'american diamond (AD)', 'moissanite'.\n"
        "Category must be one of: Anklets, Bracelets, Brooches & Lapel Pins, Charms & Pendants, Earrings, Jewelry Sets, Necklaces, Rings.\n"
        "If a necklace is shown with matching earrings, choose 'Jewelry Sets'. If only a necklace is present, choose 'Necklaces'. If only earrings, choose 'Earrings'."
    ),
)
AI_DEBUG = os.getenv("AI_DEBUG", "0").lower() in {"1", "true", "yes"}

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
    price_in_paise = Column(Integer, nullable=False)
    cost_in_paise = Column(Integer, nullable=False, default=0)
    qty = Column(Integer, nullable=False, default=0)
    available = Column(Boolean, nullable=False, default=True)
    images = Column(JSON, nullable=False, default=list)  # [url, ...]
    images_small = Column(JSON, nullable=True, default=list)
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

# Create tables automatically on startup for local dev (quick and simple).
# For production, prefer Alembic migrations instead.
Base.metadata.create_all(engine)

# Lightweight schema patch for SQLite: add columns if missing, ensure helpful indexes
try:
    with engine.connect() as conn:
        # Products: add cost_in_paise if missing
        cols_products = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(products)").fetchall()}
        if "cost_in_paise" not in cols_products:
            conn.exec_driver_sql("ALTER TABLE products ADD COLUMN cost_in_paise INTEGER DEFAULT 0")
        if "images_small" not in cols_products:
            conn.exec_driver_sql("ALTER TABLE products ADD COLUMN images_small JSON")

        # Orders: add confirmed_at and removed_items if missing
        cols_orders = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(orders)").fetchall()}
        if "confirmed_at" not in cols_orders:
            conn.exec_driver_sql("ALTER TABLE orders ADD COLUMN confirmed_at TEXT")
        if "removed_items" not in cols_orders:
            # JSON resolves to TEXT on SQLite; default to empty array
            conn.exec_driver_sql("ALTER TABLE orders ADD COLUMN removed_items JSON DEFAULT '[]'")
        if "session_id" not in cols_orders:
            conn.exec_driver_sql("ALTER TABLE orders ADD COLUMN session_id TEXT")

        # Helpful indexes for common filters/sorts (SQLite IF NOT EXISTS support)
        try:
            conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_products_created_at ON products(created_at)")
        except Exception:
            pass
        try:
            conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_products_category ON products(category)")
        except Exception:
            pass
        try:
            conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_products_available_qty ON products(available, qty)")
        except Exception:
            pass
        try:
            conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_orders_session_created ON orders(session_id, created_at)")
        except Exception:
            pass
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

def admin_backfill_thumbnails(token: Optional[str] = None, limit: Optional[int] = None, dry_run: bool = False):
    """Generate images_small for products that are missing them.

    Security: guarded by ADMIN_TOKEN env var. Provide via ?token=... or header X-Admin-Token.
    Only intended for local/UAT. For production, move to a proper admin auth.
    """
    admin_token = os.getenv("ADMIN_TOKEN", "")
    # allow header fallback
    from fastapi import Request
    # FastAPI dependency-less header read: we'll access from threadlocal scope using request object via contextvar isn't trivial.
    # Simpler: rely on query param token; if missing and env not set, allow (local dev).
    if admin_token and token != admin_token:
        raise HTTPException(401, "Unauthorized")
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
    s3.put_object(Bucket=AWS_S3_BUCKET, Key=key, Body=data, **extra)
    if AWS_S3_CDN_BASE_URL:
        return f"{AWS_S3_CDN_BASE_URL.rstrip('/')}/{key}"
    if AWS_S3_REGION:
        return f"https://{AWS_S3_BUCKET}.s3.{AWS_S3_REGION}.amazonaws.com/{key}"
    # regionless fallback
    return f"https://{AWS_S3_BUCKET}.s3.amazonaws.com/{key}"

def s3_upload_images(product_id: str, images: list[str]) -> tuple[list[str], list[str]]:
    import uuid
    full_urls, thumb_urls = [], []
    for val in (images or []):
        name = uuid.uuid4().hex
        key_full = f"products/{product_id}/{name}.jpg"
        key_thumb = f"products/{product_id}/{name}.thumb.jpg"
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
        # re-encode originals to jpeg at 85 to cap size
        full_jpeg = _jpeg_resize_bytes(data, max_size=(2000,2000), quality=85)
        thumb_jpeg = _jpeg_resize_bytes(data, max_size=(640,640), quality=75)
        full_url = s3_upload_bytes(key_full, full_jpeg, "image/jpeg")
        thumb_url = s3_upload_bytes(key_thumb, thumb_jpeg, "image/jpeg")
        full_urls.append(full_url)
        thumb_urls.append(thumb_url)
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
    price: int = Field(..., ge=0, description="Price in INR whole rupees")
    cost: int = Field(0, ge=0, description="Optional cost in INR whole rupees")
    qty: int = Field(..., ge=0)
    available: bool = True
    images: List[str] = []

class ProductOut(BaseModel):
    id: str; title: str; description: str; category: str
    price: int; cost: int; qty: int; available: bool; images: List[str]

class ProductListOut(BaseModel):
    items: List[ProductOut]
    total: Optional[int] = None
    next_offset: Optional[int] = None

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

@app.get("/health")
def health():
    return {"ok": True, "db": DATABASE_URL}

@app.get("/config")
def get_config():
    """Public config consumed by the frontend (no secrets).

    Returning this from the backend avoids hard-coding owner phone/brand
    inside the frontend bundle.
    """
    return {"owner_phone": OWNER_PHONE, "brand_name": BRAND_NAME}

# ---- S3 presign (PUT) ----
# ---- AI describe (server calls OpenAI Vision) ----

@app.post("/ai/describe", response_model=DescribeResponse)
def ai_describe(req: DescribeRequest):
    """Generate product title/description/category from image(s) using AI.

    - Reads model and prompts from env (see AI_* variables above)
    - Accepts image URLs (including base64 data URLs) from the frontend
    - Returns a small JSON that the UI uses to prefill the upload form
    """
    # If AI_DEBUG is enabled, or OpenAI client is unavailable, return a safe dummy
    if AI_DEBUG or not _openai_client:
        cat = "Jewelry Sets" if len(req.image_urls) != 1 else "Earrings"
        title = "Elegant kundan jewelry set" if cat == "Jewelry Sets" else "Minimalist oxidized earrings"
        description = (
            "Developer mode (AI_DEBUG) is ON: sample metadata. Silver-tone detailing; suitable for festive and everyday wear."
        )
        return DescribeResponse(title=title, description=description, category=cat)
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
        resp = _openai_client.chat.completions.create(
            model=AI_MODEL,
            messages=messages,
            max_tokens=600,
            temperature=0.2,
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
        return DescribeResponse(**data)
    except json.JSONDecodeError:
        LOGGER.error("OpenAI returned invalid JSON: %s", content)
        raise HTTPException(502, "Invalid response from OpenAI")
    except Exception as e:
        raise HTTPException(500, f"OpenAI error: {e}")

# ---- Products ----
# CRUD endpoints used by both Customer and Owner apps
@app.get("/products", response_model=ProductListOut, response_model_exclude={"items": {"__all__": {"cost"}}})
def list_products(
    category: Optional[str] = None,
    q: Optional[str] = None,
    min_price: Optional[int] = None,
    max_price: Optional[int] = None,
    only_available: bool = False,
    limit: int = 100,
    offset: int = 0,
    img_first: bool = False,
):
    with session_scope() as db:
        qry = db.query(Product)
        if category:
            qry = qry.filter(Product.category == category)
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
        items = []
        for p in page:
            imgs = (p.images_small or []) or (p.images or [])
            if img_first and isinstance(imgs, list) and len(imgs) > 1:
                imgs = [imgs[0]]
            items.append(
                ProductOut(
                    id=p.id,
                    title=p.title,
                    description=p.description,
                    category=p.category,
                    price=p.price_in_paise // 100,
                    cost=(getattr(p, 'cost_in_paise', 0) or 0) // 100,
                    qty=p.qty,
                    available=p.available,
                    images=imgs,
                )
            )
        next_offset = (offset + limit) if (offset + limit) < total else None
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
    category: Optional[str] = None,
    q: Optional[str] = None,
    min_price: Optional[int] = None,
    max_price: Optional[int] = None,
    only_available: bool = False,
    limit: int = 100,
    offset: int = 0,
):
    # Same as public listing but includes cost in response
    with session_scope() as db:
        qry = db.query(Product)
        if category:
            qry = qry.filter(Product.category == category)
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
        items = []
        for p in page:
            imgs = (p.images_small or []) or (p.images or [])
            items.append(ProductOut(
                id=p.id,
                title=p.title,
                description=p.description,
                category=p.category,
                price=p.price_in_paise // 100,
                cost=(getattr(p, 'cost_in_paise', 0) or 0) // 100,
                qty=p.qty,
                available=p.available,
                images=imgs,
            ))
        next_offset = (offset + limit) if (offset + limit) < total else None
        return {"items": items, "total": total, "next_offset": next_offset}

@app.post("/products", response_model=ProductOut)
def create_product(p: ProductIn):
    with session_scope() as db:
        LOGGER.info("Create product: id=%s, title=%s, cat=%s, images=%d", p.id, p.title, p.category, len(p.images or []))
        if db.get(Product, p.id):
            raise HTTPException(409, "Product ID already exists")
        imgs_full, thumbs = maybe_upload_to_cdn(p.images, p.id)
        m = Product(
            id=p.id,
            title=p.title,
            description=p.description,
            category=p.category,
            price_in_paise=p.price * 100,
            cost_in_paise=(p.cost or 0) * 100,
            qty=p.qty,
            available=p.available and p.qty > 0,
            images=imgs_full,
            images_small=thumbs,
        )
        db.add(m)
        db.flush()
        return ProductOut(id=m.id, title=m.title, description=m.description, category=m.category, price=m.price_in_paise // 100, cost=(m.cost_in_paise or 0)//100, qty=m.qty, available=m.available, images=m.images or [])

@app.patch("/products/{pid}", response_model=ProductOut)
def update_product(pid: str, p: ProductIn):
    with session_scope() as db:
        LOGGER.info("Update product: id=%s, title=%s, cat=%s, images=%d", pid, p.title, p.category, len(p.images or []))
        m: Product = db.get(Product, pid)
        if not m:
            raise HTTPException(404, "Product not found")
        m.title = p.title
        m.description = p.description
        m.category = p.category
        m.price_in_paise = p.price * 100
        m.cost_in_paise = (p.cost or 0) * 100
        m.qty = p.qty
        m.available = p.available and p.qty > 0
        imgs_full, thumbs = maybe_upload_to_cdn(p.images, p.id)
        m.images = imgs_full
        m.images_small = thumbs
        db.flush()
        return ProductOut(id=m.id, title=m.title, description=m.description, category=m.category, price=m.price_in_paise // 100, cost=(m.cost_in_paise or 0)//100, qty=m.qty, available=m.available, images=m.images or [])

@app.delete("/products/{pid}")
def delete_product(pid: str):
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
def confirm_order(oid: str):
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
                raise HTTPException(409, f"Item not available: {item.product_id}")
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
def list_orders(status: Optional[str] = None, limit: int = 100, offset: int = 0):
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
def remove_order_item(oid: str, pid: str = Body(..., embed=True)):
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
def add_order_item(oid: str, pid: str = Body(..., embed=True)):
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
