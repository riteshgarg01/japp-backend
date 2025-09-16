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
    qty = Column(Integer, nullable=False, default=0)
    available = Column(Boolean, nullable=False, default=True)
    images = Column(JSON, nullable=False, default=list)  # [url, ...]
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
    qty: int = Field(..., ge=0)
    available: bool = True
    images: List[str] = []

class ProductOut(BaseModel):
    id: str; title: str; description: str; category: str
    price: int; qty: int; available: bool; images: List[str]

class ProductListOut(BaseModel):
    items: List[ProductOut]
    total: Optional[int] = None
    next_offset: Optional[int] = None

class OrderCreate(BaseModel):
    items: List[str]  # product IDs; fixed quantity = 1 per item
    customer_phone: str

class OrderOut(BaseModel):
    id: str; status: str; customer_phone: str; items: List[str]

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
@app.get("/products", response_model=ProductListOut)
def list_products(
    category: Optional[str] = None,
    q: Optional[str] = None,
    min_price: Optional[int] = None,
    max_price: Optional[int] = None,
    only_available: bool = False,
    limit: int = 100,
    offset: int = 0,
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
        items = [
            ProductOut(
                id=p.id,
                title=p.title,
                description=p.description,
                category=p.category,
                price=p.price_in_paise // 100,
                qty=p.qty,
                available=p.available,
                images=p.images or [],
            )
            for p in page
        ]
        next_offset = (offset + limit) if (offset + limit) < total else None
        return {"items": items, "total": total, "next_offset": next_offset}

@app.post("/products", response_model=ProductOut)
def create_product(p: ProductIn):
    with session_scope() as db:
        LOGGER.info("Create product: id=%s, title=%s, cat=%s, images=%d", p.id, p.title, p.category, len(p.images or []))
        if db.get(Product, p.id):
            raise HTTPException(409, "Product ID already exists")
        m = Product(
            id=p.id,
            title=p.title,
            description=p.description,
            category=p.category,
            price_in_paise=p.price * 100,
            qty=p.qty,
            available=p.available and p.qty > 0,
            images=p.images,
        )
        db.add(m)
        db.flush()
        return ProductOut(id=m.id, title=m.title, description=m.description, category=m.category, price=m.price_in_paise // 100, qty=m.qty, available=m.available, images=m.images or [])

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
        m.qty = p.qty
        m.available = p.available and p.qty > 0
        m.images = p.images
        db.flush()
        return ProductOut(id=m.id, title=m.title, description=m.description, category=m.category, price=m.price_in_paise // 100, qty=m.qty, available=m.available, images=m.images or [])

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
        order = Order(id=oid, customer_phone=req.customer_phone, status="pending")
        db.add(order)
        for pid in req.items:
            # validate product exists
            prod = db.get(Product, pid)
            if not prod:
                raise HTTPException(400, f"Product not found: {pid}")
            db.add(OrderItem(order=order, product_id=pid, qty=1))
        db.flush()
        return OrderOut(id=order.id, status=order.status, customer_phone=order.customer_phone, items=[i.product_id for i in order.items])

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
        db.flush()
        return OrderOut(id=order.id, status=order.status, customer_phone=order.customer_phone, items=[i.product_id for i in order.items])

# Optional: list orders for owner inbox
@app.get("/orders", response_model=List[OrderOut])
def list_orders(status: Optional[str] = None):
    with session_scope() as db:
        qry = db.query(Order)
        if status:
            qry = qry.filter(Order.status == status)
        out: List[OrderOut] = []
        for o in qry.order_by(Order.created_at.desc()).all():
            out.append(OrderOut(id=o.id, status=o.status, customer_phone=o.customer_phone, items=[i.product_id for i in o.items]))
        return out

# ----- Notes -----
# • For Lambda deploy, add:
#     pip install mangum
#     from mangum import Mangum
#     handler = Mangum(app)
# • For migrations, integrate Alembic later. This starter uses create_all for speed.
# • Replace sqlite fallback with real RDS in env before prod use.
