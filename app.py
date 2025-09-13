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
import os, json, uuid
from typing import List, Optional
from contextlib import contextmanager

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, String, Integer, Boolean, JSON, text, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session

# ----- OpenAI (server-side only; never expose keys to frontend) -----
try:
    from openai import OpenAI
    _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception:
    _openai_client = None

# ----- Config -----
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+pysqlite:///./arohi.db")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]

engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()


# ----- DB Models -----
class Product(Base):
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
    __tablename__ = "orders"
    id = Column(String, primary_key=True)  # e.g., ORD-xxx
    customer_phone = Column(String, nullable=False)
    status = Column(String, nullable=False, default="pending")  # pending | confirmed | cancelled
    created_at = Column(String, server_default=text("CURRENT_TIMESTAMP"))
    items = relationship("OrderItem", back_populates="order", cascade="all, delete-orphan")

class OrderItem(Base):
    __tablename__ = "order_items"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    order_id = Column(String, ForeignKey("orders.id"), nullable=False)
    product_id = Column(String, ForeignKey("products.id"), nullable=False)
    qty = Column(Integer, nullable=False, default=1)
    order = relationship("Order", back_populates="items")
    product = relationship("Product")

Base.metadata.create_all(engine)

@contextmanager
def session_scope():
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

class OrderCreate(BaseModel):
    items: List[str]  # product IDs; fixed quantity = 1 per item
    customer_phone: str

class OrderOut(BaseModel):
    id: str; status: str; customer_phone: str; items: List[str]

class DescribeRequest(BaseModel):
    image_urls: List[str] = Field(default_factory=list)

class DescribeResponse(BaseModel):
    title: str
    description: str
    category: str

# ----- FastAPI -----
app = FastAPI(title="Arohi Backend", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOWED_ORIGINS == ["*"] else ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

# ---- S3 presign (PUT) ----
# ---- AI describe (server calls OpenAI Vision) ----
PROMPT = (
    "You are a content writer for Shopify store that sells artificial jewelry in India. "
    "Analyze the product image(s) and provide strictly JSON with keys: title, description, category. "
    "Rules: Avoid any reference to imitation, gold, silver, or gemstones. "
    "Audience: Indian women 18-50. "
    "Category must be one of: 'Apparel & Accessories > Jewelry > Anklets', 'Apparel & Accessories > Jewelry > Bracelets', "
    "'Apparel & Accessories > Jewelry > Brooches & Lapel Pins', 'Apparel & Accessories > Jewelry > Charms & Pendants', "
    "'Apparel & Accessories > Jewelry > Earrings', 'Apparel & Accessories > Jewelry > Jewelry Sets', "
    "'Apparel & Accessories > Jewelry > Necklaces', 'Apparel & Accessories > Jewelry > Rings'."
)

@app.post("/ai/describe", response_model=DescribeResponse)
def ai_describe(req: DescribeRequest):
    if not _openai_client:
        # Dev fallback: return neutral placeholders
        return DescribeResponse(title="Elegant festive piece", description="Lightweight, versatile piece for everyday and festive looks.", category="Apparel & Accessories > Jewelry > Earrings")
    if not req.image_urls:
        raise HTTPException(400, "image_urls required")
    try:
        # Use a vision-capable model via Chat Completions
        messages = [
            {"role": "system", "content": PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Generate JSON with keys: title, description, category."},
                    *[{"type": "image_url", "image_url": {"url": u}} for u in req.image_urls],
                ],
            },
        ]
        # Model name can be adjusted; choose a vision-capable model
        resp = _openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2,
        )
        content = resp.choices[0].message.content or "{}"
        data = json.loads(content)
        return DescribeResponse(**data)
    except Exception as e:
        raise HTTPException(500, f"OpenAI error: {e}")

# ---- Products ----
@app.get("/products", response_model=ProductListOut)
def list_products(category: Optional[str] = None, q: Optional[str] = None, min_price: Optional[int] = None, max_price: Optional[int] = None, only_available: bool = False):
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
            for p in qry.order_by(Product.created_at.desc()).all()
        ]
        return {"items": items}

@app.post("/products", response_model=ProductOut)
def create_product(p: ProductIn):
    with session_scope() as db:
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
