## Local dev
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt   # if you have one
# or: pip install fastapi "uvicorn[standard]" pydantic "sqlalchemy" python-dotenv openai
uvicorn app:app --reload --host 127.0.0.1 --port 8000

### Env
Copy `.env.example` to `.env` and set values. Frontend reads owner phone via `GET /config`:

- `OWNER_PHONE=+919999999999`
- `BRAND_NAME=Arohi's collection` (optional)
