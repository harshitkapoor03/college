import os
import random
import sqlite3
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp.server.auth.provider import AccessToken
import httpx
import html

# --- Load environment variables (trim whitespace) ---
load_dotenv()
TOKEN = os.getenv("AUTH_TOKEN", "") or ""
TOKEN = TOKEN.strip()
MY_NUMBER = os.getenv("MY_NUMBER", "") or ""
MY_NUMBER = MY_NUMBER.strip()

assert TOKEN, "Please set AUTH_TOKEN in .env or Railway variables"
assert MY_NUMBER, "Please set MY_NUMBER in .env or Railway variables"

# --- Auth Provider ---
class SimpleBearerAuth(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        # the BearerAuthProvider is deprecated but still works; replace with JWTVerifier if desired
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        # token is expected to be the raw bearer token (no "Bearer " prefix)
        if token == self.token:
            return AccessToken(token=token, client_id=MY_NUMBER, scopes=["*"], expires_at=None)
        return None

# --- MCP Server (ASGI app) ---
mcp = FastMCP("College Quiz MCP Server", auth=SimpleBearerAuth(TOKEN))

# --- Create FastAPI app so we can add HTTP endpoints for health, register, discovery ---
app = FastAPI()

# Mount the MCP ASGI app at /mcp (FastMCP itself is an ASGI app)
app.mount("/mcp", mcp)

# --- SQLite leaderboard ---
DB_PATH = os.getenv("LEADERBOARD_DB", "leaderboard.db")
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cur = conn.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS colleges(college TEXT PRIMARY KEY, total_score INTEGER)")
for c in ("BITS", "P", "G/H"):
    cur.execute("INSERT OR IGNORE INTO colleges VALUES(?,0)", (c,))
conn.commit()

# --- In-memory quiz state ---
active_quizzes: dict[str, dict] = {}

# --- MCP tools (still registered on the MCP app object) ---
@mcp.tool
async def validate() -> str:
    return MY_NUMBER

async def fetch_questions():
    questions = []
    async with httpx.AsyncClient() as client:
        for diff, count in (("easy", 1), ("medium", 1), ("hard", 3)):
            resp = await client.get(
                "https://opentdb.com/api.php",
                params={"amount": count, "difficulty": diff, "type": "multiple"},
                timeout=10
            )
            resp.raise_for_status()
            data = resp.json().get("results", [])
            for item in data:
                # decode HTML entities so questions look clean
                q = html.unescape(item["question"])
                choices = [html.unescape(c) for c in item["incorrect_answers"] + [item["correct_answer"]]]
                random.shuffle(choices)
                questions.append({
                    "diff": diff.capitalize(),
                    "q": q,
                    "choices": choices,
                    "ans": html.unescape(item["correct_answer"])
                })
    return questions

@mcp.tool
async def enter_competition(college: str | None = None, answer: int | None = None) -> str:
    # get the client id (phone) from the last MCP message's access token
    phone = mcp.last_message().access_token.client_id

    # Main menu
    if phone not in active_quizzes and college is None and answer is None:
        return (
            "🏁 Main Menu:\n"
            "• Start Competition → @enter_competition college=<A|B|C>\n"
            "• View Leaderboard → @show_leaderboard"
        )

    # College selection
    if college and phone not in active_quizzes:
        mapping = {"A": "BITS", "B": "P", "C": "G/H"}
        col = mapping.get(college.upper())
        if not col:
            return "Invalid choice. Use A, B, or C."
        qs = await fetch_questions()
        active_quizzes[phone] = {"college": col, "questions": qs, "current": 0}
        qd = qs[0]
        opts = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(qd["choices"]))
        return (
            f"Quiz for {col}:\n"
            f"Q1 ({qd['diff']}): {qd['q']}\n"
            f"{opts}\n"
            "Reply with @enter_competition answer=<number>"
        )

    # Answer handling
    if answer is not None and phone in active_quizzes:
        session = active_quizzes[phone]
        idx = session["current"]
        qd = session["questions"][idx]
        # answer is 1-based index from user
        try:
            selected = qd["choices"][answer - 1]
        except Exception:
            return "Invalid answer number. Pick a valid option."

        correct = (selected == qd["ans"])
        feedback = "✅ Correct! +10 pts." if correct else f"❌ Wrong. Answer was: {qd['ans']}"
        if correct:
            cur.execute(
                "UPDATE colleges SET total_score = total_score + 10 WHERE college = ?",
                (session["college"],)
            )
            conn.commit()
        session["current"] += 1
        idx = session["current"]

        # Quiz complete
        if idx >= len(session["questions"]):
            col = session["college"]
            del active_quizzes[phone]
            return (
                f"{feedback}\n\n"
                f"🎉 Quiz complete for {col}!\n\n"
                "🏁 Main Menu:\n"
                "• Start Competition → @enter_competition college=<A|B|C>\n"
                "• View Leaderboard → @show_leaderboard"
            )

        # Next question
        qd = session["questions"][idx]
        opts = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(qd["choices"]))
        return (
            f"{feedback}\n"
            f"Q{idx+1} ({qd['diff']}): {qd['q']}\n"
            f"{opts}\n"
            "Reply with @enter_competition answer=<number>"
        )

    # Fallback
    return "Use @enter_competition to start or @show_leaderboard to view rankings."

@mcp.tool
async def show_leaderboard() -> str:
    rows = cur.execute(
        "SELECT college, total_score FROM colleges ORDER BY total_score DESC"
    ).fetchall()
    lines = [f"{c}: {s}" for c, s in rows]
    return "🏆 Leaderboard 🏆\n" + "\n".join(lines)

# --- HTTP endpoints to satisfy clients / Railway health checks ---


@app.get("/")
async def root():
    return {"status": "ok", "message": "College Quiz MCP running"}


@app.post("/register")
async def register(request: Request):
    """
    Minimal registration endpoint: returns the server TOKEN so clients expecting /register succeed.
    This is intentionally simple: the client registers and receives the bearer token.
    """
    # Optionally, you could validate incoming registration body here.
    return JSONResponse({"access_token": TOKEN, "token_type": "Bearer", "expires_in": 3600})


@app.get("/.well-known/oauth-authorization-server")
async def well_known_oauth(request: Request):
    host = request.url.scheme + "://" + request.url.hostname
    # Use actual host:port if needed; clients typically only require these keys exist
    return JSONResponse({
        "issuer": host,
        "token_endpoint": str(request.base_url) + "token",
        "authorization_endpoint": str(request.base_url) + "auth",
        "registration_endpoint": str(request.base_url) + "register"
    })


@app.get("/.well-known/openid-configuration")
async def openid_config(request: Request):
    host = request.url.scheme + "://" + request.url.hostname
    return JSONResponse({
        "issuer": host,
        "jwks_uri": str(request.base_url) + "jwks",
        "token_endpoint": str(request.base_url) + "token",
        "authorization_endpoint": str(request.base_url) + "auth",
        "registration_endpoint": str(request.base_url) + "register"
    })


# Also support paths under /mcp/... that some clients request for discovery.
# If client requests /.well-known/openid-configuration/mcp/<id> it will likely 404;
# we add a permissive route here to help discovery under /mcp too.
@app.get("/mcp/.well-known/openid-configuration")
async def openid_config_mcp(request: Request):
    return await openid_config(request)

@app.get("/mcp/.well-known/oauth-authorization-server")
async def well_known_oauth_mcp(request: Request):
    return await well_known_oauth(request)


# --- Run server (ASGI: mounts + FastAPI) ---
if __name__ == "__main__":
    import uvicorn
    # Run uvicorn app — it serves both regular HTTP endpoints and the mounted /mcp ASGI app.
    uvicorn.run(app, host="0.0.0.0", port=8086)
