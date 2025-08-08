import os, random, sqlite3
from dotenv import load_dotenv
from fastapi import FastAPI
from fastmcp import FastMCP
from fastmcp.server.auth.providers.jwt import JWTVerifier, RSAKeyPair
from mcp.server.auth.provider import AccessToken
import httpx
import uvicorn

# 1. Load credentials
load_dotenv()
TOKEN = os.getenv("AUTH_TOKEN")
MY_NUMBER = os.getenv("MY_NUMBER")
assert TOKEN and MY_NUMBER, "Set AUTH_TOKEN and MY_NUMBER in .env"

# 2. Authentication Provider
class SimpleBearerAuth(JWTVerifier):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token
    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(token=token, client_id=MY_NUMBER, scopes=["*"], expires_at=None)
        return None

# 3. Main FastAPI app for health + MCP
app = FastAPI()

@app.get("/")
def home():
    return {"status": "ok", "message": "College Quiz MCP is running"}

mcp = FastMCP("College Quiz MCP", auth=SimpleBearerAuth(TOKEN))
app.mount("/mcp", mcp.app)

# 4. SQLite leaderboard
conn = sqlite3.connect("leaderboard.db", check_same_thread=False)
cur = conn.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS colleges(college TEXT PRIMARY KEY, total_score INTEGER)")
for c in ("BITS", "P", "G/H"):
    cur.execute("INSERT OR IGNORE INTO colleges VALUES(?,0)", (c,))
conn.commit()

active_quizzes: dict[str, dict] = {}

@mcp.tool
async def validate() -> str:
    return MY_NUMBER

async def fetch_questions():
    import html
    questions = []
    async with httpx.AsyncClient() as client:
        for diff, count in (("easy", 1), ("medium", 1), ("hard", 3)):
            resp = await client.get(
                "https://opentdb.com/api.php",
                params={"amount": count, "difficulty": diff, "type": "multiple"},
                timeout=10
            )
            data = resp.json().get("results", [])
            for item in data:
                q = html.unescape(item["question"])
                choices = [html.unescape(c) for c in item["incorrect_answers"] + [item["correct_answer"]]]
                random.shuffle(choices)
                questions.append({"diff": diff.capitalize(), "q": q, "choices": choices, "ans": item["correct_answer"]})
    return questions

@mcp.tool
async def enter_competition(college: str | None = None, answer: int | None = None) -> str:
    phone = mcp.last_message().access_token.client_id
    if phone not in active_quizzes and college is None and answer is None:
        return "🏁 Main Menu:\n• Start Competition → @enter_competition college=<A|B|C>\n• View Leaderboard → @show_leaderboard"
    if college and phone not in active_quizzes:
        mapping = {"A": "BITS", "B": "P", "C": "G/H"}
        col = mapping.get(college.upper())
        if not col:
            return "Invalid choice. Use A, B, or C."
        qs = await fetch_questions()
        active_quizzes[phone] = {"college": col, "questions": qs, "current": 0}
        qd = qs[0]
        opts = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(qd["choices"]))
        return f"Quiz for {col}:\nQ1 ({qd['diff']}): {qd['q']}\n{opts}\nReply with @enter_competition answer=<number>"
    if answer is not None and phone in active_quizzes:
        session = active_quizzes[phone]
        idx = session["current"]
        qd = session["questions"][idx]
        correct = (qd["choices"][answer-1] == qd["ans"])
        feedback = "Correct! +10 pts." if correct else f"Wrong. Answer was: {qd['ans']}"
        if correct:
            cur.execute("UPDATE colleges SET total_score = total_score + 10 WHERE college = ?", (session["college"],))
            conn.commit()
        session["current"] += 1
        idx = session["current"]
        if idx >= len(session["questions"]):
            col = session["college"]
            del active_quizzes[phone]
            return f"{feedback}\n\nQuiz complete for {col}!\n\n🏁 Main Menu:\n• Start Competition → @enter_competition college=<A|B|C>\n• View Leaderboard → @show_leaderboard"
        qd = session["questions"][idx]
        opts = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(qd["choices"]))
        return f"{feedback}\nQ{idx+1} ({qd['diff']}): {qd['q']}\n{opts}\nReply with @enter_competition answer=<number>"
    return "Use @enter_competition to start or @show_leaderboard to view rankings."

@mcp.tool
async def show_leaderboard() -> str:
    rows = cur.execute("SELECT college,total_score FROM colleges ORDER BY total_score DESC").fetchall()
    lines = [f"{c}: {s}" for c, s in rows]
    return "🏆 Leaderboard 🏆\n" + "\n".join(lines)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8086)
