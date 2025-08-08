import os
import random
import sqlite3
import asyncio
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp.server.auth.provider import AccessToken
import httpx

# --- Load environment variables ---
load_dotenv()
TOKEN = os.getenv("AUTH_TOKEN")
MY_NUMBER = os.getenv("MY_NUMBER")

assert TOKEN, "Please set AUTH_TOKEN in .env"
assert MY_NUMBER, "Please set MY_NUMBER in .env"

# --- Auth Provider ---
class SimpleBearerAuth(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(token=token, client_id=MY_NUMBER, scopes=["*"], expires_at=None)
        return None

# --- MCP Server ---
mcp = FastMCP("College Quiz MCP Server", auth=SimpleBearerAuth(TOKEN))

# --- SQLite leaderboard ---
conn = sqlite3.connect("leaderboard.db", check_same_thread=False)
cur = conn.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS colleges(college TEXT PRIMARY KEY, total_score INTEGER)")
for c in ("BITS", "P", "G/H"):
    cur.execute("INSERT OR IGNORE INTO colleges VALUES(?, 0)", (c,))
conn.commit()

# --- In-memory quiz state ---
active_quizzes: dict[str, dict] = {}

# --- Tool: validate (required by Puch) ---
@mcp.tool
async def validate() -> str:
    return MY_NUMBER

# --- Fetch dynamic questions from Open Trivia DB ---
async def fetch_questions():
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
                q = item["question"]
                choices = item["incorrect_answers"] + [item["correct_answer"]]
                random.shuffle(choices)
                questions.append({
                    "diff": diff.capitalize(),
                    "q": q,
                    "choices": choices,
                    "ans": item["correct_answer"]
                })
    return questions

# --- Tool: enter_competition ---
@mcp.tool
async def enter_competition(
    college: str | None = None,
    answer: int | None = None
) -> str:
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
        correct = (qd["choices"][answer-1] == qd["ans"])
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

# --- Tool: show_leaderboard ---
@mcp.tool
async def show_leaderboard() -> str:
    rows = cur.execute(
        "SELECT college, total_score FROM colleges ORDER BY total_score DESC"
    ).fetchall()
    lines = [f"{c}: {s}" for c, s in rows]
    return "🏆 Leaderboard 🏆\n" + "\n".join(lines)

# --- Run MCP Server ---
async def main():
    print("🚀 Starting College Quiz MCP on http://0.0.0.0:8086")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)

if __name__ == "__main__":
    asyncio.run(main())
