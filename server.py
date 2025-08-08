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

# --- Auth Provider (WORKS WITH FASTMCP 2.x) ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id=MY_NUMBER,  # .env phone number is always used as "client" identifier
                scopes=["*"],
                expires_at=None,
            )
        return None

# --- MCP Server (correct usage) ---
mcp = FastMCP("College Quiz MCP Server", auth=SimpleBearerAuthProvider(TOKEN))

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
    try:
        async with httpx.AsyncClient() as client:
            for diff, count in (("easy", 1), ("medium", 1), ("hard", 3)):
                resp = await client.get(
                    "https://opentdb.com/api.php",
                    params={"amount": count, "difficulty": diff, "type": "multiple"},
                    timeout=10
                )
                if resp.status_code == 200:
                    data = resp.json().get("results", [])
                    for item in data:
                        q = (item["question"]
                             .replace("&quot;", '"')
                             .replace("&#039;", "'")
                             .replace("&amp;", "&"))
                        choices = item["incorrect_answers"] + [item["correct_answer"]]
                        choices = [
                            ch.replace("&quot;", '"').replace("&#039;", "'").replace("&amp;", "&")
                            for ch in choices
                        ]
                        correct = item["correct_answer"].replace("&quot;", '"').replace("&#039;", "'").replace("&amp;", "&")
                        random.shuffle(choices)
                        questions.append({
                            "diff": diff.capitalize(),
                            "q": q,
                            "choices": choices,
                            "ans": correct
                        })
    except Exception as e:
        print(f"Error fetching trivia: {e}")
    # Fallback to static questions if needed
    if len(questions) < 5:
        questions.extend([
            {"diff": "Easy", "q": "What is 2 + 2?", "choices": ["3", "4", "5", "6"], "ans": "4"},
            {"diff": "Medium", "q": "What is the capital of France?", "choices": ["London", "Berlin", "Paris", "Madrid"], "ans": "Paris"},
            {"diff": "Hard", "q": "What is the derivative of x²?", "choices": ["x", "2x", "x²", "2"], "ans": "2x"},
            {"diff": "Hard", "q": "What is the square root of 144?", "choices": ["10", "11", "12", "13"], "ans": "12"},
            {"diff": "Hard", "q": "What year did World War II end?", "choices": ["1944", "1945", "1946", "1947"], "ans": "1945"},
        ][:5 - len(questions)])
    return questions[:5]

# --- Main public user tool: the entire quiz UX ---
@mcp.tool
async def enter_competition(
    college: str | None = None,
    answer: int | None = None
) -> str:
    phone = MY_NUMBER#mcp.last_message().access_token.client_id

    # Main menu
    if phone not in active_quizzes and college is None and answer is None:
        return (
            "🏁 Main Menu:\n"
            "• Start Competition → @enter_competition college=<A|B|C>\n"
            "• View Leaderboard → @show_leaderboard"
        )

    # College selection: start new quiz
    if college and phone not in active_quizzes:
        mapping = {"A": "BITS", "B": "P", "C": "G/H"}
        col = mapping.get(college.upper())
        if not col:
            return "❌ Invalid choice. Please use A (BITS), B (P), or C (G/H)."
        qs = await fetch_questions()
        active_quizzes[phone] = {"college": col, "questions": qs, "current": 0}
        qd = qs[0]
        opts = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(qd["choices"]))
        return (
            f"🎓 Quiz for {col}!\n"
            f"Q1 ({qd['diff']}): {qd['q']}\n"
            f"{opts}\n"
            "Reply with @enter_competition answer=<number>"
        )

    # Answer handling
    if answer is not None and phone in active_quizzes:
        session = active_quizzes[phone]
        idx = session["current"]
        qd = session["questions"][idx]
        # Defensive: answer must be in range
        if not (1 <= answer <= len(qd["choices"])):
            return f"❌ Invalid! Pick 1-{len(qd['choices'])}."
        is_correct = (qd["choices"][answer-1] == qd["ans"])
        feedback = "✅ Correct! +10 pts." if is_correct else f"❌ Wrong. Correct: {qd['ans']}"
        if is_correct:
            cur.execute(
                "UPDATE colleges SET total_score = total_score + 10 WHERE college = ?",
                (session["college"],)
            )
            conn.commit()
        session["current"] += 1
        idx = session["current"]
        # Quiz done
        if idx >= len(session["questions"]):
            col = session["college"]
            total = cur.execute("SELECT total_score FROM colleges WHERE college = ?", (col,)).fetchone()[0]
            del active_quizzes[phone]
            return (
                f"{feedback}\n\n"
                f"🎉 Quiz complete for {col}! Total: {total}\n\n"
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

    if college and phone in active_quizzes:
        return "You're already in a quiz! Finish answering the current questions first."

    return (
        "Use @enter_competition college=<A|B|C> to start or @show_leaderboard for rankings."
    )

# --- Tool: show_leaderboard ---
@mcp.tool
async def show_leaderboard() -> str:
    rows = cur.execute(
        "SELECT college, total_score FROM colleges ORDER BY total_score DESC"
    ).fetchall()
    if not any(score > 0 for _, score in rows):
        return (
            "🏆 College Competition Leaderboard\n"
            "No scores yet—be the first!\n"
            "Start quiz: @enter_competition college=<A|B|C>"
        )
    medals = ['🥇', '🥈', '🥉']
    lines = [f"{medals[i] if i < 3 else ''} {c}: {s} points" for i, (c, s) in enumerate(rows)]
    return "🏆 College Competition Leaderboard\n" + "\n".join(lines)

# --- Run MCP Server ---
if __name__ == "__main__":
    asyncio.run(mcp.run_async("streamable-http", host="0.0.0.0", port=8086))

