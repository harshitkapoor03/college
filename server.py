import asyncio
import random
import sqlite3
import httpx
from typing import Annotated
from fastmcp import FastMCP
from mcp.types import Field
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp.server.auth.provider import AccessToken
from fastmcp.server.types import Context  # ✅ Correct context import for session_id

# --- DB Setup ---
conn = sqlite3.connect("leaderboard.db", check_same_thread=False)
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS colleges (
    college TEXT PRIMARY KEY,
    total_score INTEGER DEFAULT 0
)
""")
for c in ["BITS", "P", "G/H"]:
    cur.execute("INSERT OR IGNORE INTO colleges (college) VALUES (?)", (c,))
conn.commit()

# --- MCP Config ---
TOKEN = "730E024D"
OWNER_PHONE = "917044607962"  # ✅ for Puch AI validate tool
COLLEGE_MAP = {"A": "BITS", "B": "P", "C": "G/H"}
active_quizzes = {}  # Stores per-session quiz state

# --- Auth Provider ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key,
                         jwks_uri=None,
                         issuer=None,
                         audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(token=token,
                               client_id="client",
                               scopes=["*"],
                               expires_at=None)
        return None

# --- MCP Server ---
mcp = FastMCP("College Quiz MCP Server", auth=SimpleBearerAuthProvider(TOKEN))

# --- Fetch Questions ---
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
                        q = item["question"].replace("&quot;", '"').replace("&#039;", "'").replace("&amp;", "&")
                        choices = [
                            ch.replace("&quot;", '"').replace("&#039;", "'").replace("&amp;", "&")
                            for ch in (item["incorrect_answers"] + [item["correct_answer"]])
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
        print(f"[ERROR] Error fetching trivia: {e}")

    if len(questions) < 5:
        questions.extend([
            {"diff": "Easy", "q": "What is 2 + 2?", "choices": ["3", "4", "5", "6"], "ans": "4"},
            {"diff": "Medium", "q": "What is the capital of France?", "choices": ["London", "Berlin", "Paris", "Madrid"], "ans": "Paris"},
            {"diff": "Hard", "q": "What is the derivative of x²?", "choices": ["x", "2x", "x²", "2"], "ans": "2x"},
            {"diff": "Hard", "q": "What is the square root of 144?", "choices": ["10", "11", "12", "13"], "ans": "12"},
            {"diff": "Hard", "q": "What year did World War II end?", "choices": ["1944", "1945", "1946", "1947"], "ans": "1945"},
        ][:5 - len(questions)])
    return questions[:5]

# --- Tools ---
@mcp.tool
async def validate() -> str:
    """Required by Puch AI to authenticate MCP server owner."""
    return OWNER_PHONE

@mcp.tool
async def show_leaderboard() -> str:
    rows = cur.execute("SELECT college, total_score FROM colleges ORDER BY total_score DESC").fetchall()
    medals = ['🥇', '🥈', '🥉']
    header = "🏆 College Competition Leaderboard\n"
    if not any(score > 0 for _, score in rows):
        return header + "No scores yet—be the first!\nStart quiz: @start_quiz college=<A|B|C>"
    lines = []
    for i, (college, score) in enumerate(rows):
        medal = medals[i] if i < 3 else ""
        lines.append(f"{medal} {college}: {score} points")
    return header + "\n".join(lines)

@mcp.tool
async def start_quiz(context: Context,
                     college: Annotated[str, Field(description="College choice: A, B, or C")]) -> str:
    session_id = context.session_id
    if not session_id:
        return "❌ Session context not found. Please reconnect."

    if session_id in active_quizzes:
        return "❗ You already have a quiz active! Finish it before starting another."

    selected_college = COLLEGE_MAP.get((college or "").upper())
    if not selected_college:
        return "❌ Invalid college. Please use A (BITS), B (P), or C (G/H)."

    questions = await fetch_questions()
    active_quizzes[session_id] = {
        "college": selected_college,
        "questions": questions,
        "current": 0,
        "score": 0
    }
    qd = questions[0]
    opts = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(qd["choices"]))
    return (
        f"🎓 Quiz for {selected_college}!\n"
        f"Q1 ({qd['diff']}): {qd['q']}\n{opts}\n"
        "Reply with @answer_question answer=<number>"
    )

@mcp.tool
async def answer_question(context: Context,
                          answer: Annotated[int, Field(description="Answer number for current question")]) -> str:
    session_id = context.session_id
    session = active_quizzes.get(session_id)
    if not session:
        return "❌ No quiz in progress. Start one with @start_quiz college=<A|B|C>."

    idx = session["current"]
    qd = session["questions"][idx]

    if not (1 <= answer <= len(qd["choices"])):
        return f"❌ Invalid choice! Pick 1–{len(qd['choices'])}."

    chosen = qd["choices"][answer - 1]
    is_correct = chosen == qd["ans"]
    feedback = "✅ Correct! +10 pts." if is_correct else f"❌ Wrong. Correct answer: {qd['ans']}"

    if is_correct:
        session["score"] += 10

    session["current"] += 1

    if session["current"] >= len(session["questions"]):
        # Quiz finished
        cur.execute(
            "UPDATE colleges SET total_score = total_score + ? WHERE college = ?",
            (session["score"], session["college"])
        )
        conn.commit()
        total_score = cur.execute(
            "SELECT total_score FROM colleges WHERE college = ?",
            (session["college"],)
        ).fetchone()[0]
        del active_quizzes[session_id]
        return (
            f"{feedback}\n\n"
            f"🎉 Quiz complete for {session['college']}! You scored: {session['score']} pts.\n"
            f"🏆 College Total: {total_score}\n\n"
            "To play again: @start_quiz college=<A|B|C>\n"
            "Or view rankings: @show_leaderboard"
        )

    # Next question
    next_q = session["questions"][session["current"]]
    opts = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(next_q["choices"]))
    return (
        f"{feedback}\n"
        f"Q{session['current'] + 1} ({next_q['diff']}): {next_q['q']}\n{opts}\n"
        "Reply with @answer_question answer=<number>"
    )

# --- Run server ---
async def main():
    print("🚀 Starting MCP server on http://0.0.0.0:8086")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)

if __name__ == "__main__":
    asyncio.run(main())
