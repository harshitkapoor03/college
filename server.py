import os
import random
import sqlite3
import asyncio
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.jwt import JWTVerifier, JWTVerifierSettings
import httpx

# ── ENVIRONMENT VARIABLES ─────────────
load_dotenv()
TOKEN = os.getenv("AUTH_TOKEN")
MY_NUMBER = os.getenv("MY_NUMBER")
assert TOKEN and MY_NUMBER, "Set AUTH_TOKEN and MY_NUMBER in .env"

# ── JWT AUTH PROVIDER SETUP ───────────
jwt_settings = JWTVerifierSettings(
    issuer="puchai",
    audiences=["puchai-mcp"],
    jwks_uri=None,
    algorithms=["HS256"],
)
verifier = JWTVerifier(jwt_settings)

# Accepts only your static AUTH_TOKEN (custom)
async def load_access_token(token: str):
    if token == TOKEN:
        return {"client_id": MY_NUMBER, "scopes": ["*"]}
    return None

verifier.load_access_token = load_access_token

# ── FASTMCP SERVER ────────────────────
mcp = FastMCP("College Quiz MCP Server", auth=verifier)


# ── SQLITE LEADERBOARD ────────────────
conn = sqlite3.connect("leaderboard.db", check_same_thread=False)
cur = conn.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS colleges(college TEXT PRIMARY KEY, total_score INTEGER)")
for c in ("BITS", "P", "G/H"):
    cur.execute("INSERT OR IGNORE INTO colleges VALUES(?, 0)", (c,))
conn.commit()


# ── IN-MEMORY ACTIVE QUIZ STATE ───────
active_quizzes = {}


# ── TOOL: validate (required by Puch AI) ──
@mcp.tool
async def validate() -> str:
    return MY_NUMBER


# ── DYNAMIC QUESTION FETCH (Open Trivia DB) ──
async def fetch_questions():
    questions = []
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
                    choices = [ch.replace("&quot;", '"').replace("&#039;", "'").replace("&amp;", "&") for ch in choices]
                    correct = item["correct_answer"].replace("&quot;", '"').replace("&#039;", "'").replace("&amp;", "&")
                    random.shuffle(choices)
                    questions.append({
                        "diff": diff.capitalize(),
                        "q": q,
                        "choices": choices,
                        "ans": correct,
                    })
    if len(questions) < 5:  # Fallback if not enough
        questions += [
            {"diff": "Easy", "q": "What is 2 + 2?", "choices": ["3", "4", "5", "6"], "ans": "4"},
            {"diff": "Medium", "q": "What is the capital of France?", "choices": ["London", "Berlin", "Paris", "Madrid"], "ans": "Paris"},
            {"diff": "Hard", "q": "Derivative of x²?", "choices": ["x", "2x", "x²", "2"], "ans": "2x"},
            {"diff": "Hard", "q": "Square root of 144?", "choices": ["10", "11", "12", "13"], "ans": "12"},
            {"diff": "Hard", "q": "Year WWII ended?", "choices": ["1944", "1945", "1946", "1947"], "ans": "1945"},
        ][:5-len(questions)]
    return questions


# ── MAIN PUBLIC QUIZ TOOL ──────────────
@mcp.tool
async def enter_competition(
    college: str | None = None,
    answer: int | None = None
) -> str:
    phone = mcp.last_message().access_token["client_id"]

    # Show Main Menu
    if phone not in active_quizzes and college is None and answer is None:
        return (
            "🏁 Main Menu:\n"
            "• Start Competition → @enter_competition college=<A|B|C>\n"
            "• View Leaderboard → @show_leaderboard"
        )

    # College selection: start quiz
    if college and phone not in active_quizzes:
        mapping = {"A": "BITS", "B": "P", "C": "G/H"}
        col = mapping.get(college.upper())
        if not col:
            return "❌ Invalid college. Use A (BITS), B (P), or C (G/H)."
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
        # Quiz completed
        if idx >= len(session["questions"]):
            col = session["college"]
            del active_quizzes[phone]
            total = cur.execute("SELECT total_score FROM colleges WHERE college = ?", (col,)).fetchone()[0]
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

    # If user tries to pick college while quiz in progress
    if college and phone in active_quizzes:
        return "You're already in a quiz! Finish answering the questions."

    # Fallback
    return (
        "Use @enter_competition college=<A|B|C> to start, or @show_leaderboard to view rankings."
    )


# ── LEADERBOARD TOOL ───────────────
@mcp.tool
async def show_leaderboard() -> str:
    rows = cur.execute("SELECT college, total_score FROM colleges ORDER BY total_score DESC").fetchall()
    if not any(score > 0 for _,score in rows):
        return (
            "🏆 College Competition Leaderboard\n"
            "No scores yet—be the first!\n"
            "Start quiz: @enter_competition college=<A|B|C>"
        )
    medals = ['🥇', '🥈', '🥉']
    lines = [f"{medals[i] if i < 3 else ''} {c}: {s} points" for i, (c,s) in enumerate(rows)]
    return "🏆 College Competition Leaderboard\n" + "\n".join(lines)


# ── RUN THE SERVER ─────────────────
if __name__ == "__main__":
    asyncio.run(mcp.run_async("streamable-http", host="0.0.0.0", port=8086))
