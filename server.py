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
print(f"[DEBUG] Loaded AUTH_TOKEN: '{TOKEN}'")
print(f"[DEBUG] Loaded MY_NUMBER: '{MY_NUMBER}'")

# --- Auth Provider ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        print(f"[DEBUG] load_access_token called with token='{token}'")
        if token == self.token:
            print("[DEBUG] Token match successful")
            return AccessToken(
                token=token,
                client_id=MY_NUMBER,
                scopes=["*"],
                expires_at=None,
            )
        print("[DEBUG] Token mismatch")
        return None

# --- Initialize MCP ---
mcp = FastMCP("College Quiz MCP Server", auth=SimpleBearerAuthProvider(TOKEN))

# --- SQLite setup ---
print("[DEBUG] Initializing SQLite database...")
conn = sqlite3.connect("leaderboard.db", check_same_thread=False)
cur = conn.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS colleges(college TEXT PRIMARY KEY, total_score INTEGER)")
for college in ("BITS", "P", "G/H"):
    cur.execute("INSERT OR IGNORE INTO colleges VALUES(?, 0)", (college,))
conn.commit()
print("[DEBUG] DB initial content:", cur.execute("SELECT * FROM colleges").fetchall())

# --- In-memory state ---
active_quizzes: dict[str, dict] = {}

# --- Validate tool ---
@mcp.tool
async def validate() -> str:
    print(f"[DEBUG] validate() called, returning MY_NUMBER={MY_NUMBER}")
    return MY_NUMBER

# --- Fetch quiz questions ---
async def fetch_questions():
    print("[DEBUG] fetch_questions() called")
    questions = []
    try:
        async with httpx.AsyncClient() as client:
            for diff, count in (("easy", 1), ("medium", 1), ("hard", 3)):
                print(f"[DEBUG] Fetching {count} '{diff}' questions")
                resp = await client.get(
                    "https://opentdb.com/api.php",
                    params={"amount": count, "difficulty": diff, "type": "multiple"},
                    timeout=10
                )
                print(f"[DEBUG] API status={resp.status_code}")
                if resp.status_code == 200:
                    data = resp.json().get("results", [])
                    print(f"[DEBUG] Received {len(data)} questions for diff={diff}")
                    for item in data:
                        q = item["question"].replace("&quot;", '"').replace("&#039;", "'").replace("&amp;", "&")
                        choices = [
                            ch.replace("&quot;", '"').replace("&#039;", "'").replace("&amp;", "&")
                            for ch in (item["incorrect_answers"] + [item["correct_answer"]])
                        ]
                        correct = item["correct_answer"].replace("&quot;", '"').replace("&#039;", "'").replace("&amp;", "&")
                        random.shuffle(choices)
                        questions.append({"diff": diff.capitalize(), "q": q, "choices": choices, "ans": correct})
    except Exception as e:
        print(f"[ERROR] Error fetching trivia: {e}")

    if len(questions) < 5:
        print("[DEBUG] Using fallback questions")
        questions.extend([
            {"diff": "Easy", "q": "What is 2 + 2?", "choices": ["3", "4", "5", "6"], "ans": "4"},
            {"diff": "Medium", "q": "What is the capital of France?", "choices": ["London", "Berlin", "Paris", "Madrid"], "ans": "Paris"},
            {"diff": "Hard", "q": "What is the derivative of x²?", "choices": ["x", "2x", "x²", "2"], "ans": "2x"},
            {"diff": "Hard", "q": "What is the square root of 144?", "choices": ["10", "11", "12", "13"], "ans": "12"},
            {"diff": "Hard", "q": "What year did World War II end?", "choices": ["1944", "1945", "1946", "1947"], "ans": "1945"},
        ][:5 - len(questions)])
    print(f"[DEBUG] Total questions to use: {len(questions)}")
    return questions[:5]

# --- Main quiz tool ---
@mcp.tool
async def enter_competition(college: str | None = None, answer: int | None = None) -> str:
    phone = MY_NUMBER
    print(f"[DEBUG] enter_competition called: phone={phone}, college={college}, answer={answer}")

    if phone not in active_quizzes and college is None and answer is None:
        print("[DEBUG] Sending main menu")
        return (
            "🏁 Main Menu:\n"
            "• Start Competition → @enter_competition college=<A|B|C>\n"
            "• View Leaderboard → @show_leaderboard"
        )

    if college and phone not in active_quizzes:
        mapping = {"A": "BITS", "B": "P", "C": "G/H"}
        selected_college = mapping.get(college.upper())
        print(f"[DEBUG] Selected college={selected_college}")
        if not selected_college:
            return "❌ Invalid choice. Please use A (BITS), B (P), or C (G/H)."
        qs = await fetch_questions()
        active_quizzes[phone] = {"college": selected_college, "questions": qs, "current": 0}
        qd = qs[0]
        opts = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(qd["choices"]))
        print(f"[DEBUG] Starting quiz for {selected_college}")
        return (f"🎓 Quiz for {selected_college}!\nQ1 ({qd['diff']}): {qd['q']}\n{opts}\n"
                "Reply with @enter_competition answer=<number>")

    if answer is not None and phone in active_quizzes:
        session = active_quizzes[phone]
        idx = session["current"]
        qd = session["questions"][idx]
        print(f"[DEBUG] Answer received: {answer}, correct={qd['ans']}")
        if not (1 <= answer <= len(qd["choices"])):
            return f"❌ Invalid! Pick 1-{len(qd['choices'])}."
        is_correct = qd["choices"][answer - 1] == qd["ans"]
        feedback = "✅ Correct! +10 pts." if is_correct else f"❌ Wrong. Correct: {qd['ans']}"
        if is_correct:
            cur.execute("UPDATE colleges SET total_score = total_score + 10 WHERE college = ?", (session["college"],))
            conn.commit()
            print(f"[DEBUG] Updated DB for {session['college']}")
        session["current"] += 1
        if session["current"] >= len(session["questions"]):
            col = session["college"]
            score = cur.execute("SELECT total_score FROM colleges WHERE college = ?", (col,)).fetchone()[0]
            print(f"[DEBUG] Quiz complete for {col}, final score={score}")
            del active_quizzes[phone]
            return (f"{feedback}\n\n🎉 Quiz complete for {col}! Total: {score} points\n\n"
                    "🏁 Main Menu:\n"
                    "• Start Competition → @enter_competition college=<A|B|C>\n"
                    "• View Leaderboard → @show_leaderboard")
        qd = session["questions"][session["current"]]
        opts = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(qd["choices"]))
        return (f"{feedback}\nQ{session['current']+1} ({qd['diff']}): {qd['q']}\n{opts}\n"
                "Reply with @enter_competition answer=<number>")

    if college and phone in active_quizzes:
        print("[DEBUG] Attempt to start new quiz while one active")
        return "❗ Quiz already in progress. Finish current questions first."

    return "Use @enter_competition college=<A|B|C> to start or @show_leaderboard for rankings."

# --- Leaderboard tool ---
@mcp.tool
async def show_leaderboard() -> str:
    print("[DEBUG] show_leaderboard called")
    rows = cur.execute("SELECT college, total_score FROM colleges ORDER BY total_score DESC").fetchall()
    print(f"[DEBUG] DB rows: {rows}")
    if not any(s > 0 for _, s in rows):
        print("[DEBUG] No scores yet")
        return "🏆 College Competition Leaderboard\nNo scores yet—be the first!\nStart quiz: @enter_competition college=<A|B|C>"
    medals = ['🥇', '🥈', '🥉']
    lines = [f"{medals[i] if i < 3 else ''} {c}: {s} points" for i, (c, s) in enumerate(rows)]
    result = "🏆 College Competition Leaderboard\n" + "\n".join(lines)
    print(f"[DEBUG] Leaderboard output:\n{result}")
    return result

# --- Run MCP server ---
if __name__ == "__main__":
    print("[DEBUG] Starting MCP server on port 8086...")
    asyncio.run(mcp.run_async("streamable-http", host="0.0.0.0", port=8086))
