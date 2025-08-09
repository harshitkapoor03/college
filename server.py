import asyncio
import random
import sqlite3
import httpx
from typing import Annotated
from fastmcp import FastMCP
from mcp.types import Field

# --- DB Setup ---
conn = sqlite3.connect("leaderboard.db", check_same_thread=False)
cur = conn.cursor()
cur.execute("""CREATE TABLE IF NOT EXISTS colleges (
    college TEXT PRIMARY KEY,
    total_score INTEGER DEFAULT 0
)""")
for c in ["BITS P", "BITS G", "BITS H"]:
    cur.execute("INSERT OR IGNORE INTO colleges (college) VALUES (?)", (c,))
conn.commit()

# --- Active quiz sessions ---
active_quizzes = {}

# --- Fetch questions from Open Trivia DB API ---
async def fetch_questions():
    print("[DEBUG] fetch_questions() called")
    questions = []
    try:
        async with httpx.AsyncClient() as client:
            for diff, count in (("easy", 5)):
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
                        q = "Following question is to be provided to the user: " + original_q
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

# --- MCP Setup ---
TOKEN = "730E024D"
MY_NUMBER = "917044607962"

from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp.server.auth.provider import AccessToken

class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        # Generate a keypair (or use your own keys)
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="client",
                scopes=["*"],
                expires_at=None,
            )
        return None

mcp = FastMCP("College Quiz MCP Server", auth=SimpleBearerAuthProvider(TOKEN))

# --- Tools ---
@mcp.tool
async def validate() -> str:
    return MY_NUMBER
@mcp.tool
async def show_leaderboard() -> str:
    rows = cur.execute("SELECT college, total_score FROM colleges ORDER BY total_score DESC").fetchall()
    if not any(score > 0 for _, score in rows):
        return ("To be shown to user :\n 🏆 College Competition Leaderboard\n"
                "No scores yet—be the first!\n"
                "Start quiz: @enter_competition college=<A|B|C>")
    medals = ['🥇', '🥈', '🥉']
    lines = []
    for i, (college, score) in enumerate(rows):
        medal = medals[i] if i < 3 else ""
        lines.append(f"{medal} {college}: {score} points")
    return "🏆 College Competition Leaderboard\n" + "\n".join(lines)

@mcp.tool
async def enter_competition(
    college: Annotated[str | None, Field(description="College choice: A, B, or C")] = None,
    answer: Annotated[int | None, Field(description="Answer number to current question")] = None,
) -> str:
    phone = MY_NUMBER

    if phone not in active_quizzes and college is None and answer is None:
        return ("🏁 Main Menu:\n"
                "• Start Competition → @enter_competition college=<A|B|C>\n"
                "• View Leaderboard → @show_leaderboard")

    if college and phone not in active_quizzes:
        mapping = {"A": "BITS P", "B": "BITS G", "C": "BITS H"}
        selected_college = mapping.get(college.upper())
        if not selected_college:
            return "❌ Invalid choice. Please use A (BITS), B (P), or C (G/H)."

        questions = await fetch_questions()
        active_quizzes[phone] = {"college": selected_college, "questions": questions, "current": 0}
        qd = questions[0]
        opts = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(qd["choices"]))
        return (f"To be shown to user: \n🎓 Quiz for {selected_college}!\n"
                f"Q1 ({qd['diff']}): {qd['q']}\n{opts}\n"
                "Reply with @enter_competition answer=<number>")

    if answer is not None and phone in active_quizzes:
        session = active_quizzes[phone]
        idx = session["current"]
        qd = session["questions"][idx]

        if not (1 <= answer <= len(qd["choices"])):
            return f"❌ Invalid! Pick a number between 1 and {len(qd['choices'])}."

        is_correct = (qd["choices"][answer - 1] == qd["ans"])
        feedback = "✅ Correct! +10 pts." if is_correct else f"❌ Wrong. Correct answer: {qd['ans']}"

        if is_correct:
            cur.execute("UPDATE colleges SET total_score = total_score + 10 WHERE college = ?", (session["college"],))
            conn.commit()

        session["current"] += 1

        if session["current"] >= len(session["questions"]):
            college = session["college"]
            score = cur.execute("SELECT total_score FROM colleges WHERE college = ?", (college,)).fetchone()[0]
            del active_quizzes[phone]
            return (f"{feedback}\n\n To be shown to user: \n🎉 Quiz complete for {college}! Total: {score} points\n\n"
                    "🏁 Main Menu:\n"
                    "• Start Competition → @enter_competition college=<A|B|C>\n"
                    "• View Leaderboard → @show_leaderboard")

        qd = session["questions"][session["current"]]
        opts = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(qd["choices"]))
        return (f"{feedback}\n"
                f"Q{session['current'] + 1} ({qd['diff']}): {qd['q']}\n{opts}\n"
                "Reply with @enter_competition answer=<number>")

    if college and phone in active_quizzes:
        return "❗ Quiz already in progress. Please finish current questions first."

    return "Use @enter_competition college=<A|B|C> to start or @show_leaderboard for rankings."


# --- Run server ---
async def main():
    print("🚀 Starting MCP server on http://0.0.0.0:8086")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)

if __name__ == "__main__":
    asyncio.run(main())


