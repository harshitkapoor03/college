import os, random, sqlite3, asyncio
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.jwt import JWTVerifier, RSAKeyPair
from fastmcp.server.auth.models import AccessToken  # Corrected import path
import httpx
from fastapi import FastAPI

# 1. Load credentials
load_dotenv()
TOKEN = os.getenv("AUTH_TOKEN")
MY_NUMBER = os.getenv("MY_NUMBER")
assert TOKEN and MY_NUMBER, "Set AUTH_TOKEN and MY_NUMBER in .env"

# 2. Updated Authentication Provider
class SimpleBearerAuth(JWTVerifier):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token
        
    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id=MY_NUMBER,
                scopes=["*"],
                expires_at=None,
                claims={}  # Add empty claims dictionary
            )
        return None

# 3. Initialize FastAPI and MCP
app = FastAPI()
mcp = FastMCP("College Quiz MCP", auth=SimpleBearerAuth(TOKEN))

# Health check endpoint
@app.get("/")
def health_check():
    return {"status": "healthy", "service": "College Quiz MCP"}

# Mount MCP under /mcp path
app.mount("/mcp", mcp.app)

# 4. Persistent leaderboard (SQLite)
conn = sqlite3.connect("leaderboard.db", check_same_thread=False)
cur = conn.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS colleges(college TEXT PRIMARY KEY, total_score INTEGER)")
for c in ("BITS","P","G/H"):
    cur.execute("INSERT OR IGNORE INTO colleges VALUES(?,0)", (c,))
conn.commit()

# 5. In-memory quiz state
active_quizzes: dict[str, dict] = {}

# 6. validate tool (required by Puch AI)
@mcp.tool
async def validate() -> str:
    return MY_NUMBER

# 7. Fetch dynamic questions from Open Trivia DB
async def fetch_questions():
    questions = []
    async with httpx.AsyncClient() as client:
        for diff,count in (("easy",1),("medium",1),("hard",3)):
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

# 8. Public tool: single entry point
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
        mapping = {"A":"BITS","B":"P","C":"G/H"}
        col = mapping.get(college.upper())
        if not col:
            return "Invalid choice. Use A, B, or C."
        qs = await fetch_questions()
        active_quizzes[phone] = {"college":col, "questions":qs, "current":0}
        qd = qs[0]
        opts = "\n".join(f"{i+1}. {opt}" for i,opt in enumerate(qd["choices"]))
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
        feedback = "Correct! +10 pts." if correct else f"Wrong. Answer was: {qd['ans']}"
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
                f"Quiz complete for {col}!\n\n"
                "🏁 Main Menu:\n"
                "• Start Competition → @enter_competition college=<A|B|C>\n"
                "• View Leaderboard → @show_leaderboard"
            )

        # Next question
        qd = session["questions"][idx]
        opts = "\n".join(f"{i+1}. {opt}" for i,opt in enumerate(qd["choices"]))
        return (
            f"{feedback}\n"
            f"Q{idx+1} ({qd['diff']}): {qd['q']}\n"
            f"{opts}\n"
            "Reply with @enter_competition answer=<number>"
        )

    # Fallback
    return "Use @enter_competition to start or @show_leaderboard to view rankings."

# 9. Public tool: leaderboard
@mcp.tool
async def show_leaderboard() -> str:
    rows = cur.execute(
        "SELECT college,total_score FROM colleges ORDER BY total_score DESC"
    ).fetchall()
    lines = [f"{c}: {s}" for c,s in rows]
    return "🏆 Leaderboard 🏆\n" + "\n".join(lines)

# 10. Run the server
if __name__ == "__main__":
    asyncio.run(app.run_async(host="0.0.0.0", port=8086))
