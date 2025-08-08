import os
import random
import sqlite3
import asyncio
from dotenv import load_dotenv
from fastmcp import FastMCP
import httpx

# --- Load environment variables ---
load_dotenv()
TOKEN = os.getenv("AUTH_TOKEN")
MY_NUMBER = os.getenv("MY_NUMBER")

assert TOKEN, "Please set AUTH_TOKEN in .env"
assert MY_NUMBER, "Please set MY_NUMBER in .env"

# --- MCP Server with Bearer Token ---
mcp = FastMCP("College Quiz MCP Server", bearer_token=TOKEN)

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
def validate() -> str:
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
                        q = item["question"].replace("&quot;", '"').replace("&#039;", "'").replace("&amp;", "&")
                        choices = item["incorrect_answers"] + [item["correct_answer"]]
                        # Clean up HTML entities in choices
                        choices = [choice.replace("&quot;", '"').replace("&#039;", "'").replace("&amp;", "&") for choice in choices]
                        correct_answer = item["correct_answer"].replace("&quot;", '"').replace("&#039;", "'").replace("&amp;", "&")
                        random.shuffle(choices)
                        questions.append({
                            "diff": diff.capitalize(),
                            "q": q,
                            "choices": choices,
                            "ans": correct_answer
                        })
    except Exception as e:
        print(f"Error fetching questions: {e}")
        # Fallback questions if API fails
        questions = [
            {"diff": "Easy", "q": "What is 2 + 2?", "choices": ["3", "4", "5", "6"], "ans": "4"},
            {"diff": "Medium", "q": "What is the capital of France?", "choices": ["London", "Berlin", "Paris", "Madrid"], "ans": "Paris"},
            {"diff": "Hard", "q": "What is the derivative of x²?", "choices": ["x", "2x", "x²", "2"], "ans": "2x"},
            {"diff": "Hard", "q": "What is the square root of 144?", "choices": ["10", "11", "12", "13"], "ans": "12"},
            {"diff": "Hard", "q": "What year did World War II end?", "choices": ["1944", "1945", "1946", "1947"], "ans": "1945"}
        ]
    return questions

# --- Tool: enter_competition ---
@mcp.tool
def enter_competition(
    college: str = None,
    answer: int = None
) -> str:
    # Use a simple phone number simulation since we can't access request context reliably
    phone = "default_user"  # In production, this would come from the authenticated user
    
    # Main menu
    if phone not in active_quizzes and college is None and answer is None:
        return (
            "🏁 Welcome to College Quiz Competition!\n\n"
            "🏛️ Available Colleges:\n"
            "• A. BITS\n"
            "• B. P\n" 
            "• C. G/H\n\n"
            "📝 To start: @enter_competition college=A\n"
            "📊 View leaderboard: @show_leaderboard"
        )

    # College selection
    if college and phone not in active_quizzes:
        mapping = {"A": "BITS", "B": "P", "C": "G/H"}
        col = mapping.get(college.upper())
        if not col:
            return "❌ Invalid choice. Please use A (BITS), B (P), or C (G/H)."
        
        # This is a simplified version - in production you'd use asyncio.create_task
        # For now, let's use fallback questions
        qs = [
            {"diff": "Easy", "q": "What is 2 + 2?", "choices": ["3", "4", "5", "6"], "ans": "4"},
            {"diff": "Medium", "q": "What is the capital of France?", "choices": ["London", "Berlin", "Paris", "Madrid"], "ans": "Paris"},
            {"diff": "Hard", "q": "What is the derivative of x²?", "choices": ["x", "2x", "x²", "2"], "ans": "2x"},
            {"diff": "Hard", "q": "What is the square root of 144?", "choices": ["10", "11", "12", "13"], "ans": "12"},
            {"diff": "Hard", "q": "What year did World War II end?", "choices": ["1944", "1945", "1946", "1947"], "ans": "1945"}
        ]
        
        active_quizzes[phone] = {"college": col, "questions": qs, "current": 0}
        qd = qs[0]
        opts = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(qd["choices"]))
        return (
            f"🎓 Starting quiz for {col}!\n\n"
            f"Q1 ({qd['diff']}): {qd['q']}\n\n"
            f"{opts}\n\n"
            "💡 Answer with: @enter_competition answer=1"
        )

    # Answer handling
    if answer is not None and phone in active_quizzes:
        session = active_quizzes[phone]
        idx = session["current"]
        qd = session["questions"][idx]
        
        if answer < 1 or answer > len(qd["choices"]):
            return f"❌ Invalid answer. Please choose 1-{len(qd['choices'])}"
        
        correct = (qd["choices"][answer-1] == qd["ans"])
        feedback = "✅ Correct! +10 pts." if correct else f"❌ Wrong. Correct answer: {qd['ans']}"
        
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
            total_score = cur.execute("SELECT total_score FROM colleges WHERE college = ?", (col,)).fetchone()[0]
            del active_quizzes[phone]
            return (
                f"{feedback}\n\n"
                f"🎉 Quiz complete for {col}!\n"
                f"🏛️ {col} total score: {total_score}\n\n"
                "🏁 Main Menu:\n"
                "📝 New quiz: @enter_competition\n"
                "📊 Leaderboard: @show_leaderboard"
            )

        # Next question
        qd = session["questions"][idx]
        opts = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(qd["choices"]))
        return (
            f"{feedback}\n\n"
            f"Q{idx+1} ({qd['diff']}): {qd['q']}\n\n"
            f"{opts}\n\n"
            f"💡 Answer with: @enter_competition answer=<number>"
        )

    # Fallback
    return (
        "🏁 College Quiz Competition\n\n"
        "📝 Start quiz: @enter_competition college=<A|B|C>\n"
        "📊 View leaderboard: @show_leaderboard"
    )

# --- Tool: show_leaderboard ---
@mcp.tool
def show_leaderboard() -> str:
    rows = cur.execute(
        "SELECT college, total_score FROM colleges ORDER BY total_score DESC"
    ).fetchall()
    
    if not any(score > 0 for _, score in rows):
        return (
            "🏆 College Competition Leaderboard\n\n"
            "No scores yet! Be the first to compete!\n\n"
            "📝 Start quiz: @enter_competition"
        )
    
    lines = [f"🥇 {c}: {s} points" if i == 0 else f"🥈 {c}: {s} points" if i == 1 else f"🥉 {c}: {s} points" for i, (c, s) in enumerate(rows)]
    return "🏆 College Competition Leaderboard\n\n" + "\n".join(lines)

# --- Run MCP Server ---
if __name__ == "__main__":
    mcp.run()
