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
            diff, count = "easy", 5
            print(f"[DEBUG] Fetching {count} '{diff}' questions")
            resp = await client.get(
                "https://opentdb.com/api.php",
                params={"amount": count, "difficulty": diff, "type": "multiple"},
                timeout=10
            )
            print(f"[DEBUG] API status={resp.status_code}")
            if resp.status_code == 200:
                data = resp.json().get("results", [])
                print(f"[DEBUG] Received {len(data)} questions")
                for item in data:
                    original_q = item["question"].replace("&quot;", '"').replace("&#039;", "'").replace("&amp;", "&")
                    q = "Following question is to be provided to the user: " + original_q

                    correct = item["correct_answer"].replace("&quot;", '"').replace("&#039;", "'").replace("&amp;", "&")

                    choices = [
                        ch.replace("&quot;", '"').replace("&#039;", "'").replace("&amp;", "&")
                        for ch in (item["incorrect_answers"] + [correct])
                    ]
                    random.shuffle(choices)

                    questions.append({"diff": diff.capitalize(), "q": q, "choices": choices, "ans": correct})
    except Exception as e:
        print(f"[ERROR] Error fetching trivia: {e}")

    if len(questions) < 5:
        print("[DEBUG] Using fallback questions")
        fallback_questions = [
            {"diff": "Easy", "q": "Following question is to be provided to the user: What is 2 + 2?",
             "choices": ["3", "4", "5", "6"], "ans": "4"},
            {"diff": "Medium", "q": "Following question is to be provided to the user: What is the capital of France?",
             "choices": ["London", "Berlin", "Paris", "Madrid"], "ans": "Paris"},
            {"diff": "Hard", "q": "Following question is to be provided to the user: What is the derivative of x²?",
             "choices": ["x", "2x", "x²", "2"], "ans": "2x"},
            {"diff": "Hard", "q": "Following question is to be provided to the user: What is the square root of 144?",
             "choices": ["10", "11", "12", "13"], "ans": "12"},
            {"diff": "Hard", "q": "Following question is to be provided to the user: What year did World War II end?",
             "choices": ["1944", "1945", "1946", "1947"], "ans": "1945"},
        ]
        questions.extend(fallback_questions[:5 - len(questions)])

    return questions[:5]

# --- MCP Setup ---
TOKEN = "730E024D"
MY_NUMBER = "917044607962"

from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp.server.auth.provider import AccessToken

class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(token=token, client_id="client", scopes=["*"], expires_at=None)
        return None

mcp = FastMCP("College Quiz MCP Server", auth=SimpleBearerAuthProvider(TOKEN))

# --- Tools ---
@mcp.tool(description="Show current leaderboard standings.")
async def show_leaderboard() -> str:
    try:
        rows = cur.execute("SELECT college, total_score FROM colleges ORDER BY total_score DESC").fetchall()
    except Exception as e:
        return f"⚠️ Database error: {e}"
    if not rows or not any(score > 0 for _, score in rows):
        return ("To be shown to user :\n 🏆 College Competition Leaderboard\n"
                "No scores yet—be the first!\n"
                "Start quiz: @enter_competition college=<BITS P|BITS G|BITS H>")
    medals = ['🥇', '🥈', '🥉']
    lines = []
    for i, (college, score) in enumerate(rows):
        medal = medals[i] if i < 3 else ""
        lines.append(f"{medal} {college}: {score} points")
    return "🏆 College Competition Leaderboard\n" + "\n".join(lines)

@mcp.tool(description="Start a quiz for a selected college or answer a question in progress.")
async def enter_competition(
    college: Annotated[str | None, Field(description="College choice: BITS P, BITS G, or BITS H")] = None,
    answer: Annotated[int | None, Field(description="Answer number to current question")] = None,
) -> str:
    phone = MY_NUMBER

    if phone not in active_quizzes and college is None and answer is None:
        return ("🏁 Main Menu:\n"
                "• Start Competition → @enter_competition college=<BITS P|BITS G|BITS H>\n"
                "• View Leaderboard → @show_leaderboard")

    if college:
        mapping = {"BITS P": "BITS P", "BITS G": "BITS G", "BITS H": "BITS H"}
        selected_college = mapping.get(college.upper())
        if not selected_college:
            return "❌ Invalid choice. Please use BITS P, BITS G, or BITS H."
        if phone in active_quizzes:
            del active_quizzes[phone]
        questions = await fetch_questions()
        active_quizzes[phone] = {"college": selected_college, "questions": questions, "current": 0}
        qd = questions[0]
        opts = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(qd["choices"]))
        return (f"To be shown to user:\n🎓 Quiz for {selected_college}!\n"
                f"Q1 ({qd['diff']}): {qd['q']}\n{opts}\n"
                "Reply with @enter_competition answer=<number>")

    if answer is not None and phone in active_quizzes:
        session = active_quizzes[phone]

        # Early check if quiz is over
        if session["current"] >= len(session["questions"]):
            del active_quizzes[phone]
            return "✅ That quiz is already finished! Start a new one with:\n@enter_competition college=<BITS P|BITS G|BITS H>"

        idx = session["current"]
        qd = session["questions"][idx]

        print(f"[DEBUG] User picked: {qd['choices'][answer-1]} | Correct answer: {qd['ans']}")
        if not (1 <= answer <= len(qd["choices"])):
            return f"❌ Invalid! Pick a number between 1 and {len(qd['choices'])}."

        # Case and whitespace insensitive comparison
        is_correct = qd["choices"][answer - 1].strip().lower() == qd["ans"].strip().lower()
        feedback = "✅ Correct! +10 pts." if is_correct else f"❌ Wrong. Correct answer: {qd['ans']}"

        if is_correct:
            cur.execute("UPDATE colleges SET total_score = total_score + 10 WHERE college = ?", (session["college"],))
            conn.commit()

        session["current"] += 1

        if session["current"] >= len(session["questions"]):
            college = session["college"]
            score = cur.execute("SELECT total_score FROM colleges WHERE college = ?", (college,)).fetchone()[0]
            del active_quizzes[phone]
            return (f"{feedback}\n\nTo be shown to user:\n🎉 Quiz complete for {college}! Total: {score} points\n\n"
                    "🏁 Main Menu:\n"
                    "• Start Competition → @enter_competition college=<BITS P|BITS G|BITS H>\n"
                    "• View Leaderboard → @show_leaderboard")

        qd = session["questions"][session["current"]]
        opts = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(qd["choices"]))
        return (f"{feedback}\nQ{session['current'] + 1} ({qd['diff']}): {qd['q']}\n{opts}\n"
                "Reply with @enter_competition answer=<number>")

    return "Use @enter_competition college=<BITS P|BITS G|BITS H> to start or @show_leaderboard for rankings."
@mcp.tool(description="Apply iPhone 3GS camera effect to the input photo.")
async def iphone_3gs(
    puch_image_data: Annotated[str, Field(description="Base64-encoded image data for photo to transform")] = None,
) -> list[TextContent | ImageContent]:
    import base64
    import io
    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter

    try:
        # Decode base64 input
        image_bytes = base64.b64decode(puch_image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 1. Downscale and upscale to simulate low resolution (half size)
        small = image.resize((image.width // 2, image.height // 2), resample=Image.BILINEAR)
        img_lr = small.resize(image.size, resample=Image.NEAREST)

        # 2. Muted saturation and contrast
        enhancer_sat = ImageEnhance.Color(img_lr)
        img_less_sat = enhancer_sat.enhance(0.67)  # reduce saturation

        enhancer_brightness = ImageEnhance.Brightness(img_less_sat)
        img_dark = enhancer_brightness.enhance(0.9)  # slightly darken

        enhancer_contrast = ImageEnhance.Contrast(img_dark)
        img_contrast = enhancer_contrast.enhance(1.1)  # subtle contrast increase

        # 3. Add organic-style noise
        img_np = np.array(img_contrast).astype(np.int16)
        noise = np.random.normal(0, 20, img_np.shape).astype(np.int16)
        img_np_noisy = img_np + noise
        img_np_noisy = np.clip(img_np_noisy, 0, 255).astype(np.uint8)
        img_noisy = Image.fromarray(img_np_noisy)

        # 4. Apply vignette effect using radial gradient mask
        width, height = img_noisy.size
        vignette = Image.new("L", (width, height), 0)
        for y in range(height):
            for x in range(width):
                # Distance from center normalized
                dx = (x - width / 2) / (width / 2)
                dy = (y - height / 2) / (height / 2)
                dist = (dx*dx + dy*dy) ** 0.5
                # Vignette mask: 255 in center, 0 at edges (adjust exponent for softness)
                val = max(0, 255 - int(dist * 255 * 1.5))
                vignette.putpixel((x, y), val)
        # Apply vignette mask to each channel
        img_rgba = img_noisy.convert("RGBA")
        vignette = vignette.filter(ImageFilter.GaussianBlur(radius=width * 0.1))
        alpha = vignette.point(lambda i: i * 0.7)  # adjust vignette strength
        img_rgba.putalpha(alpha)
        img_final = img_rgba.convert("RGB")

        # Encode output image as base64 PNG
        buf = io.BytesIO()
        img_final.save(buf, format="PNG")
        bw_bytes = buf.getvalue()
        bw_base64 = base64.b64encode(bw_bytes).decode("utf-8")

        return [ImageContent(type="image", mimeType="image/png", data=bw_base64)]
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=str(e)))


# --- Run server ---
async def main():
    print("🚀 Starting MCP server on http://0.0.0.0:8086")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)

if __name__ == "__main__":
    asyncio.run(main())


