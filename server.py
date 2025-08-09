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


# from typing import Annotated
# from mcp.types import TextContent, ImageContent, INTERNAL_ERROR
# from mcp import ErrorData, McpError
# from pydantic import Field

# @mcp.tool(description="Apply Kodak vintage-style filter to an input photo.")
# async def vintage_photo_filter(
#     puch_image_data: Annotated[str, Field(description="Base64-encoded image data to transform")] = None,
# ) -> list[TextContent | ImageContent]:
#     import base64
#     import io
#     from PIL import Image, ImageEnhance, ImageFilter
#     import numpy as np

#     try:
#         # Decode input base64 image
#         image_bytes = base64.b64decode(puch_image_data)
#         image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

#         # Apply Kodak-style warm tint
#         r, g, b = image.split()
#         r = r.point(lambda i: i * 1.02)
#         g = g.point(lambda i: i * 1.01)
#         b = b.point(lambda i: i * 0.95)
#         image = Image.merge("RGB", (r, g, b))

#         # Add slight Gaussian blur for vintage feel
#         image = image.filter(ImageFilter.GaussianBlur(radius=1.3))

#         # (Optional) Add grain/noise - convert to np, add noise, convert back
#         img_np = np.array(image).astype(np.int16)
#         noise = np.random.normal(0, 10, img_np.shape).astype(np.int16)
#         img_noisy = np.clip(img_np + noise, 0, 255).astype(np.uint8)
#         image = Image.fromarray(img_noisy)

#         # Save processed image to bytes buffer as PNG
#         buf = io.BytesIO()
#         image.save(buf, format="PNG")
#         output_bytes = buf.getvalue()

#         # Encode output image as base64
#         encoded_img = base64.b64encode(output_bytes).decode("utf-8")

#         # Return as list with ImageContent object
#         return [ImageContent(type="image", mimeType="image/png", data=encoded_img)]

#     except Exception as e:
#         raise McpError(ErrorData(code=INTERNAL_ERROR, message=str(e)))
#original kodak from here from typing import Annotated
# from mcp.types import TextContent, ImageContent, INTERNAL_ERROR
# from mcp import ErrorData, McpError
# from pydantic import Field

# @mcp.tool(description="Apply Kodak + iPhone 3GS vintage-style filter to an input photo.")
# async def vintage_photo_filter(
#     puch_image_data: Annotated[str, Field(description="Base64-encoded image data to transform")] = None,
# ) -> list[TextContent | ImageContent]:
#     import base64
#     import io
#     from PIL import Image, ImageEnhance, ImageFilter
#     import numpy as np

#     try:
#         # Decode base64 input
#         image_bytes = base64.b64decode(puch_image_data)
#         image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#         w, h = image.size

#         # -----------------
#         # Step 1: Kodak-style warm tint
#         # -----------------
#         r, g, b = image.split()
#         r = r.point(lambda i: i * 1.02)
#         g = g.point(lambda i: i * 1.01)
#         b = b.point(lambda i: i * 0.95)
#         image = Image.merge("RGB", (r, g, b))

#         # -----------------
#         # Step 2: Slight Gaussian blur
#         # -----------------
#         image = image.filter(ImageFilter.GaussianBlur(radius=1.3))

#         # -----------------
#         # Step 3: Add film grain
#         # -----------------
#         img_np = np.array(image).astype(np.int16)
#         noise = np.random.normal(0, 10, img_np.shape).astype(np.int16)
#         img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
#         image = Image.fromarray(img_np)

#         # -----------------
#         # Step 4: Low-res simulation (iPhone 3GS style)
#         # -----------------
#         lowres_size = (w // 2, h // 2)
#         image_small = image.resize(lowres_size, Image.BILINEAR)
#         image = image_small.resize((w, h), Image.NEAREST)

#         # -----------------
#         # Step 5: Reduce saturation
#         # -----------------
#         enhancer = ImageEnhance.Color(image)
#         image = enhancer.enhance(0.67)  # ~67% saturation

#         # -----------------
#         # Step 6: Add vignette effect
#         # -----------------
#         img_np = np.array(image).astype(np.float32)
#         y, x = np.ogrid[:h, :w]
#         center_x, center_y = w / 2, h / 2
#         sigma_x, sigma_y = w / 2, h / 2
#         mask = np.exp(-((x - center_x) ** 2 / (2 * sigma_x ** 2) +
#                         (y - center_y) ** 2 / (2 * sigma_y ** 2)))
#         mask = (mask / mask.max())  # normalize
#         for i in range(3):
#             img_np[:, :, i] *= mask
#         image = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))

#         # -----------------
#         # Step 7: Encode output as base64 PNG
#         # -----------------
#         buf = io.BytesIO()
#         image.save(buf, format="PNG")
#         encoded_img = base64.b64encode(buf.getvalue()).decode("utf-8")

#         return [ImageContent(type="image", mimeType="image/png", data=encoded_img)]

#     except Exception as e:
#         raise McpError(ErrorData(code=INTERNAL_ERROR, message=str(e)))
from typing import Annotated
from mcp.types import TextContent, ImageContent, INTERNAL_ERROR
from mcp import ErrorData, McpError
from pydantic import Field

@mcp.tool(description="Apply Kodak + iPhone 3GS vintage-style filter to an input photo.")
async def vintage_photo_filter(
    puch_image_data: Annotated[str, Field(description="Base64-encoded image data to transform")] = None,
) -> list[TextContent | ImageContent]:
    import base64
    import io
    from PIL import Image, ImageEnhance, ImageFilter
    import numpy as np

    try:
        # Decode base64 input
        image_bytes = base64.b64decode(puch_image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = image.size

        # Step 1: Kodak-style warm tint
        r, g, b = image.split()
        r = r.point(lambda i: i * 1.02)
        g = g.point(lambda i: i * 1.01)
        b = b.point(lambda i: i * 0.95)
        image = Image.merge("RGB", (r, g, b))

        # Step 2: Slight Gaussian blur
        image = image.filter(ImageFilter.GaussianBlur(radius=1.3))

        # Step 3: Add film grain
        img_np = np.array(image).astype(np.int16)
        noise = np.random.normal(0, 17, img_np.shape).astype(np.int16)
        img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(img_np)

        # Step 4: Low-res simulation
        yu = max(w, h)
        if yu > 4000:
            yi = 2
        elif yu > 3000:
            yi = 1.5
        elif yu > 2000:
            yi = 1
        elif yu > 1100:
            yi = 0.5
        else:
            yi = -0.15

        lowres_size = (int(w // (1.4 + yi)), int(h // (1.4 + yi)))
        image_small = image.resize(lowres_size, Image.BILINEAR)
        image = image_small.resize((w, h), Image.NEAREST)

        # Step 5: Reduce saturation
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(0.71)

        # Step 6: Add vignette effect
        img_np = np.array(image).astype(np.float32)
        y, x = np.ogrid[:h, :w]
        center_x, center_y = w / 2, h / 2
        sigma_x, sigma_y = w / 2, h / 2
        mask = np.exp(-((x - center_x) ** 2 / (2 * sigma_x ** 2) +
                        (y - center_y) ** 2 / (2 * sigma_y ** 2)))
        mask = mask / mask.max()
        for i in range(3):
            img_np[:, :, i] *= mask
        image = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))

        # Step 7: Increase brightness slightly
        brightness_enhancer = ImageEnhance.Brightness(image)
        image = brightness_enhancer.enhance(1.1)

        # Step 8: Encode output as base64 PNG
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        encoded_img = base64.b64encode(buf.getvalue()).decode("utf-8")

        return [ImageContent(type="image", mimeType="image/png", data=encoded_img)]

    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=str(e)))
#----------------------------------------------------------------tool horoscope#
# import os
# import re
# import io
# import base64
# import random
# import sqlite3
# import asyncio
# from typing import Annotated
# import httpx

# # FastMCP + auth
# from fastmcp import FastMCP
# from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
# from mcp.server.auth.provider import AccessToken
# from mcp import ErrorData, McpError
# from mcp.types import Field, TextContent, ImageContent, INTERNAL_ERROR
# VALID_SIGNS = {
#     "aries","taurus","gemini","cancer","leo","virgo","libra","scorpio","sagittarius","capricorn","aquarius","pisces"
# }

# # @mcp.tool(description="Get daily horoscope. Usage: @horoscope sign=<aries> day=<today|yesterday|tomorrow>")
# # async def horoscope(
# #     sign: Annotated[str, Field(description="Zodiac sign e.g. aries")] = None,
# #     day: Annotated[str | None, Field(description="today|yesterday|tomorrow")] = "today",
# # ) -> str:
# #     if not sign:
# #         raise McpError(ErrorData(code=INTERNAL_ERROR, message="Please provide a zodiac sign (e.g. aries)."))
# #     sign_l = sign.strip().lower()
# #     if sign_l not in VALID_SIGNS:
# #         return f"❌ Unknown sign '{sign}'. Valid: {', '.join(sorted(VALID_SIGNS))}"

# #     params = {"sign": sign_l, "day": day or "today"}
# #     try:
# #         async with httpx.AsyncClient() as client:
# #             resp = await client.post("https://aztro.sameerkumar.website/", params=params, timeout=10)
# #             data = resp.json()
# #     except Exception as e:
# #         raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Horoscope fetch failed: {e}"))

# #     # Aztro response fields: description, mood, color, lucky_number, lucky_time, compatibility, current_date
# #     desc = data.get("description", "(no description)")
# #     mood = data.get("mood", "")
# #     color = data.get("color", "")
# #     lucky_num = data.get("lucky_number", "")
# #     lucky_time = data.get("lucky_time", "")
# #     date = data.get("current_date", "")

# #     return (f"🔮 Horoscope — {sign_l.capitalize()} — {date}\n\n"
# #             f"{desc}\n\n"
# #             f"• Mood: {mood}\n• Color: {color}\n• Lucky number: {lucky_num}\n• Lucky time: {lucky_time}")
# # # import httpx
# # # from mcp import McpError, ErrorData, INTERNAL_ERROR
# # # from mcp.types import Field

# # # VALID_SIGNS = {
# # #     "aries","taurus","gemini","cancer","leo","virgo","libra","scorpio","sagittarius","capricorn","aquarius","pisces"
# # # }

# # # @mcp.tool(description="Get daily horoscope. Usage: @horoscope sign=<aries>")
# # # async def horoscope(
# # #     sign: Annotated[str, Field(description="Zodiac sign e.g. aries")] = None,
# # # ) -> str:
# # #     if not sign:
# # #         raise McpError(ErrorData(code=INTERNAL_ERROR, message="Please provide a zodiac sign (e.g. aries)."))
# # #     sign_l = sign.strip().lower()
# # #     if sign_l not in VALID_SIGNS:
# # #         return f"❌ Unknown sign '{sign}'. Valid: {', '.join(sorted(VALID_SIGNS))}"

# # #     url = f"https://horoscope-api.herokuapp.com/horoscope/today/{sign_l}"
# # #     try:
# # #         async with httpx.AsyncClient() as client:
# # #             resp = await client.get(url, timeout=10)
# # #             resp.raise_for_status()
# # #             data = resp.json()
# # #     except Exception as e:
# # #         raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Horoscope fetch failed: {e}"))

# # #     description = data.get("horoscope", "(no description)")
# # #     date_range = data.get("date_range", "")
# # #     current_date = data.get("current_date", "") or ""

# # #     return (f"🔮 Horoscope — {sign_l.capitalize()} — {current_date}\n"
# # #             f"Date Range: {date_range}\n\n"
# # #             f"{description}")
# @mcp.tool(description="Get daily horoscope. Usage: @horoscope sign=<aries> day=<today|yesterday|tomorrow>")
# async def horoscope(
#     sign: Annotated[str, Field(description="Zodiac sign e.g. aries")] = None,
#     day: Annotated[str | None, Field(description="today|yesterday|tomorrow")] = "today",
# ) -> str:
#     if not sign:
#         raise McpError(ErrorData(code=INTERNAL_ERROR, message="Please provide a zodiac sign (e.g. aries)."))

#     sign_l = sign.strip().lower()
#     if sign_l not in VALID_SIGNS:
#         return f"❌ Unknown sign '{sign}'. Valid: {', '.join(sorted(VALID_SIGNS))}"

#     url = "https://sameer-kumar-aztro-v1.p.rapidapi.com/"
#     params = {"sign": sign_l, "day": day or "today"}
#     headers = {
#         "x-rapidapi-key": "7bd7d59100msha77016cf106a0aap196edejsnabf8fbb51149",  # your RapidAPI key
#         "x-rapidapi-host": "sameer-kumar-aztro-v1.p.rapidapi.com",
#         "Content-Type": "application/json"
#     }

#     try:
#         async with httpx.AsyncClient() as client:
#             resp = await client.post(url, headers=headers, params=params, timeout=10)
#             resp.raise_for_status()
#             data = resp.json()
#     except Exception as e:
#         raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Horoscope fetch failed: {e}"))

#     desc = data.get("description", "(no description)")
#     mood = data.get("mood", "")
#     color = data.get("color", "")
#     lucky_num = data.get("lucky_number", "")
#     lucky_time = data.get("lucky_time", "")
#     date = data.get("current_date", "")

#     return (f"🔮 Horoscope — {sign_l.capitalize()} — {date}\n\n"
#             f"{desc}\n\n"
#             f"• Mood: {mood}\n• Color: {color}\n• Lucky number: {lucky_num}\n• Lucky time: {lucky_time}")

import asyncio
import sqlite3
import httpx
from fastmcp import FastMCP
from mcp.types import Field

# mcp = FastMCP()

# --- DB Setup ---
astro_conn = sqlite3.connect("horoscope.db")
curs = astro_conn.cursor()
curs.execute("""
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sign TEXT,
    day TEXT,
    prediction TEXT
)
""")
astro_conn.commit()

API_URL = "https://aztro.sameerkumar.website"

@mcp.tool()
async def horoscope(
    sign: Annotated[str, Field(description="Zodiac sign (e.g., 'aries', 'leo')")],
    day: Annotated[str, Field(description="'today', 'tomorrow', or 'yesterday'")]
) -> str:
    """
    Get daily horoscope for a given sign and day from Aztro API.
    """
    # try:
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            API_URL,
            params={"sign": sign, "day": day}  # Aztro takes params in POST
        )
        resp.raise_for_status()
        data = resp.json()

    prediction = data.get("description", "No prediction available.")

    # Save to DB
    curs.execute(
        "INSERT INTO history (sign, day, prediction) VALUES (?, ?, ?)",
        (sign, day, prediction)
    )
    astro_conn.commit()

    return f"Horoscope for {sign} ({day}): {prediction}"

    # except httpx.RequestError as e:
    #     return f"Error connecting to Aztro API: {e}"
    # except httpx.HTTPStatusError as e:
    #     return f"API returned error {e.response.status_code}: {e.response.text}"
    # except Exception as e:
    #     return f"Unexpected error: {e}"



# --- Run server ---
async def main():
    print("🚀 Starting MCP server on http://0.0.0.0:8086")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)

if __name__ == "__main__":
    asyncio.run(main())





















