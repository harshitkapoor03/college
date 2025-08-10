import asyncio
import random
import sqlite3
import httpx
from typing import Annotated
from fastmcp import FastMCP
from mcp.types import Field


# # --- DB Setup ---
# conn = sqlite3.connect("leaderboard.db", check_same_thread=False)
# cur = conn.cursor()
# cur.execute("""CREATE TABLE IF NOT EXISTS colleges (
#     college TEXT PRIMARY KEY,
#     total_score INTEGER DEFAULT 0
# )""")
# for c in ["BITS P", "BITS G", "BITS H"]:
#     cur.execute("INSERT OR IGNORE INTO colleges (college) VALUES (?)", (c,))
# conn.commit()

# # --- Active quiz sessions ---
# active_quizzes = {}

# # --- Fetch questions from Open Trivia DB API ---
# async def fetch_questions():
#     print("[DEBUG] fetch_questions() called")
#     questions = []
#     try:
#         async with httpx.AsyncClient() as client:
#             diff, count = "easy", 5
#             print(f"[DEBUG] Fetching {count} '{diff}' questions")
#             resp = await client.get(
#                 "https://opentdb.com/api.php",
#                 params={"amount": count, "difficulty": diff, "type": "multiple"},
#                 timeout=10
#             )
#             print(f"[DEBUG] API status={resp.status_code}")
#             if resp.status_code == 200:
#                 data = resp.json().get("results", [])
#                 print(f"[DEBUG] Received {len(data)} questions")
#                 for item in data:
#                     original_q = item["question"].replace("&quot;", '"').replace("&#039;", "'").replace("&amp;", "&")
#                     q = "Following question is to be provided to the user: " + original_q

#                     correct = item["correct_answer"].replace("&quot;", '"').replace("&#039;", "'").replace("&amp;", "&")

#                     choices = [
#                         ch.replace("&quot;", '"').replace("&#039;", "'").replace("&amp;", "&")
#                         for ch in (item["incorrect_answers"] + [correct])
#                     ]
#                     random.shuffle(choices)

#                     questions.append({"diff": diff.capitalize(), "q": q, "choices": choices, "ans": correct})
#     except Exception as e:
#         print(f"[ERROR] Error fetching trivia: {e}")

#     if len(questions) < 5:
#         print("[DEBUG] Using fallback questions")
#         fallback_questions = [
#             {"diff": "Easy", "q": "Following question is to be provided to the user: What is 2 + 2?",
#              "choices": ["3", "4", "5", "6"], "ans": "4"},
#             {"diff": "Medium", "q": "Following question is to be provided to the user: What is the capital of France?",
#              "choices": ["London", "Berlin", "Paris", "Madrid"], "ans": "Paris"},
#             {"diff": "Hard", "q": "Following question is to be provided to the user: What is the derivative of x²?",
#              "choices": ["x", "2x", "x²", "2"], "ans": "2x"},
#             {"diff": "Hard", "q": "Following question is to be provided to the user: What is the square root of 144?",
#              "choices": ["10", "11", "12", "13"], "ans": "12"},
#             {"diff": "Hard", "q": "Following question is to be provided to the user: What year did World War II end?",
#              "choices": ["1944", "1945", "1946", "1947"], "ans": "1945"},
#         ]
#         questions.extend(fallback_questions[:5 - len(questions)])

#     return questions[:5]

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
@mcp.tool
async def validate() -> str:
    return MY_NUMBER

# # --- Tools ---
# @mcp.tool(description="Show current leaderboard standings.")
# async def show_leaderboard() -> str:
#     try:
#         rows = cur.execute("SELECT college, total_score FROM colleges ORDER BY total_score DESC").fetchall()
#     except Exception as e:
#         return f"⚠️ Database error: {e}"
#     if not rows or not any(score > 0 for _, score in rows):
#         return ("To be shown to user :\n 🏆 College Competition Leaderboard\n"
#                 "No scores yet—be the first!\n"
#                 "Start quiz: @enter_competition college=<BITS P|BITS G|BITS H>")
#     medals = ['🥇', '🥈', '🥉']
#     lines = []
#     for i, (college, score) in enumerate(rows):
#         medal = medals[i] if i < 3 else ""
#         lines.append(f"{medal} {college}: {score} points")
#     return "🏆 College Competition Leaderboard\n" + "\n".join(lines)

# @mcp.tool(description="Start a quiz for a selected college or answer a question in progress.")
# async def enter_competition(
#     college: Annotated[str | None, Field(description="College choice: BITS P, BITS G, or BITS H")] = None,
#     answer: Annotated[int | None, Field(description="Answer number to current question")] = None,
# ) -> str:
#     phone = MY_NUMBER

#     if phone not in active_quizzes and college is None and answer is None:
#         return ("🏁 Main Menu:\n"
#                 "• Start Competition → @enter_competition college=<BITS P|BITS G|BITS H>\n"
#                 "• View Leaderboard → @show_leaderboard")

#     if college:
#         mapping = {"BITS P": "BITS P", "BITS G": "BITS G", "BITS H": "BITS H"}
#         selected_college = mapping.get(college.upper())
#         if not selected_college:
#             return "❌ Invalid choice. Please use BITS P, BITS G, or BITS H."
#         if phone in active_quizzes:
#             del active_quizzes[phone]
#         questions = await fetch_questions()
#         active_quizzes[phone] = {"college": selected_college, "questions": questions, "current": 0}
#         qd = questions[0]
#         opts = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(qd["choices"]))
#         return (f"To be shown to user:\n🎓 Quiz for {selected_college}!\n"
#                 f"Q1 ({qd['diff']}): {qd['q']}\n{opts}\n"
#                 "Reply with @enter_competition answer=<number>")

#     if answer is not None and phone in active_quizzes:
#         session = active_quizzes[phone]

#         # Early check if quiz is over
#         if session["current"] >= len(session["questions"]):
#             del active_quizzes[phone]
#             return "✅ That quiz is already finished! Start a new one with:\n@enter_competition college=<BITS P|BITS G|BITS H>"

#         idx = session["current"]
#         qd = session["questions"][idx]

#         print(f"[DEBUG] User picked: {qd['choices'][answer-1]} | Correct answer: {qd['ans']}")
#         if not (1 <= answer <= len(qd["choices"])):
#             return f"❌ Invalid! Pick a number between 1 and {len(qd['choices'])}."

#         # Case and whitespace insensitive comparison
#         is_correct = qd["choices"][answer - 1].strip().lower() == qd["ans"].strip().lower()
#         feedback = "✅ Correct! +10 pts." if is_correct else f"❌ Wrong. Correct answer: {qd['ans']}"

#         if is_correct:
#             cur.execute("UPDATE colleges SET total_score = total_score + 10 WHERE college = ?", (session["college"],))
#             conn.commit()

#         session["current"] += 1

#         if session["current"] >= len(session["questions"]):
#             college = session["college"]
#             score = cur.execute("SELECT total_score FROM colleges WHERE college = ?", (college,)).fetchone()[0]
#             del active_quizzes[phone]
#             return (f"{feedback}\n\nTo be shown to user:\n🎉 Quiz complete for {college}! Total: {score} points\n\n"
#                     "🏁 Main Menu:\n"
#                     "• Start Competition → @enter_competition college=<BITS P|BITS G|BITS H>\n"
#                     "• View Leaderboard → @show_leaderboard")

#         qd = session["questions"][session["current"]]
#         opts = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(qd["choices"]))
#         return (f"{feedback}\nQ{session['current'] + 1} ({qd['diff']}): {qd['q']}\n{opts}\n"
#                 "Reply with @enter_competition answer=<number>")

#     return "Use @enter_competition college=<BITS P|BITS G|BITS H> to start or @show_leaderboard for rankings."


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

# import asyncio
# import sqlite3
# import httpx
# from fastmcp import FastMCP
# from mcp.types import Field

# # mcp = FastMCP()

# # --- DB Setup ---
# astro_conn = sqlite3.connect("horoscope.db")
# curs = astro_conn.cursor()
# curs.execute("""
# CREATE TABLE IF NOT EXISTS history (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     sign TEXT,
#     day TEXT,
#     prediction TEXT
# )
# """)
# astro_conn.commit()

# API_URL = "https://aztro.sameerkumar.website"

# @mcp.tool()
# async def horoscope(
#     sign: Annotated[str, Field(description="Zodiac sign (e.g., 'aries', 'leo')")],
#     day: Annotated[str, Field(description="'today', 'tomorrow', or 'yesterday'")]
# ) -> str:
#     """
#     Get daily horoscope for a given sign and day from Aztro API.
#     """
#     # try:
#     async with httpx.AsyncClient(timeout=10.0) as client:
#         resp = await client.post(
#             API_URL,
#             params={"sign": sign, "day": day}  # Aztro takes params in POST
#         )
#         resp.raise_for_status()
#         data = resp.json()

#     prediction = data.get("description", "No prediction available.")

#     # Save to DB
#     curs.execute(
#         "INSERT INTO history (sign, day, prediction) VALUES (?, ?, ?)",
#         (sign, day, prediction)
#     )
#     astro_conn.commit()

#     return f"Horoscope for {sign} ({day}): {prediction}"

    # except httpx.RequestError as e:
    #     return f"Error connecting to Aztro API: {e}"
    # except httpx.HTTPStatusError as e:
    #     return f"API returned error {e.response.status_code}: {e.response.text}"
    # except Exception as e:
    #     return f"Unexpected error: {e}"

# API_URL = "https://sameer-kumar-aztro-v1.p.rapidapi.com/"
# API_KEY = "7bd7d59100msha77016cf106a0aap196edejsnabf8fbb51149"  # Replace with your real RapidAPI key

# @mcp.tool(description="Get daily horoscope")
# async def horoscope(
#     sign: Annotated[str, Field(description="Zodiac sign (e.g., 'aries', 'leo')")],
#     day: Annotated[str, Field(description="'today', 'tomorrow', or 'yesterday'")] = "today"
# ) -> str:
#     """
#     Get daily horoscope for a given sign and day from the Aztro API via RapidAPI.
#     """
#     # headers = {
#     #     "x-rapidapi-key": API_KEY,
#     #     "x-rapidapi-host": "sameer-kumar-aztro-v1.p.rapidapi.com",
#     #     "Content-Type": "application/json"
#     # }
#     # params = {"sign": sign, "day": day}

#     # async with httpx.AsyncClient(timeout=10.0) as client:
#     #     resp = await client.post(API_URL, headers=headers, params=params)
#     #     resp.raise_for_status()
#     #     data = resp.json()

#     # return data.get("description", "No prediction available.")
#     headers = {
#         "x-rapidapi-key": API_KEY,
#         "x-rapidapi-host": "sameer-kumar-aztro-v1.p.rapidapi.com",
#         "Content-Type": "application/x-www-form-urlencoded"
#     }
#     data = {
#         "sign": sign.lower(),
#         "day": day.lower()
#     }
#     async with httpx.AsyncClient(timeout=10) as client:
#             resp = await client.post(API_URL, headers=headers, data=data)
#             resp.raise_for_status()
#             return resp.json()
# @mcp.tool(description="Get daily horoscope")
# async def horoscope(
#     sign: Annotated[str, Field(description="Zodiac sign (e.g., 'aries', 'leo')")],
#     lang: Annotated[str, Field(description="Language code (e.g., 'en')")] = "en",
#     type_: Annotated[str, Field(description="Horoscope type (e.g., 'daily')")] = "daily",
#     timezone: Annotated[str, Field(description="Timezone (e.g., 'UTC')")] = "UTC"
# ) -> str:
#     """
#     Get daily horoscope for a given sign from the Astropredict API via RapidAPI.
#     """
#     import httpx
#     import os

#     API_KEY = "7bd7d59100msha77016cf106a0aap196edejsnabf8fbb51149"  # store key in environment variables
#     API_URL = "https://astropredict-daily-horoscopes-lucky-insights.p.rapidapi.com/horoscope"

#     headers = {
#         "x-rapidapi-key": API_KEY,
#         "x-rapidapi-host": "astropredict-daily-horoscopes-lucky-insights.p.rapidapi.com"
#     }
#     params = {
#         "lang": lang,
#         "zodiac": sign.lower(),
#         "type": type_,
#         "timezone": timezone
#     }
#     data="uunavailable rn"

#     async with httpx.AsyncClient(timeout=10.0) as client:
#         resp = await client.get(API_URL, headers=headers, params=params)
#         resp.raise_for_status()
#         data = resp.json()

#     # The API returns multiple fields — adjust as needed
#     # Assuming the main horoscope text is in "prediction" or similar
#     return data.get("horoscope", "No prediction available.")
@mcp.tool(description="Get daily horoscope")
async def horoscope(
    sign: Annotated[str, Field(description="Zodiac sign (e.g., 'aries', 'leo')")],
    #lang: Annotated[str, Field(description="Language code (e.g., 'en')")] = "en",
    # day: Annotated[str, Field(description="Horoscope type (e.g., 'daily')")] = "daily",
    # timezone: Annotated[str, Field(description="Timezone (e.g., 'UTC')")] = "UTC"
) -> str:
    """
    Get daily horoscope for a given sign from the Astropredict API via RapidAPI.
    """
    import httpx
    import os

    API_KEY = "7bd7d59100msha77016cf106a0aap196edejsnabf8fbb51149"  # store key in environment variables
    API_URL = "https://horoscope-daily-api2.p.rapidapi.com/get-horoscope"

    headers = {
        "x-rapidapi-key": API_KEY,
        "x-rapidapi-host": "horoscope-daily-api2.p.rapidapi.com"
    }
    params = {
        "zodiacSign":sign,"day":"today"
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(API_URL, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()

    # The API returns multiple fields — adjust as needed
    # Assuming the main horoscope text is in "prediction" or similar
    # return data.get("prediction", "No prediction available.")
    return data.get("horoscope", "No prediction available.")
#=---------------------------------------------------------------------------------------------------------compt tool new 
import asyncio
import os
import json
import uuid
from typing import Annotated, Optional, Literal
from datetime import datetime
from dotenv import dotenv_values
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import TextContent, INVALID_PARAMS, INTERNAL_ERROR
from pydantic import BaseModel, Field
import requests

class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None

# # Global user data storage
# USER_DATA: dict[str, dict] = {}
# COLLEGE_SCORES: dict[str, int] = {}

def _get_user_data(puch_user_id: str) -> dict:
    if not puch_user_id:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="puch_user_id is required"))
    return USER_DATA.setdefault(puch_user_id, {})

def _error(code, msg):
    raise McpError(ErrorData(code=code, message=msg))

# Fetch questions from Open Trivia Database
FetchQuestionsDescription = RichToolDescription(
    description="Fetches one exciting trivia multiple-choice question for the user from the Open Trivia Database!",
    use_when="When a new quiz session is started or user asks for trivia questions or to enter a quiz.",
    side_effects="None",
)

@mcp.tool(description=FetchQuestionsDescription.model_dump_json())
async def fetch_trivia(
    puch_user_id: Annotated[str, Field(description="Puch User Unique Identifier")]
) -> str:
    try:
        response = requests.get("https://opentdb.com/api.php?amount=1&difficulty=medium&type=multiple")
        response.raise_for_status()
        questions = response.json().get("results", [])
        first_q = questions[0]
        cans = first_q["correct_answer"]
        q = first_q["question"]
        # cans=questions["correct_answer"]
        # q=questions["question"]
        # user_data = _get_user_data(puch_user_id)
        # user_data['questions'] = questions
        # user_data['current_question'] = 0
        # user_data['score'] = 0
        return f"{q} The correct answer to this questionto be judged by the puchai client after the user replies is {cans} make sure you dont see any other character and text back that the user is correct"
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=str(e)))

# # Answer question tool
# AnswerQuestionDescription = RichToolDescription(
#     description="Submits an answer for the current question.",
#     use_when="When a user answers a question with one of the options provided in the question asked most recently.",
#     side_effects="Updates the user's score.",
# )

# @mcp.tool(description=AnswerQuestionDescription.model_dump_json())
# async def answer_question(
#     puch_user_id: Annotated[str, Field(description="Puch User Unique Identifier")],
#     answer: Annotated[int, Field(description="Answer of the latest question a string if option d was choses and it was d.june the answer will be june and will be stored in this variableo")]
# ) -> list[TextContent]:
#     try:
#         user_data = _get_user_data(puch_user_id)
#         questions = user_data.get('questions', [])
#         current_question = user_data.get('current_question', 1)
        
#         if current_question >= len(questions):
#             return [TextContent(type="text", text="No more questions available.")]
        
#         correct_answer = questions[current_question-1]['correct_answer']
#         answer=answer.lower()
#         correct_answer=correct_answer.lower()
#         if answer== correct_answer:
#             user_data['score'] += 10
            
        
#         user_data['current_question'] += 1
#         # Check if quiz is complete
#         if user_data['current_question'] >= len(questions):
#             college = user_data.get('college')
#             if college:
#                 COLLEGE_SCORES[college] = COLLEGE_SCORES.get(college, 0) + user_data['score']
#                 return [TextContent(type="text", text=f"Quiz complete! Final score: {user_data['score']}. Added to {college} total.")]
#             else:
#                 return [TextContent(type="text", text=f"Your current score is {user_data['score']}. Enter your college name when you enter the quiz next time if you want to contribute to it winning!")]

#         return [TextContent(type="text", text=f"Your current score is {user_data['score']}")]
#     except Exception as e:
#         raise McpError(ErrorData(code=INTERNAL_ERROR, message=str(e)))

# # Submit college tool
# SubmitCollegeDescription = RichToolDescription(
#     description="Submits the user's college for the competition.",
#     use_when="When a user starts the quiz before starting to fetch the questions run this and store the users college before fetching questions for user.",
#     side_effects="Associates the user with a college.",
# )

# @mcp.tool(description=SubmitCollegeDescription.model_dump_json())
# async def submit_college(
#     puch_user_id: Annotated[str, Field(description="Puch User Unique Identifier")],
#     college_name: Annotated[str, Field(description="Name of the college")]
# ) -> list[TextContent]:
#     try:
#         user_data = _get_user_data(puch_user_id)
#         user_data['college'] = college_name
#         return [TextContent(type="text", text=f"College {college_name} registered successfully.")]
#     except Exception as e:
#         raise McpError(ErrorData(code=INTERNAL_ERROR, message=str(e)))

# # Leaderboard tool
# LeaderboardDescription = RichToolDescription(
#     description="Displays the current leaderboard with college scores.",
#     use_when="When a user requests to see the leaderboard.",
#     side_effects="None",
# )

# @mcp.tool(description=LeaderboardDescription.model_dump_json())
# async def leaderboard(
#     puch_user_id: Annotated[str, Field(description="Puch User Unique Identifier")]
# ) -> list[TextContent]:
#     try:
#         leaderboard_data = sorted(COLLEGE_SCORES.items(), key=lambda x: x[1], reverse=True)
#         return [TextContent(type="text", text=json.dumps(leaderboard_data))]
#     except Exception as e:
#         raise McpError(ErrorData(code=INTERNAL_ERROR, message=str(e)))

#------------------------------------------------------------------$------------------------------------------------------------------------------Resume tool-----------------
import requests
import time
import datetime
import matplotlib.pyplot as plt
import base64
from typing import Annotated
from mcp.types import TextContent, ImageContent
from pydantic import Field

# @mcp.tool(description="Fetch last 10 hours of price data for a crypto token and generate a chart.")
# async def crypto_last_10h(
#     token_id: Annotated[str, Field(description="The CoinGecko token ID (e.g., 'bitcoin', 'dogecoin')")],
#     currency: Annotated[str, Field(description="Currency for prices, e.g., 'usd', 'inr")] = "usd"
# ) -> list:
#     try:
#         now = int(time.time())
#         ten_hours_ago = now - (10 * 60 * 60)

#         url = f"https://api.coingecko.com/api/v3/coins/{token_id}/market_chart/range"
#         params = {
#             "vs_currency": currency,
#             "from": ten_hours_ago,
#             "to": now
#         }

#         res = requests.get(url, params=params)
#         res.raise_for_status()
#         data = res.json()

#         timestamps = [p[0] / 1000 for p in data["prices"]]
#         prices = [p[1] for p in data["prices"]]

#         offset = datetime.timedelta(hours=5, minutes=30)
#         times = [datetime.datetime.fromtimestamp(ts) + offset for ts in timestamps]

#         hourly_times, hourly_prices, seen_hours = [], [], set()
#         for t, price in zip(times, prices):
#             hour_key = t.replace(minute=0, second=0, microsecond=0)
#             if hour_key not in seen_hours:
#                 seen_hours.add(hour_key)
#                 hourly_times.append(hour_key.strftime("%I:%M %p"))
#                 hourly_prices.append(price)

#         if len(hourly_prices) >= 2:
#             start_price = hourly_prices[0]
#             end_price = hourly_prices[-1]
#             percent_change = ((end_price - start_price) / start_price) * 100
#         else:
#             percent_change = 0

#         # Save chart as PNG
#         chart_path = f"/tmp/{token_id}_last_10h.png"
#         plt.plot(hourly_times, hourly_prices, marker='o', color='blue')
#         plt.xticks(rotation=45)
#         plt.title(f"{token_id.capitalize()} Price (Last 10 Hours)\nChange: {percent_change:+.2f}%")
#         plt.xlabel("Time (IST)")
#         plt.ylabel(f"Price ({currency.upper()})")
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig(chart_path)
#         plt.close()

#         # Encode PNG to base64
#         with open(chart_path, "rb") as f:
#             img_b64 = base64.b64encode(f.read()).decode("utf-8")

#         # Build text table
#         table_text = "\nHourly Prices (Last 10 Hours):\n"
#         for t_str, price in zip(hourly_times, hourly_prices):
#             table_text += f"{t_str}  ->  {currency.upper()} {price:,.2f}\n"
#         table_text += f"\nPercentage Change: {percent_change:+.2f}%"

#         return [
#             TextContent(type="text", text=table_text),
#             ImageContent(type="image", mimeType="image/png", data=img_b64)
#         ]

#     except Exception as e:
#         raise McpError(ErrorData(code=INTERNAL_ERROR, message=str(e)))
import requests
import datetime
import time
import io
import base64
import matplotlib.pyplot as plt
from typing import Annotated
from fastmcp import  TextContent, ImageContent, McpError, ErrorData, INTERNAL_ERROR

@mcp.tool(description="Fetch historical cryptocurrency prices from CoinGecko for a given coin, currency, time period, and interval; returns a formatted price table, percentage change, and an IST-based time series chart.")
async def crypto_time_series(
    coin_id: Annotated[str, Field(description="CoinGecko coin ID, e.g., 'bitcoin', 'ethereum', 'solana'")],
    currency: Annotated[str, Field(description="Target currency, e.g., 'usd', 'inr'")] = "inr",
    period_hours: Annotated[int, Field(description="How many past hours to fetch")] = 10,
    interval: Annotated[str, Field(description="Data interval: 'minute' or 'hour'")] = "hour",
) -> list:
    """
    Returns:
      - TextContent: a formatted table of timestamps (IST) and prices + % change
      - ImageContent: PNG chart encoded base64
    """
    try:
        # Calculate UNIX timestamps
        now = int(time.time())
        start_time = now - ((period_hours - 1) * 60 * 60)

        # Fetch from CoinGecko
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"
        params = {
            "vs_currency": currency,
            "from": start_time,
            "to": now
        }

        res = requests.get(url, params=params)
        if res.status_code != 200:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Error fetching data: {res.status_code}"))

        data = res.json()
        if "prices" not in data or not data["prices"]:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message="No price data returned from CoinGecko"))

        # Extract data
        timestamps = [p[0] / 1000 for p in data["prices"]]  # ms → s
        prices = [p[1] for p in data["prices"]]

        # Add IST offset
        ist_offset = datetime.timedelta(hours=5, minutes=30)
        times = [datetime.datetime.fromtimestamp(ts) + ist_offset for ts in timestamps]

        # Filter if hourly
        if interval == "hour":
            filtered_times, filtered_prices = [], []
            seen_hours = set()
            for t, price in zip(times, prices):
                hour_key = t.replace(minute=0, second=0, microsecond=0)
                if hour_key not in seen_hours:
                    seen_hours.add(hour_key)
                    filtered_times.append(hour_key.strftime("%d-%b %I:%M %p"))
                    filtered_prices.append(price)
        else:
            filtered_times = [t.strftime("%d-%b %I:%M %p") for t in times]
            filtered_prices = prices

        # Compute percent change
        start_price = filtered_prices[0]
        end_price = filtered_prices[-1]
        percent_change = ((end_price - start_price) / start_price) * 100

        # X-axis ticks
        max_labels = 20
        n = len(filtered_times)
        if n > max_labels:
            step = max(1, n // max_labels)
            x_ticks = list(range(0, n, step))
            if x_ticks[-1] != n - 1:
                x_ticks.append(n - 1)
        else:
            x_ticks = list(range(n))

        # Plot chart
        plt.figure(figsize=(10, 5))
        color = "green" if percent_change >= 0 else "red"
        plt.plot(filtered_times, filtered_prices, marker='o', color=color, linewidth=1.5, markersize=4)
        plt.xticks(x_ticks, [filtered_times[i] for i in x_ticks], rotation=45, ha='right')
        plt.title(f"{coin_id.capitalize()} Price\nPeriod: {period_hours}h | Interval: {interval} | Change: {percent_change:+.2f}%",
                  fontsize=12, fontweight='bold')
        plt.xlabel("Time (IST)")
        plt.ylabel(f"Price ({currency.upper()})")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()

        # Encode PNG
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=200, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")

        # Build text table
        table_lines = [
            f"{coin_id.capitalize()} Prices — Period: {period_hours}h, Interval: {interval}",
            "=" * 60
        ]
        for t, p in zip(filtered_times, filtered_prices):
            table_lines.append(f"{t}  ->  {currency.upper()} {p:,.2f}")
        table_lines.append("=" * 60)
        table_lines.append(f"Percentage Change: {percent_change:+.2f}%")
        table_lines.append(f"Total Data Points: {len(filtered_prices)}")

        table_text = "\n".join(table_lines)

        return [
            TextContent(type="text", text=table_text),
            ImageContent(type="image", mimeType="image/png", data=img_b64)
        ]

    except McpError:
        raise
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=str(e)))

#####______________________________________________________________________________________________________________________________________________

import base64
import io
import datetime
from typing import Annotated

import yfinance as yf
import matplotlib.pyplot as plt

# you may already have these in your environment from fastmcp/mcp types
# from fastmcp import mcp
# from mcp.types import Field, TextContent, ImageContent, ErrorData, McpError, INTERNAL_ERROR

@mcp.tool(description="Fetch historical stock prices for a specified symbol, exchange, period, and interval; returns a formatted price table, percentage change, and an IST-based time series chart.\n You can include stock symbols like SBIN (State Bank of India),INFY (Infosys Limited) and NSE/BSE for extra accurate results.")
async def stock_time_series(
    stock_symbol: Annotated[str, Field(description="Stock symbol without exchange, e.g. RELIANCE,360ONE, 3MINDIA, ABB, ACC, AIAENG, APLAPOLLO, AUBANK, AARTIIND, AAVAS, ABBOTINDIA,ACE, ADANIENSOL, ADANIENT, ADANIGREEN, ADANIPORTS, ADANIPOWER, ATGL, AWL, ABCAPITAL, ABFRL,AEGISLOG, AETHER, AFFLE, AJANTPHARM, APLLTD, ALKEM, ALKYLAMINE, ALLCARGO, ALOKINDS, ARE&M,AMBER, AMBUJACEM, ANANDRATHI, ANGELONE, ANURAS, APARINDS, APOLLOHOSP, APOLLOTYRE, APTUS, ACI,ASAHIINDIA, ASHOKLEY, ASIANPAINT, ASTERDM, ASTRAZEN, ASTRAL, ATUL, AUROPHARMA, AVANTIFEED, DMART,AXISBANK, BEML, BLS, BSE, BAJAJ-AUTO, BAJFINANCE, BAJAJFINSV, BAJAJHLDNG, BALAMINES, BALKRISIND,BALRAMCHIN, BANDHANBNK, BANKBARODA, BANKINDIA, MAHABANK, BATAINDIA, BAYERCROP, BERGEPAINT, BDL, BEL,BHARATFORG, BHEL, BPCL, BHARTIARTL, BIKAJI, BIOCON, BIRLACORPN, BSOFT, BLUEDART, BLUESTARCO,BBTC, BORORENEW, BOSCHLTD, BRIGADE, BRITANNIA, MAPMYINDIA, CCL, CESC, CGPOWER, CIEINDIA,CRISIL, CSBBANK, CAMPUS, CANFINHOME, CANBK, CAPLIPOINT, CGCL, CARBORUNIV, CASTROLIND, CEATLTD,CELLO, CENTRALBK, CDSL, CENTURYPLY, ABREL, CERA, CHALET, CHAMBLFERT, CHEMPLASTS, CHENNPETRO,CHOLAHLDNG, CHOLAFIN, CIPLA, CUB, CLEAN, COALINDIA, COCHINSHIP, COFORGE, COLPAL, CAMS,CONCORDBIO, CONCOR, COROMANDEL, CRAFTSMAN, CREDITACC, CROMPTON, CUMMINSIND, CYIENT, DCMSHRIRAM, DLF,DOMS, DABUR, DALBHARAT, DATAPATTNS, DEEPAKFERT, DEEPAKNTR, DELHIVERY, DEVYANI, DIVISLAB, DIXON,LALPATHLAB, DRREDDY, EIDPARRY, EIHOTEL, EPL, EASEMYTRIP, EICHERMOT, ELECON, ELGIEQUIP, EMAMILTD,ENDURANCE, ENGINERSIN, EQUITASBNK, ERIS, ESCORTS, EXIDEIND, FDC, NYKAA, FEDERALBNK, FACT,FINEORG, FINCABLES, FINPIPE, FSL, FIVESTAR, FORTIS, GAIL, GMMPFAUDLR, GMRINFRASTRUCT, GRSE,GICRE, GILLETTE, GLAND, GLAXO, ALIVUS, GLENMARK, MEDANTA, GPIL, GODFRYPHLP, GODREJCP,GODREJIND, GODREJPROP, GRANULES, GRAPHITE, GRASIM, GESHIP, GRINDWELL, GAEL, FLUOROCHEM, GUJGASLTD,GMDCLTD, GNFC, GPPL, GSFC, GSPL, HEG, HBLENGINE, HCLTECH, HDFCAMC, HDFCBANK,HDFCLIFE, HFCL, HAPPSTMNDS, HAPPYFORGE, HAVELLS, HEROMOTOCO, HSCL, HINDALCO, HAL, HINDCOPPER,HINDPETRO, HINDUNILVR, HINDZINC, POWERINDIA, HOMEFIRST, HONASA, HONAUT, HUDCO, ICICIBANK, ICICIGI,ICICIPRULI, ISEC, IDBI, IDFCFIRSTB, IFCI, IIFL, IRB, IRCON, ITC, ITI,INDIACEM, INDIAMART, INDIANB, IEX, INDHOTEL, IOC, IOB, IRCTC, IRFC, INDIGOPNTS,IGL, INDUSTOWER, INDUSINDBK, NAUKRI, INFY, INOXWIND, INTELLECT, INDIGO, IPCALAB, JBCHEPHARM,JKCEMENT, JBMA, JKLAKSHMI, JKPAPER, JMFINANCIL, JSWENERGY, JSWINFRA, JSWSTEEL, JAIBALAJI, J&KBANK,JINDALSAW, JSL, JINDALSTEL, JIOFIN, JUBLFOOD, JUBLINGREA, JUBLPHARMA, JWL, JUSTDIAL, JYOTHYLAB,KPRMILL, KEI, KNRCON, KPITTECH, KRBL, KSB, KAJARIACER, KPIL, KALYANKJIL, KANSAINER,KARURVYSYA, KAYNES, KEC, KFINTECH, KOTAKBANK, KIMS, LTF, LTTS, LICHSGFIN, LTIM,LT, LATENTVIEW, LAURUSLABS, LXCHEM, LEMONTREE, LICI, LINDEINDIA, LLOYDSME, LUPIN, MMTC,MRF, MTARTECH, LODHA, MGL, MAHSEAMLES, M&MFIN, M&M, MHRIL, MAHLIFE, MANAPPURAM,MRPL, MANKIND, MARICO, MARUTI, MASTEK, MFSL, MAXHEALTH, MAZDOCK, MEDPLUS, METROBRAND,METROPOLIS, MINDACORP, MSUMI, MOTILALOFS, MPHASIS, MCX, MUTHOOTFIN, NATCOPHARM, NBCC, NCC,NHPC, NLCINDIA, NMDC, NSLNISP, NTPC, NH, NATIONALUM, NAVINFLUOR, NESTLEIND, NETWORK18,NAM-INDIA, NUVAMA, NUVOCO, OBEROIRLTY, ONGC, OIL, OLECTRA, PAYTM, OFSS, POLICYBZR,PCBL, PIIND, PNBHOUSING, PNCINFRA, PVRINOX, PAGEIND, PATANJALI, PERSISTENT, PETRONET, PHOENIXLTD,PIDILITIND, PEL, PPLPHARMA, POLYMED, POLYCAB, POONAWALLA, PFC, POWERGRID, PRAJIND, PRESTIGE,PRINCEPIPE, PRSMJOHNSN, PGHH, PNB, QUESS, RRKABEL, RBLBANK, RECLTD, RHIM, RITES,RADICO, RVNL, RAILTEL, RAINBOW, RAJESHEXPO, RKFORGE, RCF, RATNAMANI, RTNINDIA, RAYMOND,REDINGTON, RELIANCE, RBA, ROUTE, SBFC, SBICARD, SBILIFE, SJVN, SKFINDIA, SRF,SAFARI, SAMMAANCAP, MOTHERSON, SANOFI, SAPPHIRE, SAREGAMA, SCHAEFFLER, SCHNEIDER, SHREECEM, RENUKA,SHRIRAMFIN, SHYAMMETL, SIEMENS, SIGNATURE, SOBHA, SOLARINDS, SONACOMS, SONATSOFTW, STARHEALTH, SBIN,SAIL, SWSOLAR, STLTECH, SUMICHEM, SPARC, SUNPHARMA, SUNTV, SUNDARMFIN, SUNDRMFAST, SUNTECK,SUPREMEIND, SUVENPHAR, SUZLON, SWANENERGY, SYNGENE, SYRMA, TBOTEK, TVSMOTOR, TVSSCS, TMB,TANLA, TATACHEM, TATACOMM, TCS, TATACONSUM, TATAELXSI, TATAINVEST, TATAMOTORS, TATAPOWER, TATASTEEL,TATATECH, TTML, TECHM, TEJASNET, NIACL, RAMCOCEM, THERMAX, TIMKEN, TITAGARH, TITAN,TORNTPHARM, TORNTPOWER, TRENT, TRIDENT, TRIVENI, TRITURBINE, TIINDIA, UCOBANK, UNOMINDA, UPL,UTIAMC, UJJIVANSFB, ULTRACEMCO, UNIONBANK, UBL, UNITDSPR, USHAMART, VGUARD, VIPIND, VAIBHAVGBL,VTL, VARROC, VBL, MANYAVAR, VEDL, VIJAYA, IDEA, VOLTAS, WELCORP, WELSPUNLIV,WESTLIFE, WHIRLPOOL, WIPRO, YESBANK, ZFCVINDIA, ZEEL, ZENSARTECH, ETERNAL, ZYDUSLIFE, ECLERX")],
    exchange: Annotated[str, Field(description="Exchange suffix, e.g. 'NS' for NSE or 'BO' for BSE")] = "NS",
    period: Annotated[str, Field(description="Data period, e.g. '1d','5d','60d','1mo','3mo'")] = "2d",
    interval: Annotated[str, Field(description="Data interval, e.g. '1m','5m','15m','30m','1h','1d'")] = "1h",
    # currency: Annotated[str, Field(description="Currency label for display, e.g. 'INR' or 'USD'")] = "INR",
) -> list:
    """
    Returns:
      - TextContent: a formatted table of timestamps (IST) and Close prices + % change
      - ImageContent: PNG chart encoded base64
    """
    currency="INR"
    try:
        # Build ticker symbol
        full_symbol = f"{stock_symbol}.{exchange}"
        
        # Fetch data
        ticker = yf.Ticker(full_symbol)
        data = ticker.history(period=period, interval=interval)

        if data.empty:
            raise McpError(ErrorData(
                code=INTERNAL_ERROR,
                message=f"No data found for {full_symbol} with period={period}, interval={interval}"
            ))

        # Prepare times (convert to IST by adding offset)
        ist_offset = datetime.timedelta(hours=5, minutes=30)
        # Some indexes are tz-aware; convert to naive UTC first then add offset safely
      
        
        times_dt = [idx.to_pydatetime() for idx in data.index]
        
        # Format times for display
        times_str = [dt.strftime("%d-%b %I:%M %p") for dt in times_dt]


        # Prices (use Close)
        prices = data['Close'].tolist()

        # Format times for display
        times_str = [dt.strftime("%d-%b %I:%M %p") for dt in times_dt]

        # Compute percent change (first -> last)
        if len(prices) >= 2:
            start_price = prices[0]
            end_price = prices[-1]
            percent_change = ((end_price - start_price) / start_price) * 100
        else:
            percent_change = 0.0

        # Prepare x-axis ticks up to 20 labels evenly spaced
        max_labels = 20
        n = len(times_str)
        if n > max_labels:
            step = max(1, n // max_labels)
            x_ticks = list(range(0, n, step))
            # ensure last tick is included
            if x_ticks[-1] != n - 1:
                x_ticks.append(n - 1)
        else:
            x_ticks = list(range(n))

        # Plot chart (only plotting the available points)
        plt.figure(figsize=(10, 5))
        color = "green" if percent_change >= 0 else "red"
        plt.plot(times_str, prices, marker='o', color=color, linewidth=1.5, markersize=4)
        plt.xticks(x_ticks, [times_str[i] for i in x_ticks], rotation=45, ha='right')
        plt.title(f"{stock_symbol.upper()} ({exchange}) Price\nPeriod: {period} | Interval: {interval} | Change: {percent_change:+.2f}%")
        plt.xlabel("Time (IST)")
        plt.ylabel(f"Price ({currency.upper()})")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()

        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=200, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")

        # Build text table (optionally limit rows if huge — here we include all)
        table_lines = [
            f"{stock_symbol.upper()} ({exchange}) Prices — Period: {period}, Interval: {interval}",
            "=" * 60
        ]
        for t, p in zip(times_str, prices):
            table_lines.append(f"{t}  ->  {currency.upper()} {p:,.2f}")
        table_lines.append("=" * 60)
        table_lines.append(f"Percentage Change: {percent_change:+.2f}%")
        table_lines.append(f"Total Data Points: {len(prices)}")

        # Optional company info (best-effort)
        try:
            info = ticker.info
            if info:
                long_name = info.get("longName") or info.get("shortName")
                if long_name:
                    table_lines.append("")
                    table_lines.append(f"Company: {long_name}")
                sector = info.get("sector")
                if sector:
                    table_lines.append(f"Sector: {sector}")
                market_cap = info.get("marketCap")
                if market_cap:
                    table_lines.append(f"Market Cap: {market_cap:,}")
        except Exception:
            # ignore metadata errors
            pass

        table_text = "\n".join(table_lines)

        return [
            TextContent(type="text", text=table_text),
            ImageContent(type="image", mimeType="image/png", data=img_b64)
        ]

    except McpError:
        # re-raise MCP errors as-is
        raise
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=str(e)))

# --- Run server ---
async def main():
    print("🚀 Starting MCP server on http://0.0.0.0:8086")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)

if __name__ == "__main__":
    asyncio.run(main())

































































