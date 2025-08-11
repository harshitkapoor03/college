import asyncio
import random
import sqlite3
import httpx
from typing import Annotated
from fastmcp import FastMCP
from mcp.types import Field



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


from mcp.types import TextContent, ImageContent, INTERNAL_ERROR
from mcp import ErrorData, McpError
from pydantic import Field
import base64
import io
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np





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


def _get_user_data(puch_user_id: str) -> dict:
    if not puch_user_id:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="puch_user_id is required"))
    return USER_DATA.setdefault(puch_user_id, {})

def _error(code, msg):
    raise McpError(ErrorData(code=code, message=msg))

# # Fetch questions from Open Trivia Database
# FetchQuestionsDescription = RichToolDescription(
#     description="Fetches one exciting trivia multiple-choice question for the user from the Open Trivia Database!",
#     use_when="When a new quiz session is started or user asks for trivia questions or to enter a quiz.",
#     side_effects="None",
# )

# @mcp.tool(description=FetchQuestionsDescription.model_dump_json())
# async def fetch_trivia(
#     puch_user_id: Annotated[str, Field(description="Puch User Unique Identifier")]
# ) -> str:
#     try:
#         response = requests.get("https://opentdb.com/api.php?amount=1&difficulty=medium&type=multiple")
#         response.raise_for_status()
#         questions = response.json().get("results", [])
#         first_q = questions[0]
#         cans = first_q["correct_answer"]
#         q = first_q["question"]
#         # cans=questions["correct_answer"]
#         # q=questions["question"]
#         # user_data = _get_user_data(puch_user_id)
#         # user_data['questions'] = questions
#         # user_data['current_question'] = 0
#         # user_data['score'] = 0
#         return f"{q} The correct answer to this questionto be judged by the puchai client after the user replies is {cans} make sure you dont see any other character and text back that the user is correct"
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

import requests
import datetime
import time
import io
import base64
import matplotlib.pyplot as plt
from typing import Annotated



import base64
import io
import time
import datetime
from typing import Annotated
import requests
import matplotlib.pyplot as plt

# from fastmcp import mcp
# from mcp.types import Field, TextContent, ImageContent, ErrorData, McpError, INTERNAL_ERROR

@mcp.tool(description="-Fetch historical crypto prices from CoinGecko by coin, currency, period, and interval (minute/hour) \neg) Bitcoin price in INR for last 10 hours")
async def crypto_time_series( 
    coin_id: Annotated[str, Field(description="CoinGecko coin ID, e.g., 'bitcoin', 'ethereum', 'solana'")],
    currency: Annotated[str, Field(description="Target currency, e.g., 'usd', 'inr'")] = "inr",
    period_hours: Annotated[int, Field(description="How many past hours to fetch,convert any input time duration to hours")] = 10,
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
        start_time = now - ((period_hours ) * 60 * 60)

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
        last_actual_time = times[-1].strftime("%d-%b %I:%M %p")
        last_actual_price = prices[-1]
        
        if last_actual_time != filtered_times[-1]:
            filtered_times.append(last_actual_time)
            filtered_prices.append(last_actual_price)

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

        # Build text table (formatted like stock_time_series)
        table_lines = [
            f"{coin_id.capitalize()} Prices — Period: {period_hours}h, Interval: {interval}",
            "=" * 60
        ]
        for t, p in zip(filtered_times, filtered_prices):
            table_lines.append(f"{t:<20} ->  {currency.upper()} {p:,.4f}")
        table_lines.append("=" * 60)
        table_lines.append(f"Percentage Change: {percent_change:+.4f}%")
        table_lines.append(f"Total Data Points: {len(filtered_prices)}")
        # # Add latest price row
        # latest_time = filtered_times[-1]
        # latest_price = filtered_prices[-1]
        # table_lines.append(f"Latest Price ({latest_time}): {currency.upper()} {latest_price:,.4f}")

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


@mcp.tool(description="-Fetch historical stock prices by symbol, exchange, period, and time interval\neg) SBIN stock price in NSE for last 3 days with time interval of 1 hour")
async def stock_time_series(
    stock_symbol: Annotated[str, Field(description="Stock symbol without exchange, e.g. RELIANCE,360ONE, 3MINDIA, ABB, ACC, AIAENG, APLAPOLLO, AUBANK, AARTIIND, AAVAS, ABBOTINDIA,ACE, ADANIENSOL, ADANIENT, ADANIGREEN, ADANIPORTS, ADANIPOWER, ATGL, AWL, ABCAPITAL, ABFRL,AEGISLOG, AETHER, AFFLE, AJANTPHARM, APLLTD, ALKEM, ALKYLAMINE, ALLCARGO, ALOKINDS, ARE&M,AMBER, AMBUJACEM, ANANDRATHI, ANGELONE, ANURAS, APARINDS, APOLLOHOSP, APOLLOTYRE, APTUS, ACI,ASAHIINDIA, ASHOKLEY, ASIANPAINT, ASTERDM, ASTRAZEN, ASTRAL, ATUL, AUROPHARMA, AVANTIFEED, DMART,AXISBANK, BEML, BLS, BSE, BAJAJ-AUTO, BAJFINANCE, BAJAJFINSV, BAJAJHLDNG, BALAMINES, BALKRISIND,BALRAMCHIN, BANDHANBNK, BANKBARODA, BANKINDIA, MAHABANK, BATAINDIA, BAYERCROP, BERGEPAINT, BDL, BEL,BHARATFORG, BHEL, BPCL, BHARTIARTL, BIKAJI, BIOCON, BIRLACORPN, BSOFT, BLUEDART, BLUESTARCO,BBTC, BORORENEW, BOSCHLTD, BRIGADE, BRITANNIA, MAPMYINDIA, CCL, CESC, CGPOWER, CIEINDIA,CRISIL, CSBBANK, CAMPUS, CANFINHOME, CANBK, CAPLIPOINT, CGCL, CARBORUNIV, CASTROLIND, CEATLTD,CELLO, CENTRALBK, CDSL, CENTURYPLY, ABREL, CERA, CHALET, CHAMBLFERT, CHEMPLASTS, CHENNPETRO,CHOLAHLDNG, CHOLAFIN, CIPLA, CUB, CLEAN, COALINDIA, COCHINSHIP, COFORGE, COLPAL, CAMS,CONCORDBIO, CONCOR, COROMANDEL, CRAFTSMAN, CREDITACC, CROMPTON, CUMMINSIND, CYIENT, DCMSHRIRAM, DLF,DOMS, DABUR, DALBHARAT, DATAPATTNS, DEEPAKFERT, DEEPAKNTR, DELHIVERY, DEVYANI, DIVISLAB, DIXON,LALPATHLAB, DRREDDY, EIDPARRY, EIHOTEL, EPL, EASEMYTRIP, EICHERMOT, ELECON, ELGIEQUIP, EMAMILTD,ENDURANCE, ENGINERSIN, EQUITASBNK, ERIS, ESCORTS, EXIDEIND, FDC, NYKAA, FEDERALBNK, FACT,FINEORG, FINCABLES, FINPIPE, FSL, FIVESTAR, FORTIS, GAIL, GMMPFAUDLR, GMRINFRASTRUCT, GRSE,GICRE, GILLETTE, GLAND, GLAXO, ALIVUS, GLENMARK, MEDANTA, GPIL, GODFRYPHLP, GODREJCP,GODREJIND, GODREJPROP, GRANULES, GRAPHITE, GRASIM, GESHIP, GRINDWELL, GAEL, FLUOROCHEM, GUJGASLTD,GMDCLTD, GNFC, GPPL, GSFC, GSPL, HEG, HBLENGINE, HCLTECH, HDFCAMC, HDFCBANK,HDFCLIFE, HFCL, HAPPSTMNDS, HAPPYFORGE, HAVELLS, HEROMOTOCO, HSCL, HINDALCO, HAL, HINDCOPPER,HINDPETRO, HINDUNILVR, HINDZINC, POWERINDIA, HOMEFIRST, HONASA, HONAUT, HUDCO, ICICIBANK, ICICIGI,ICICIPRULI, ISEC, IDBI, IDFCFIRSTB, IFCI, IIFL, IRB, IRCON, ITC, ITI,INDIACEM, INDIAMART, INDIANB, IEX, INDHOTEL, IOC, IOB, IRCTC, IRFC, INDIGOPNTS,IGL, INDUSTOWER, INDUSINDBK, NAUKRI, INFY, INOXWIND, INTELLECT, INDIGO, IPCALAB, JBCHEPHARM,JKCEMENT, JBMA, JKLAKSHMI, JKPAPER, JMFINANCIL, JSWENERGY, JSWINFRA, JSWSTEEL, JAIBALAJI, J&KBANK,JINDALSAW, JSL, JINDALSTEL, JIOFIN, JUBLFOOD, JUBLINGREA, JUBLPHARMA, JWL, JUSTDIAL, JYOTHYLAB,KPRMILL, KEI, KNRCON, KPITTECH, KRBL, KSB, KAJARIACER, KPIL, KALYANKJIL, KANSAINER,KARURVYSYA, KAYNES, KEC, KFINTECH, KOTAKBANK, KIMS, LTF, LTTS, LICHSGFIN, LTIM,LT, LATENTVIEW, LAURUSLABS, LXCHEM, LEMONTREE, LICI, LINDEINDIA, LLOYDSME, LUPIN, MMTC,MRF, MTARTECH, LODHA, MGL, MAHSEAMLES, M&MFIN, M&M, MHRIL, MAHLIFE, MANAPPURAM,MRPL, MANKIND, MARICO, MARUTI, MASTEK, MFSL, MAXHEALTH, MAZDOCK, MEDPLUS, METROBRAND,METROPOLIS, MINDACORP, MSUMI, MOTILALOFS, MPHASIS, MCX, MUTHOOTFIN, NATCOPHARM, NBCC, NCC,NHPC, NLCINDIA, NMDC, NSLNISP, NTPC, NH, NATIONALUM, NAVINFLUOR, NESTLEIND, NETWORK18,NAM-INDIA, NUVAMA, NUVOCO, OBEROIRLTY, ONGC, OIL, OLECTRA, PAYTM, OFSS, POLICYBZR,PCBL, PIIND, PNBHOUSING, PNCINFRA, PVRINOX, PAGEIND, PATANJALI, PERSISTENT, PETRONET, PHOENIXLTD,PIDILITIND, PEL, PPLPHARMA, POLYMED, POLYCAB, POONAWALLA, PFC, POWERGRID, PRAJIND, PRESTIGE,PRINCEPIPE, PRSMJOHNSN, PGHH, PNB, QUESS, RRKABEL, RBLBANK, RECLTD, RHIM, RITES,RADICO, RVNL, RAILTEL, RAINBOW, RAJESHEXPO, RKFORGE, RCF, RATNAMANI, RTNINDIA, RAYMOND,REDINGTON, RELIANCE, RBA, ROUTE, SBFC, SBICARD, SBILIFE, SJVN, SKFINDIA, SRF,SAFARI, SAMMAANCAP, MOTHERSON, SANOFI, SAPPHIRE, SAREGAMA, SCHAEFFLER, SCHNEIDER, SHREECEM, RENUKA,SHRIRAMFIN, SHYAMMETL, SIEMENS, SIGNATURE, SOBHA, SOLARINDS, SONACOMS, SONATSOFTW, STARHEALTH, SBIN,SAIL, SWSOLAR, STLTECH, SUMICHEM, SPARC, SUNPHARMA, SUNTV, SUNDARMFIN, SUNDRMFAST, SUNTECK,SUPREMEIND, SUVENPHAR, SUZLON, SWANENERGY, SYNGENE, SYRMA, TBOTEK, TVSMOTOR, TVSSCS, TMB,TANLA, TATACHEM, TATACOMM, TCS, TATACONSUM, TATAELXSI, TATAINVEST, TATAMOTORS, TATAPOWER, TATASTEEL,TATATECH, TTML, TECHM, TEJASNET, NIACL, RAMCOCEM, THERMAX, TIMKEN, TITAGARH, TITAN,TORNTPHARM, TORNTPOWER, TRENT, TRIDENT, TRIVENI, TRITURBINE, TIINDIA, UCOBANK, UNOMINDA, UPL,UTIAMC, UJJIVANSFB, ULTRACEMCO, UNIONBANK, UBL, UNITDSPR, USHAMART, VGUARD, VIPIND, VAIBHAVGBL,VTL, VARROC, VBL, MANYAVAR, VEDL, VIJAYA, IDEA, VOLTAS, WELCORP, WELSPUNLIV,WESTLIFE, WHIRLPOOL, WIPRO, YESBANK, ZFCVINDIA, ZEEL, ZENSARTECH, ETERNAL, ZYDUSLIFE, ECLERX")],
    exchange: Annotated[str, Field(description="Exchange suffix, e.g. 'NS' for NSE or 'BO' for BSE")] = "NS",
    period: Annotated[str, Field(description="Data period,preferably convert any input to number of DAYS, e.g. '1d','5d','60d'")] = "2d",
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

@mcp.tool(description="-Apply retro iPhone 3GS vintage style filter to your photo and travel back to 505! (2009).\neg) Apply retro filter <attach picture>")
async def vintage_photo_filter(
    puch_image_data: Annotated[str, Field(description="Base64-encoded image data to transform")] = None,
) -> list[TextContent | ImageContent]:
    

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
    
@mcp.tool(description="- Start your day with your daily horoscope!\n eg) What is my horoscope for today, I'm a Libra\n\nPro Tip: use /reset if you ever feel it is  hallucinating")
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

    return data.get("horoscope", "No prediction available.")

# --- Run server ---
async def main():
    print("🚀 Starting MCP server on http://0.0.0.0:8086")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)

if __name__ == "__main__":
    asyncio.run(main())









































































