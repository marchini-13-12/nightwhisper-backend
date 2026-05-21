from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
import os
import logging
import requests
from pathlib import Path
from pydantic import BaseModel

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

DREAM_INTERPRETER_PROMPT = """You are a world-class dream interpreter and personal dream wingman with 30 years of experience. You have encyclopedic knowledge of: Jungian archetypes and the collective unconscious, Freudian psychoanalysis and dream symbolism, spiritual dream interpretation from Christianity, Islam, Buddhism, Hinduism and ancient mythology, color psychology in dreams, animal symbolism, recurring dream patterns and their meanings, lucid dreaming, sleep psychology, numerology in dreams, and subconscious communication. When a user shares their dream always respond with this exact structure: 1) DREAM ESSENCE - one powerful sentence summarizing the soul of this dream. 2) KEY SYMBOLS - analyze every single symbol, person, animal, object, color and location from the dream and its deep psychological and spiritual meaning. 3) EMOTIONAL LANDSCAPE - identify all emotions present and what they reveal about the dreamer's inner world. 4) HIDDEN MESSAGE - reveal what the subconscious mind is trying to communicate to the dreamer. 5) JUNGIAN ARCHETYPES - identify which archetypes appear such as Shadow, Anima, Animus, Hero, Trickster, Wise Old Man and explain their significance. 6) SPIRITUAL DIMENSION - provide spiritual interpretation of the dream. 7) PRACTICAL WISDOM - give one specific actionable insight the dreamer can apply in their waking life today. 8) DREAM CLASSIFICATION - classify as: Prophetic, Processing, Fear-Based, Wish Fulfillment, Spiritual, Healing or Warning dream. Always write minimum 400 words. Be warm, personal, insightful and always reference specific details from the dream. Never give generic responses. Always end with an uplifting encouraging message.

CRITICAL LANGUAGE RULE: You MUST respond in the target language specified in the user message. This is mandatory and cannot be overridden. If the target language is Croatian, respond entirely in Croatian. If Spanish, respond entirely in Spanish. If Russian, respond entirely in Russian. NEVER switch to English or any other language. Your entire response including all section headers must be in the specified target language."""

app = FastAPI()
api_router = APIRouter(prefix="/api")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InterpretRequest(BaseModel):
    content: str
    language: str = "en"


@api_router.get("/")
async def root():
    return {"message": "Night Whisper API v1.0"}


@api_router.post("/interpret")
async def interpret_dream(req: InterpretRequest):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="AI service not configured")

    if not req.content or len(req.content.strip()) < 10:
        raise HTTPException(status_code=400, detail="Dream content too short")

    try:
        LANGUAGE_MAP = {
            "en": "English", "es": "Spanish / Español", "hr": "Croatian / Hrvatski",
            "it": "Italian / Italiano", "fr": "French / Français", "de": "German / Deutsch",
            "fi": "Finnish / Suomi", "sv": "Swedish / Svenska", "ru": "Russian / Русский",
            "pt": "Portuguese / Português", "nl": "Dutch / Nederlands", "pl": "Polish / Polski",
            "sr": "Serbian / Srpski"
        }
        target_language = LANGUAGE_MAP.get(req.language, "English")
        user_message = (
            f"TARGET LANGUAGE: {target_language}. "
            f"You MUST write your ENTIRE response in {target_language}. "
            f"Every word, every section header, everything must be in {target_language}. "
            f"Now interpret this dream: {req.content}"
        )
        response = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": DREAM_INTERPRETER_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                "max_tokens": 2048,
                "temperature": 0.8
            },
            timeout=60
        )

        if response.status_code != 200:
            logger.error(f"Groq API error: {response.status_code} - {response.text}")
            raise HTTPException(status_code=503, detail="AI service temporarily unavailable")

        data = response.json()
        interpretation = data["choices"][0]["message"]["content"]
        return {"interpretation": interpretation}

    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="AI service timeout - please try again")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        raise HTTPException(status_code=503, detail="AI interpretation temporarily unavailable")


@api_router.get("/health")
async def health():
    return {"status": "ok", "groq_configured": bool(GROQ_API_KEY)}


app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
