"""
Chat module for article discussions with Gemini and OpenRouter fallback.
"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables from .env file
load_dotenv()

# Initialize Google client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it in .env file.")

# Initialize OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "kwaipilot/kat-coder-pro:free")
USE_OPENROUTER_FALLBACK = os.getenv("USE_OPENROUTER_FALLBACK", "true").lower() == "true"

# Import OpenRouter if available
try:
    from openrouter import OpenRouter

    OPENROUTER_AVAILABLE = True
except ImportError:
    OPENROUTER_AVAILABLE = False
    print("⚠️  OpenRouter not available. Install with: pip install openrouter")


class GoogleChatProvider:
    """Google Gemini chat provider"""

    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found.")
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def generate_content(self, messages: list, **kwargs) -> str:
        try:
            # Enable Google Search grounding
            grounding_tool = types.Tool(google_search=types.GoogleSearch())

            config = types.GenerateContentConfig(
                tools=[grounding_tool],
                system_instruction=kwargs.get("system_instruction", ""),
            )

            response = self.client.models.generate_content(
                model="gemini-2.5-flash", contents=messages, config=config
            )

            return response.text
        except Exception as e:
            if self.is_rate_limit_error(e):
                print("🚫 Google rate limit detected")
                raise
            else:
                print(f"❌ Google chat error: {e}")
                raise

    def is_rate_limit_error(self, error: Exception) -> bool:
        """Check if error is rate limit related"""
        error_str = str(error).lower()
        return any(
            keyword in error_str
            for keyword in [
                "rate limit",
                "resource exhausted",
                "quota exceeded",
                "too many requests",
                "429",
                "quota",
            ]
        )


class OpenRouterChatProvider:
    """OpenRouter chat provider"""

    def __init__(self):
        if not OPENROUTER_AVAILABLE:
            raise ImportError("OpenRouter not available")
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY not found.")

        self.client = OpenRouter(api_key=OPENROUTER_API_KEY)
        self.model = OPENROUTER_MODEL

    def generate_content(self, messages: list, **kwargs) -> str:
        try:
            # Convert messages to OpenRouter format
            openrouter_messages = []
            for msg in messages:
                if hasattr(msg, "parts"):
                    # Google format - convert to text
                    content = "".join(
                        part.text for part in msg.parts if hasattr(part, "text")
                    )
                else:
                    # Already text format
                    content = str(msg)

                openrouter_messages.append(
                    {
                        "role": "user" if "user" in str(msg).lower() else "assistant",
                        "content": content,
                    }
                )

            response = self.client.chat.send(
                model=self.model,
                messages=openrouter_messages,
                temperature=0.7,
                max_tokens=4000,
            )

            return response.choices[0].message.content
        except Exception as e:
            if self.is_rate_limit_error(e):
                print("🚫 OpenRouter rate limit detected")
                raise
            else:
                print(f"❌ OpenRouter chat error: {e}")
                raise

    def is_rate_limit_error(self, error: Exception) -> bool:
        """Check if error is rate limit related"""
        error_str = str(error).lower()
        return any(
            keyword in error_str
            for keyword in ["rate limit", "too many requests", "429"]
        )


class HybridChatProvider:
    """Hybrid chat provider with automatic fallback from Google to OpenRouter"""

    def __init__(self):
        self.google_provider = GoogleChatProvider()
        self.openrouter_provider = None
        self.use_openrouter = False

        # Initialize OpenRouter provider if available and enabled
        if USE_OPENROUTER_FALLBACK and OPENROUTER_AVAILABLE and OPENROUTER_API_KEY:
            try:
                self.openrouter_provider = OpenRouterChatProvider()
                print("✅ OpenRouter chat fallback provider initialized")
            except Exception as e:
                print(f"⚠️  Failed to initialize OpenRouter chat: {e}")

    def generate_content(self, messages: list, **kwargs) -> str:
        """Generate content with automatic fallback"""
        # Try Google first
        try:
            print("🤖 Using Google Gemini for chat")
            return self.google_provider.generate_content(messages, **kwargs)
        except Exception as e:
            if self.google_provider.is_rate_limit_error(e) and self.openrouter_provider:
                # Google rate limit - fallback to OpenRouter
                print("🔄 Google chat rate limit hit, falling back to OpenRouter...")
                try:
                    print("🤖 Using OpenRouter for chat")
                    return self.openrouter_provider.generate_content(messages, **kwargs)
                except Exception as e2:
                    # OpenRouter failed, raise the original Google error or the OpenRouter error
                    raise e2
            else:
                # Not a rate limit or no fallback available
                raise

    def get_provider_status(self) -> dict:
        """Get current provider status"""
        return {
            "using_openrouter": self.use_openrouter,
            "openrouter_available": self.openrouter_provider is not None,
            "google_available": True,
        }


def chat_with_article(
    article_title: str,
    article_url: str,
    message: str,
    article_content: str = "",
    history: list = None,
) -> str:
    """
    Chat about an article with Google Search grounding and OpenRouter fallback.

    Args:
        article_title: Title of the article being discussed
        article_url: URL of the article
        message: User's message/question
        article_content: Full text or summary of the article
        history: Previous chat messages [{"role": "user/assistant", "content": "..."}]

    Returns:
        AI response as string
    """
    try:
        # Initialize hybrid chat provider
        chat_provider = HybridChatProvider()
        print(f"💬 Chat Provider Status: {chat_provider.get_provider_status()}")

        # Build system instruction
        system_instruction = f"""You are a research assistant helping the user explore and learn more about topics related to this news article:

Title: {article_title}
URL: {article_url}

Article Summary:
{article_content}

YOUR PRIMARY ROLE:
You are NOT limited to the article content. The article is just a STARTING POINT. Your main job is to use web search to find additional information, context, background, and latest updates from verified sources.

INSTRUCTIONS:
1. ALWAYS USE WEB SEARCH: For any question, actively search for current, verified information from multiple sources.
2. CITE SOURCES: Include URLs for all facts you provide. Format: [Source Name](URL)
3. GO BEYOND THE ARTICLE: Provide background info, related developments, expert opinions, historical context, and latest updates.
4. BE HONEST: If you cannot find reliable information, say so. Never make up facts.
5. MULTIPLE PERSPECTIVES: When relevant, show different viewpoints from various sources.

The user wants to learn MORE about this topic than what the article provides. Be proactive in searching and sharing relevant external information."""

        # Build conversation contents for Google format
        contents = []

        # Add history if exists
        if history:
            for msg in history:
                role = "user" if msg.get("role") == "user" else "model"
                contents.append(
                    types.Content(
                        role=role, parts=[types.Part(text=msg.get("content", ""))]
                    )
                )

        # Add current message
        contents.append(types.Content(role="user", parts=[types.Part(text=message)]))

        # Use hybrid provider with fallback
        response = chat_provider.generate_content(
            messages=contents, system_instruction=system_instruction
        )

        return response

    except Exception as e:
        print(f"Chat error: {e}")
        return f"Sorry, I encountered an error: {str(e)}"


if __name__ == "__main__":
    # Test the chat function
    response = chat_with_article(
        article_title="Test Article",
        article_url="https://example.com",
        message="What is the latest news about AI?",
    )
    print(response)
