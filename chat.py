"""
Chat module for article discussions with Gemini and Google Search grounding.
"""
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables from .env file
load_dotenv()

# Initialize client
API_KEY = os.getenv('GEMINI_API_KEY')
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it in .env file.")

def chat_with_article(article_title: str, article_url: str, message: str, article_content: str = "", history: list = None) -> str:
    """
    Chat about an article with Google Search grounding for real-time information.
    
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
        client = genai.Client(api_key=API_KEY)
        
        # Enable Google Search grounding
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        
        system_instruction = f"""You are a research assistant helping the user explore and learn more about topics related to this news article:

Title: {article_title}
URL: {article_url}

Article Summary:
{article_content}

YOUR PRIMARY ROLE:
You are NOT limited to the article content. The article is just a STARTING POINT. Your main job is to use Google Search to find additional information, context, background, and latest updates from verified sources.

INSTRUCTIONS:
1. ALWAYS USE GOOGLE SEARCH: For any question, actively search for current, verified information from multiple sources.
2. CITE SOURCES: Include URLs for all facts you provide. Format: [Source Name](URL)
3. GO BEYOND THE ARTICLE: Provide background info, related developments, expert opinions, historical context, and latest updates.
4. BE HONEST: If you cannot find reliable information, say so. Never make up facts.
5. MULTIPLE PERSPECTIVES: When relevant, show different viewpoints from various sources.

The user wants to learn MORE about this topic than what the article provides. Be proactive in searching and sharing relevant external information."""

        config = types.GenerateContentConfig(
            tools=[grounding_tool],
            system_instruction=system_instruction
        )
        
        # Build conversation contents
        contents = []
        
        # Add history if exists
        if history:
            for msg in history:
                role = "user" if msg.get("role") == "user" else "model"
                contents.append(types.Content(
                    role=role,
                    parts=[types.Part(text=msg.get("content", ""))]
                ))
        
        # Add current message
        contents.append(types.Content(
            role="user",
            parts=[types.Part(text=message)]
        ))
        
        # Generate response
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=config
        )
        
        return response.text
        
    except Exception as e:
        print(f"Chat error: {e}")
        return f"Sorry, I encountered an error: {str(e)}"


if __name__ == "__main__":
    # Test the chat function
    response = chat_with_article(
        article_title="Test Article",
        article_url="https://example.com",
        message="What is the latest news about AI?"
    )
    print(response)
