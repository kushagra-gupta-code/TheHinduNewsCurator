"""
Flask Web UI for The Hindu News Curation
"""

from flask import Flask, render_template, jsonify, request
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Force UTF-8 encoding
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")

from newspaper import HinduNewsCurator
from chat import chat_with_article
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
import re

app = Flask(__name__)


# Global curator instance
curator = None
status = {"message": "Ready to scrape", "step": "idle"}

# Configure APIs from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it in .env file.")
genai.configure(api_key=GEMINI_API_KEY)

# Configure OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "kwaipilot/kat-coder-pro:free")
USE_OPENROUTER_FALLBACK = os.getenv("USE_OPENROUTER_FALLBACK", "true").lower() == "true"

# Check OpenRouter availability
try:
    from openrouter import OpenRouter

    OPENROUTER_AVAILABLE = True
except ImportError:
    OPENROUTER_AVAILABLE = False
    print("⚠️  OpenRouter not available. Install with: pip install openrouter")


@app.route("/")
def index():
    """Serve the main UI page"""
    return render_template("index.html")


@app.route("/scrape", methods=["POST"])
def scrape():
    """Scrape articles from The Hindu"""
    global curator, status

    edition = request.json.get("edition", "th_delhi")
    date = request.json.get("date", None)

    try:
        status = {"message": "Scraping articles...", "step": "scraping"}
        curator = HinduNewsCurator(date=date, edition=edition)
        curator.scrape_all_sections()

        status = {
            "message": f"Scraped {len(curator.articles)} articles",
            "step": "scraped",
        }

        return jsonify(
            {
                "success": True,
                "count": len(curator.articles),
                "sections": list(set(a.section for a in curator.articles)),
            }
        )
    except Exception as e:
        status = {"message": f"Error: {str(e)}", "step": "error"}
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/analyze", methods=["POST"])
def analyze():
    """Analyze articles with Gemini AI"""
    global curator, status

    if not curator or not curator.articles:
        return jsonify(
            {"success": False, "error": "No articles to analyze. Scrape first."}
        ), 400

    top_count = request.json.get("top_count", 20) if request.json else 20

    try:
        status = {"message": "Analyzing with AI...", "step": "analyzing"}
        curator.analyze_all_articles()
        curator.curate_top_n(top_count)

        # Get provider status for response
        provider_status = curator.llm_provider.get_provider_status()

        status = {
            "message": f"Curated {len(curator.top_20)} top articles",
            "step": "analyzed",
            "provider": provider_status,
        }

        return jsonify(
            {"success": True, "count": len(curator.top_20), "provider": provider_status}
        )
    except Exception as e:
        status = {"message": f"Error: {str(e)}", "step": "error"}
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/results", methods=["GET"])
def results():
    """Get curated top articles with score breakdown"""
    global curator

    if not curator or not curator.top_20:
        return jsonify({"articles": [], "total_scraped": 0})

    articles = []
    for i, article in enumerate(curator.top_20, 1):
        # Extract individual scores from analysis
        scores = {
            "impact_scope": 0,
            "governance": 0,
            "accountability": 0,
            "geopolitical": 0,
            "anti_bubble": 0,
            "newsworthiness": 0,
        }
        reasoning = ""

        if article.analysis and isinstance(article.analysis, dict):
            reasoning = article.analysis.get("reasoning", "")
            scores["impact_scope"] = article.analysis.get("impact_scope", 0)
            scores["governance"] = article.analysis.get("governance", 0)
            scores["accountability"] = article.analysis.get("accountability", 0)
            scores["geopolitical"] = article.analysis.get("geopolitical", 0)
            scores["anti_bubble"] = article.analysis.get("anti_bubble", 0)
            scores["newsworthiness"] = article.analysis.get("newsworthiness", 0)

        articles.append(
            {
                "rank": i,
                "title": article.title,
                "section": article.section,
                "page": article.page,
                "score": article.impact_score,
                "scores": scores,
                "url": article.url,
                "smry_url": f"https://smry.ai/{article.url.replace('https://', '').replace('http://', '')}",
                "reasoning": reasoning,
            }
        )

    return jsonify(
        {
            "articles": articles,
            "total_scraped": len(curator.articles) if curator.articles else 0,
        }
    )


@app.route("/status", methods=["GET"])
def get_status():
    """Get current status"""
    global status, curator

    status_data = status.copy()

    # Add provider information if curator exists
    if curator and hasattr(curator, "llm_provider"):
        provider_status = curator.llm_provider.get_provider_status()
        status_data.update(
            {
                "provider": provider_status,
                "openrouter_configured": bool(OPENROUTER_API_KEY)
                and OPENROUTER_AVAILABLE,
                "fallback_enabled": USE_OPENROUTER_FALLBACK,
            }
        )

    return jsonify(status_data)


@app.route("/open-smry", methods=["POST"])
def open_smry():
    """Open articles in smry.ai tabs"""
    global curator

    if not curator or not curator.top_20:
        return jsonify({"success": False, "error": "No curated articles"}), 400

    try:
        curator.open_in_smry_tabs()
        return jsonify({"success": True, "count": len(curator.top_20)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def fetch_article_text(url):
    """Fetch article content from URL"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        # Try to find the main article body
        # The Hindu often uses specific IDs or classes
        article_body = soup.find("div", {"id": re.compile(r"content-body-.*")})

        if article_body:
            paragraphs = article_body.find_all("p")
        else:
            # Fallback to all paragraphs
            paragraphs = soup.find_all("p")

        text = "\n\n".join(
            [p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 30]
        )
        return text
    except Exception as e:
        print(f"Error fetching article text: {e}")
        return ""


@app.route("/chat", methods=["POST"])
def chat():
    """Chat about an article with Gemini + Google Search grounding"""
    global curator
    data = request.json

    article_title = data.get("article_title", "")
    article_url = data.get("article_url", "")
    message = data.get("message", "")
    history = data.get("history", [])

    if not message:
        return jsonify({"success": False, "error": "No message provided"}), 400

    # Try to find article in curator to get/set cached content
    article_content = ""
    if curator:
        # Check top 20 first
        found_article = next((a for a in curator.top_20 if a.url == article_url), None)
        # Then check all articles
        if not found_article:
            found_article = next(
                (a for a in curator.articles if a.url == article_url), None
            )

        if found_article:
            if not found_article.content:
                print(f"Fetching content for: {article_title}")
                found_article.content = fetch_article_text(article_url)
            article_content = found_article.content

    # If still no content (e.g. curator reset or not found), try fetching anyway
    if not article_content and article_url:
        article_content = fetch_article_text(article_url)

    try:
        response = chat_with_article(
            article_title=article_title,
            article_url=article_url,
            message=message,
            article_content=article_content,
            history=history,
        )
        return jsonify({"success": True, "response": response})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print("\n" + "=" * 60)
    print("🗞️  THE HINDU NEWS CURATOR - Web UI")
    print(f"Open http://localhost:{port} in your browser")
    print("=" * 60 + "\n")
    app.run(debug=False, host="0.0.0.0", port=port)
