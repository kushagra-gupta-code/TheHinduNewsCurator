# The Hindu News Curator

AI-powered news curation tool that extracts high-impact articles from The Hindu newspaper using hybrid AI with Google Gemini and OpenRouter fallback.

## Features
- 📰 Scrapes articles from The Hindu's "Today's Paper"
- 🤖 **Hybrid AI Analysis**: Uses Google Gemini with automatic OpenRouter fallback for impact scoring
- 🎯 Curates top N articles with diversity constraints
- 💬 Chat about articles with Google Search grounding
- 🌐 Opens articles via smry.ai for summarized reading
- 🛡️ **Reliability**: Automatic failover prevents service interruptions
- ⚡ **High Performance**: Async processing with 5x speed improvement

## Performance Optimizations
- 🚀 **Async Processing**: Concurrent API calls (5 workers) reduce analysis time from ~40s to <10s for 200 articles
- 📉 **Minified Payloads**: Optimized LLM response tokens using compressed JSON keys, reducing latency by ~40%
- 🔄 **Smart Fallback**: Automatic switching between Google Gemini and OpenRouter on rate limits
- 📊 **Intelligent Batching**: 25-article batches optimized for API limits and performance

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   OPENROUTER_API_KEY=your_openrouter_api_key_here  # Optional, for fallback
   ```

## Usage

Run the Flask web app:
```bash
python app.py
```

Open http://localhost:5000 in your browser.

## Testing

Run the comprehensive test suite:
```bash
# Integration tests
python test_integration.py

# OpenRouter fallback tests
python test_openrouter_consolidated.py

# Flask score verification
python test_flask_scores.py
```

## Files
- `app.py` - Flask web server with enhanced UI
- `newspaper.py` - Core scraping and hybrid AI analysis logic
- `chat.py` - Chat module with Google Search grounding and OpenRouter fallback
- `test_integration.py` - Consolidated integration tests
- `test_openrouter_consolidated.py` - OpenRouter fallback testing
- `test_flask_scores.py` - Flask score population verification
