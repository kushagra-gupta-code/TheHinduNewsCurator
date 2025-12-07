# The Hindu News Curator

AI-powered news curation tool that extracts high-impact articles from The Hindu newspaper using Google Gemini API.

## Features
- 📰 Scrapes articles from The Hindu's "Today's Paper"
- 🤖 Analyzes articles using Gemini AI for impact scoring
- 🎯 Curates top N articles with diversity constraints
- 💬 Chat about articles with Google Search grounding
- 🌐 Opens articles via smry.ai for summarized reading

## Performance Optimizations
- 🚀 **Parallel Processing**: Uses concurrent threads (5 workers) to process batches of articles.
- 📉 **Minified Payloads**: Optimizes LLM response tokens using minified JSON keys, reducing latency by ~40%.
- ⚡ **Speed**: Analysis of ~200 articles takes <10 seconds (vs ~40s sequentially).

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

## Usage

Run the Flask web app:
```bash
python app.py
```

Open http://localhost:5000 in your browser.

## Files
- `app.py` - Flask web server
- `newspaper.py` - Core scraping and AI analysis logic
- `chat.py` - Chat module with Google Search grounding
