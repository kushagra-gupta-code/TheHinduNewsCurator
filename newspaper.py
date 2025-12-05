"""
The Hindu News Curation Script with Gemini API
Extracts top 20 high-impact articles and opens them via smry.ai
"""

import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import json
import webbrowser
import time
from datetime import datetime
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Force UTF-8 encoding for Windows terminals
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it in .env file.")
genai.configure(api_key=GEMINI_API_KEY)

import re

class NewsArticle:
    def __init__(self, title, url, page, section, teaser=""):
        self.title = title
        self.url = url
        self.page = page
        self.section = section
        self.teaser = teaser
        self.impact_score = 0
        self.analysis = None
        self.content = ""

class HinduNewsCurator:
    def __init__(self, date=None, edition="th_delhi"):
        self.date = date or datetime.now().strftime("%Y-%m-%d")
        self.edition = edition
        self.base_url = f"https://www.thehindu.com/todays-paper/{self.date}/{edition}/"
        self.articles = []
        self.top_20 = []
        self.seen_urls = set()
        
    def scrape_all_sections(self):
        """Scrape all sections by extracting embedded JSON from the page"""
        print(f"\n🔍 Starting to scrape The Hindu Today's Paper ({self.date}, edition: {self.edition})...\n")
        
        try:
            response = requests.get(self.base_url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            response.raise_for_status()
            html_content = response.text
            
            # Extract the grouped_articles JSON from the page
            start_marker = 'grouped_articles = {"TH_'
            if start_marker not in html_content:
                print("❌ Could not find article data in the page. Try a different edition.")
                return
            
            start_idx = html_content.find(start_marker) + len('grouped_articles = ')
            
            # Find the matching closing brace
            brace_count = 0
            end_idx = start_idx
            for i, char in enumerate(html_content[start_idx:]):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = start_idx + i + 1
                        break
            
            json_str = html_content[start_idx:end_idx]
            grouped_articles = json.loads(json_str)
            
            print(f"📰 Found sections: {list(grouped_articles.keys())}\n")
            
            for section, articles in grouped_articles.items():
                print(f"  Processing {section}: {len(articles)} articles")
                for article_data in articles:
                    href = article_data.get('href', '')
                    full_url = f"https://www.thehindu.com{href}" if href.startswith('/') else href
                    
                    # Deduplication
                    if full_url in self.seen_urls:
                        continue
                    self.seen_urls.add(full_url)
                    
                    news_article = NewsArticle(
                        title=article_data.get('articleheadline', 'No Title'),
                        url=full_url,
                        page=article_data.get('pageno', 'N/A'),
                        section=section,
                        teaser=article_data.get('teaser_text', '')
                    )
                    self.articles.append(news_article)
            
            print(f"\n📊 Total unique articles extracted: {len(self.articles)}\n")
            
            # Log all URLs
            print("🔗 Extracted Article URLs:")
            for i, article in enumerate(self.articles, 1):
                print(f"{i}. [{article.section}] {article.url}")
            print("\n")
            
        except requests.RequestException as e:
            print(f"❌ Error fetching page: {e}")
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing article data: {e}")
    
    def analyze_all_articles(self):
        """Analyze all articles using Gemini API in batches"""
        if not self.articles:
            print("No articles to analyze.")
            return

        BATCH_SIZE = 70  # Process 70 articles at a time to avoid response truncation
        total = len(self.articles)
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"\n🤖 Analyzing {total} articles with Gemini API (in batches of {BATCH_SIZE})...\n")
        
        model = genai.GenerativeModel('gemini-2.5-flash')
        all_results = {}
        
        # Tracking variables for enhanced logging
        cumulative_success = 0
        batch_times = []
        
        for batch_start in range(0, total, BATCH_SIZE):
            batch_num = batch_start // BATCH_SIZE + 1
            batch_end = min(batch_start + BATCH_SIZE, total)
            batch_size_actual = batch_end - batch_start
            batch_articles = self.articles[batch_start:batch_end]
            
            print(f"📦 Batch {batch_num}/{total_batches}: Processing articles {batch_start+1}-{batch_end} of {total}...")
            
            # Start timing
            batch_start_time = time.time()
            
            # Prepare the articles for this batch
            articles_text = ""
            for i, article in enumerate(batch_articles):
                global_id = batch_start + i
                teaser_snippet = article.teaser[:150] if article.teaser else "No description"
                articles_text += f"ID: {global_id}\nTitle: {article.title}\nSection: {article.section}\nSummary: {teaser_snippet}\n---\n"

            prompt = f"""Analyze these news articles for impact value. Return ONLY valid JSON.

ARTICLES:
{articles_text}

For each article, evaluate:
- impact_scope (0-10): National=10, Regional=7-8, Sectoral=5-7, Limited=2-4
- governance (0-2): Policy/governance impact
- accountability (0-2): Holds institutions accountable
- geopolitical (0-2): Global/regional significance
- anti_bubble (0-2): Expands perspective
- newsworthiness (0-2): Breaking/developing news

Return JSON: {{"results": [{{"id": <int>, "impact_scope": <int>, "governance": <int>, "accountability": <int>, "geopolitical": <int>, "anti_bubble": <int>, "newsworthiness": <int>, "total_score": <sum>, "reasoning": "<short>"}}, ...]}}"""

            batch_success = 0
            batch_scores = []
            
            try:
                response = model.generate_content(prompt)
                json_str = response.text.strip()
                
                if '```json' in json_str:
                    json_str = json_str.split('```json')[1].split('```')[0]
                elif '```' in json_str:
                    json_str = json_str.split('```')[1].split('```')[0]
                
                data = json.loads(json_str.strip())
                
                # Handle both formats
                if isinstance(data, list):
                    results = data
                else:
                    results = data.get('results', [])
                
                # Add to global results and track scores
                for item in results:
                    try:
                        article_id = int(item['id'])
                        all_results[article_id] = item
                        batch_success += 1
                        batch_scores.append(item.get('total_score', 0))
                    except (ValueError, TypeError):
                        pass
                        
            except Exception as e:
                print(f"   ⚠️ Batch error: {e}")
            
            # End timing
            batch_elapsed = time.time() - batch_start_time
            batch_times.append(batch_elapsed)
            
            # Update cumulative success
            cumulative_success += batch_success
            
            # Calculate batch average score
            batch_avg_score = sum(batch_scores) / len(batch_scores) if batch_scores else 0
            
            # Calculate ETA based on average batch time
            avg_batch_time = sum(batch_times) / len(batch_times)
            remaining_batches = total_batches - batch_num
            eta_seconds = avg_batch_time * remaining_batches
            
            # Progress percentage
            progress_pct = (batch_end / total) * 100
            
            # Print enhanced batch summary
            print(f"   ✅ {batch_success}/{batch_size_actual} analyzed in {batch_elapsed:.1f}s | Avg score: {batch_avg_score:.1f}/20")
            print(f"   📈 Progress: {cumulative_success}/{total} ({progress_pct:.0f}%)", end="")
            if remaining_batches > 0:
                print(f" | ⏳ ETA: ~{eta_seconds:.1f}s")
            else:
                print()  # Final batch, no ETA needed
            print()  # Blank line between batches
                
            time.sleep(1)  # Small delay between batches
        
        # Map all results to articles
        success_count = 0
        total_score_sum = 0
        for i, article in enumerate(self.articles):
            if i in all_results:
                analysis = all_results[i]
                article.impact_score = analysis.get('total_score', 0)
                article.analysis = analysis
                success_count += 1
                total_score_sum += article.impact_score
            else:
                article.impact_score = 0
        
        # Final summary
        overall_avg = total_score_sum / success_count if success_count > 0 else 0
        total_time = sum(batch_times)
        print(f"{'='*60}")
        print(f"✅ Analysis Complete!")
        print(f"   📊 Articles: {success_count}/{total} successfully analyzed")
        print(f"   📈 Overall Avg Score: {overall_avg:.1f}/20")
        print(f"   ⏱️  Total Time: {total_time:.1f}s ({total_time/success_count:.2f}s per article)")
        print(f"{'='*60}")
    
    def curate_top_n(self, top_count=20):
        """Select top N articles with anti-bubble diversification"""
        print(f"🎯 Curating top {top_count} anti-bubble articles...\n")
        
        # Sort by impact score
        sorted_articles = sorted(self.articles, key=lambda x: x.impact_score, reverse=True)
        
        # Apply anti-bubble constraints (scaled to top_count)
        selected = []
        # Get unique sections from the scraped articles
        unique_sections = set(a.section for a in self.articles)
        section_count = {s: 0 for s in unique_sections}
        
        # Scale minimums based on top_count
        scale = top_count / 20
        min_per_section = {
            "TH_Foreign": max(1, int(3 * scale)),
            "TH_Business": max(1, int(2 * scale)),
            "TH_Regional": max(1, int(5 * scale)),
            "Sports": max(1, int(1 * scale)),
            "TH_Science": max(1, int(1 * scale)),
            "TH_Edit": max(1, int(1 * scale)),
            "TH_National": max(1, int(3 * scale))
        }
        
        # First pass: ensure minimum diversity
        for article in sorted_articles:
            if len(selected) >= top_count:
                break
            
            section = article.section
            if section_count[section] < min_per_section.get(section, 0):
                selected.append(article)
                section_count[section] += 1
        
        # Second pass: fill remaining slots with highest impact (no max limit)
        for article in sorted_articles:
            if len(selected) >= top_count:
                break
            
            if article not in selected:
                selected.append(article)
                section_count[article.section] += 1
        
        self.top_20 = selected[:top_count]
        
        print(f"✅ Curated {len(self.top_20)} top articles")
        print(f"\nDiversity breakdown:")
        for section in unique_sections:
            count = sum(1 for a in self.top_20 if a.section == section)
            if count > 0:
                print(f"  {section}: {count} articles")
        
        return self.top_20
    
    def open_in_smry_tabs(self):
        """Open all top 20 articles in browser tabs via smry.ai"""
        print(f"\n🌐 Opening {len(self.top_20)} articles in browser via smry.ai...\n")
        
        for i, article in enumerate(self.top_20, 1):
            smry_url = f"https://smry.ai/{article.url}"
            print(f"[{i}/20] Opening: {article.title[:50]}...")
            
            try:
                webbrowser.open(smry_url, new=1)
                time.sleep(1)  # Delay to prevent overwhelming browser
            except Exception as e:
                print(f"  ❌ Error opening tab: {e}")
        
        print("\n✅ All articles opened in smry.ai tabs!\n")
    
    def generate_report(self):
        """Generate detailed report of curated articles"""
        print("\n" + "="*80)
        print("📋 THE HINDU TOP 20 HIGH-IMPACT ARTICLES - CURATION REPORT")
        print("="*80 + "\n")
        
        for rank, article in enumerate(self.top_20, 1):
            print(f"RANK {rank}: {article.title}")
            print(f"  Section: {article.section} | Page: {article.page}")
            print(f"  Impact Score: {article.impact_score}/20")
            print(f"  URL: {article.url}")
            if article.analysis:
                print(f"  Reasoning: {article.analysis.get('reasoning', 'N/A')}")
            print()
        
        print("="*80 + "\n")

def input_with_timeout(prompt, timeout=5, default=""):
    """Get input with a timeout. Returns default if no input within timeout."""
    import sys
    import select
    
    print(prompt, end='', flush=True)
    
    # Windows doesn't support select on stdin, so we use a different approach
    if sys.platform.startswith('win'):
        import msvcrt
        import time
        
        start_time = time.time()
        input_chars = []
        
        while (time.time() - start_time) < timeout:
            if msvcrt.kbhit():
                char = msvcrt.getwch()
                if char == '\r':  # Enter pressed
                    print()  # New line
                    return ''.join(input_chars).strip()
                elif char == '\x08':  # Backspace
                    if input_chars:
                        input_chars.pop()
                        print('\b \b', end='', flush=True)
                else:
                    input_chars.append(char)
                    print(char, end='', flush=True)
            time.sleep(0.1)
        
        print(f"\n⏱️  Timeout after {timeout}s, using default: '{default}'")
        return default
    else:
        # Unix-like systems
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if ready:
            return sys.stdin.readline().strip()
        else:
            print(f"\n⏱️  Timeout after {timeout}s, using default: '{default}'")
            return default

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("🗞️  THE HINDU NEWS CURATION SYSTEM")
    print("Auto-curates top 20 high-impact articles using Gemini API")
    print("="*80 + "\n")
    
    # Initialize curator
    curator = HinduNewsCurator(edition="th_delhi")
    
    # Step 1: Scrape all sections
    curator.scrape_all_sections()
    
    # Step 2: Ask user if they want to proceed with LLM analysis (default: yes after 5s)
    proceed = input_with_timeout("\nProceed with AI analysis? (yes/no, default=yes in 5s): ", timeout=5, default="yes")
    
    if proceed in ['yes', 'y', '']:
        # API is already configured at the top of the file
        # If you need to use a different key, set GEMINI_API_KEY environment variable
        import os
        env_key = os.environ.get('GEMINI_API_KEY', '')
        if env_key:
            genai.configure(api_key=env_key)
            print("Using GEMINI_API_KEY from environment variable.")
        else:
            print("Using pre-configured API key.")
        
        # Step 3: Analyze articles with Gemini API
        curator.analyze_all_articles()
        
        # Step 4: Curate top 20 with anti-bubble criteria
        curator.curate_top_20()
        
        # Step 5: Generate report
        curator.generate_report()
        
        # Step 6: Open all articles in browser via smry.ai (default: no after 5s)
        user_input = input_with_timeout("\nOpen all 20 articles in browser via smry.ai? (yes/no, default=no in 5s): ", timeout=5, default="no")
        if user_input in ['yes', 'y']:
            curator.open_in_smry_tabs()
        else:
            print("Skipping browser opening.")
    else:
        print("\nSkipping AI analysis. Exiting...")

    print("\n✨ Process complete!\n")

if __name__ == "__main__":
    main()
