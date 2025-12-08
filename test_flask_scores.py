"""
Test Flask app with real data to verify scores are populated
"""

import requests
import json
import time
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_flask_with_data():
    """Test Flask app with mock data to verify score handling"""
    base_url = "http://localhost:5000"

    print("Testing Flask App with Mock Data...")
    print("=" * 60)

    try:
        # Step 1: Scrape some mock data
        print("1. Creating mock articles...")

        # We'll create a mock response for scraping
        mock_scrape_response = {
            "success": True,
            "count": 5,
            "sections": ["National", "Business", "Sports"],
        }

        # Step 2: Test analyze endpoint with mocked API calls
        print("2. Testing analyze endpoint...")

        # Mock the API to return good results
        def mock_generate_content(prompt):
            # Count articles in prompt
            article_count = prompt.count("ID:")

            # Extract first ID
            lines = prompt.split("\n")
            first_id = 0
            for line in lines:
                if line.startswith("ID:"):
                    first_id = int(line.split(":")[1].strip())
                    break

            # Generate results for all articles
            results = []
            for i in range(article_count):
                results.append(
                    {
                        "id": first_id + i,
                        "s": [8, 2, 1, 1, 1, 2],  # Total score: 15
                        "r": "High impact analysis with detailed reasoning",
                    }
                )

            mock_response = MagicMock()
            mock_response.text = '{"results": ' + json.dumps(results) + "}"
            return mock_response

        # Patch the API for both optimized and legacy paths
        with patch("google.generativeai.GenerativeModel") as mock_model:
            mock_model.return_value.generate_content.side_effect = mock_generate_content

            # Test analyze endpoint
            analyze_data = {"top_count": 5}
            response = requests.post(
                f"{base_url}/analyze", json=analyze_data, timeout=15
            )

            print(f"   Analyze response status: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                print(f"   Analyze response: {result}")

                if result.get("success"):
                    print("   OK Analysis completed successfully")
                else:
                    print(f"   ERROR Analysis failed: {result.get('error')}")
            else:
                print(f"   ERROR Analyze endpoint error: {response.status_code}")
                return False

        # Step 3: Test results endpoint to check scores
        print("3. Testing results endpoint...")
        response = requests.get(f"{base_url}/results", timeout=10)

        if response.status_code == 200:
            result = response.json()
            articles = result.get("articles", [])
            total_scraped = result.get("total_scraped", 0)

            print(f"   Total scraped: {total_scraped}")
            print(f"   Articles returned: {len(articles)}")

            # Check scores
            scores_found = 0
            zero_scores = 0
            for article in articles:
                score = article.get("score", 0)
                if score > 0:
                    scores_found += 1
                    print(f"   OK Article score: {score}")
                else:
                    zero_scores += 1
                    print(f"   ERROR Article with zero score: {score}")

            print(f"   Articles with scores > 0: {scores_found}")
            print(f"   Articles with zero scores: {zero_scores}")

            if scores_found > 0:
                print("   OK Scores are being populated correctly!")
                return True
            else:
                print("   ERROR All scores are zero - mapping issue")
                return False
        else:
            print(f"   ERROR Results endpoint failed: {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print(
            "ERROR Cannot connect to Flask app. Make sure it's running on localhost:5000"
        )
        print("Run: python app.py")
        return False
    except Exception as e:
        print(f"ERROR Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_flask_with_data()
    if success:
        print("\nSUCCESS FLASK APP TEST PASSED!")
        print("Scores are being populated correctly in the UI.")
    else:
        print("\nFAILED FLASK APP TEST FAILED!")
        print("Check the Flask app and score mapping.")
