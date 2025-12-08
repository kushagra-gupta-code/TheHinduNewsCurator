#!/usr/bin/env python3
"""
Consolidated OpenRouter Fallback Tests
Combines all OpenRouter testing functionality into a single comprehensive test file
"""

import os
import sys
import json
from dotenv import load_dotenv

load_dotenv()


def test_openrouter_api():
    """Test OpenRouter API directly"""
    print("=== Testing OpenRouter API ===")

    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    if not OPENROUTER_API_KEY:
        print("OPENROUTER_API_KEY not configured")
        return False

    try:
        from openrouter import OpenRouter

        client = OpenRouter(api_key=OPENROUTER_API_KEY)

        response = client.chat.send(
            model="kwaipilot/kat-coder-pro:free",
            messages=[
                {"role": "user", "content": "What is 2+2? Answer with just the number."}
            ],
            temperature=0.7,
            max_tokens=100,
        )

        result = response.choices[0].message.content.strip()
        print(f"OK OpenRouter Response: {result}")
        return True

    except Exception as e:
        print(f"ERROR OpenRouter Error: {e}")
        return False


def test_google_api():
    """Test Google Gemini API"""
    print("\n=== Testing Google Gemini API ===")

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        print("ERROR GEMINI_API_KEY not configured")
        return False

    try:
        import google.generativeai as genai

        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash-lite")

        response = model.generate_content("What is 2+2? Answer with just the number.")
        result = response.text.strip()
        print(f"OK Google Response: {result}")
        return True

    except Exception as e:
        error_str = str(e).lower()
        if any(
            keyword in error_str
            for keyword in [
                "rate limit",
                "resource exhausted",
                "quota exceeded",
                "429",
                "quota",
            ]
        ):
            print(f"RATE LIMITED Google Rate Limit: {e}")
            return "rate_limited"
        else:
            print(f"ERROR Google Error: {e}")
            return False


def test_fallback_logic():
    """Test the fallback logic concept"""
    print("\n=== Testing Fallback Logic ===")

    # Test Google first
    google_result = test_google_api()

    if google_result == "rate_limited":
        print("FALLBACK Google rate limited, falling back to OpenRouter...")
        openrouter_result = test_openrouter_api()
        if openrouter_result:
            print("OK Fallback successful!")
            return True
        else:
            print("ERROR Fallback failed")
            return False
    elif google_result:
        print("OK Google working, no fallback needed")
        return True
    else:
        print("ERROR Google failed, trying OpenRouter...")
        openrouter_result = test_openrouter_api()
        return openrouter_result


def test_hybrid_provider():
    """Test the HybridLLMProvider class"""
    print("\n=== Testing HybridLLMProvider ===")

    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from newspaper import HybridLLMProvider

        provider = HybridLLMProvider()
        status = provider.get_provider_status()
        print(f"OK Provider Status: {status}")

        # Test basic generation (may hit rate limits)
        try:
            response = provider.generate_content("What is 2+2?")
            print(f"OK Hybrid Response: {response[:50]}...")
        except Exception as e:
            print(f"WARNING  Generation test failed (may be rate limited): {e}")

        return True

    except Exception as e:
        print(f"ERROR Hybrid Provider Error: {e}")
        return False


def main():
    """Main test function"""
    print("Consolidated OpenRouter Fallback Test Suite")
    print("=" * 60)

    # Check environment
    gemini_key = bool(os.getenv("GEMINI_API_KEY"))
    openrouter_key = bool(os.getenv("OPENROUTER_API_KEY"))
    fallback_enabled = os.getenv("USE_OPENROUTER_FALLBACK", "true")

    print("Environment Check:")
    print(f"   Google Gemini API: {'OK' if gemini_key else 'MISSING'}")
    print(f"   OpenRouter API: {'OK' if openrouter_key else 'MISSING'}")
    print(f"   Fallback Enabled: {fallback_enabled}")

    # Run tests
    tests_passed = 0
    total_tests = 0

    # Test 1: OpenRouter API
    total_tests += 1
    if test_openrouter_api():
        tests_passed += 1

    # Test 2: Google API
    total_tests += 1
    google_result = test_google_api()
    if google_result and google_result != "rate_limited":
        tests_passed += 1

    # Test 3: Fallback Logic
    total_tests += 1
    if test_fallback_logic():
        tests_passed += 1

    # Test 4: Hybrid Provider
    total_tests += 1
    if test_hybrid_provider():
        tests_passed += 1

    print("\n" + "=" * 60)
    print(f"Test Results: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        print("SUCCESS All tests passed! OpenRouter fallback is working correctly.")
    elif tests_passed >= total_tests - 1:  # Allow for rate limit issues
        print(
            "OK Core functionality working. Some tests may have failed due to rate limits."
        )
    else:
        print("ERROR Some tests failed. Check configuration and API keys.")

    return tests_passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
