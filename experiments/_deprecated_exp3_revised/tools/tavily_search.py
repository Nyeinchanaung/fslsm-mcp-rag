"""
Web Search Tool implementation using Tavily API.
Free tier: 1,000 calls/month (sufficient for thesis).
Sign up: https://app.tavily.com
"""
import os
from tavily import TavilyClient


def web_search(query: str, max_results: int = 3) -> str:
    """Perform web search and return formatted results as plain text."""
    try:
        client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        results = client.search(query=query, max_results=max_results)
        formatted = []
        for r in results.get("results", []):
            formatted.append(
                f"- **{r['title']}**: {r['content'][:250]}\n"
                f"  Source: {r['url']}"
            )
        return "\n\n".join(formatted) if formatted else "No results found."
    except Exception as e:
        print(f"[tavily] search failed for '{query[:40]}': {e}")
        return "Search unavailable."


if __name__ == "__main__":
    result = web_search("latest transformer architecture developments 2025")
    print(result)
