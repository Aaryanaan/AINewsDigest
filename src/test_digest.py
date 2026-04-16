from ranking import get_top_articles_for_user
from summarization import generate_digest, format_digest_output
from interaction_ingestion import get_all_active_users
from src.llm_client import OpenAILLM

# Mock LLM (so you can test without API first)
class MockLLM:
    def generate(self, prompt):
        return """
Weekly Overview:
This week covers AI advancements, tech regulation, and startup funding trends.

Articles:
1. AI Breakthrough
- New model improves reasoning
- कंपनies investing heavily

2. Tech Regulation
- Governments pushing stricter policies
- Debate over innovation vs control
"""

def main(conn):
    users = get_all_active_users(conn)
    user_id = users[0].user_id
    llm = OpenAILLM()

    articles = get_top_articles_for_user(conn, user_id, top_k=5)

    print("\n=== RAW ARTICLES ===")
    for a in articles:
        print(a["title"])

    digest_text = generate_digest(llm, articles)
    output = format_digest_output(user_id, digest_text)

    print("\n=== FINAL DIGEST ===")
    print(output)

if __name__ == "__main__":
    from db import get_connection
    conn = get_connection()
    main(conn)