from datetime import datetime
def build_digest_prompt(articles):
    formatted_articles = "\n\n".join(
        [
            f"Title: {a['title']}\nSource: {a['source']}\nContent: {a['content'][:1000]}"
            for a in articles
        ]
    )

    return f"""
    You are generating a weekly news digest.

    For each article:
    - Write 2–3 concise bullet points
    - Focus on key insights, not fluff

    Then include a short 1–2 sentence overview at the top.

    Articles:
    {formatted_articles}
    """

def generate_digest(llm_client, articles):
    prompt = build_digest_prompt(articles)
    response = llm_client.generate(prompt)
    return response

def format_digest_output(user_id, digest_text):
    return {
        "user_id": user_id,
        "generated_at": datetime.utcnow().isoformat(),
        "digest": digest_text,
    }