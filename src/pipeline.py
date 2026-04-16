from ranking import get_top_articles_for_user
from summarization import generate_digest, format_digest_output

def run_user_digest(conn, user_id, llm_client, top_k=10):
    articles = get_top_articles_for_user(conn, user_id, top_k=top_k)
    digest_text = generate_digest(llm_client, articles)
    return format_digest_output(user_id, digest_text)