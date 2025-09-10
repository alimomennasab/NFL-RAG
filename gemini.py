import dotenv
from google import genai
from embed import retrieve

dotenv.load_dotenv()
GEMINI_API_KEY = dotenv.get_key(dotenv.find_dotenv(), "GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

if __name__ == "__main__":
    query = "Who led Week 1 in sacks?"
    results = retrieve(query)
    print("top relevant chunks: ", len(results))

    combined_context = ""
    for res in results:
        print(f"Rank: {res['rank']}, Score: {res.get('score', 'N/A')}, Text: {res['text'][:100]}...")
        combined_context += f"{res['text']}\n"

    prompt = (
        "You are a factual assistant that only uses the information inside 'context' to answer questions. "
        "If the answer isn't there, say you don't know. Don't make up information. "
        f"context: {combined_context}"
        f"Question: {query}"
     )


    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt
    )
    print(response.text)

