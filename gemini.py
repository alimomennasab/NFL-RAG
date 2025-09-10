from google import genai
import dotenv

dotenv.load_dotenv()

GEMINI_API_KEY = dotenv.get_key(dotenv.find_dotenv(), "GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

response = client.models.generate_content(
    model="gemini-2.5-flash", contents="Explain how AI works in a few words"
)
print(response.text)