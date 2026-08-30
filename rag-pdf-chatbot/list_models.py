from google import genai
from app.config import GOOGLE_API_KEY

client = genai.Client(api_key=GOOGLE_API_KEY)

for m in client.models.list():
    print(m.name, "-", m.supported_actions)