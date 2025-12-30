from google import genai

client = genai.Client(api_key="AIzaSyAVyNZR_h6RL8rVFJNY7FbqB0xIq6NxNzQ")  # Please replace with your new key

response = client.models.generate_content(
    model="gemini-1.5-flash",  # or "gemini-1.5-flash", "gemini-1.5-pro"
    contents="Explain how AI works in a few words",
)

print(response.text)