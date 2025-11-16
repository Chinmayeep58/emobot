from openai import OpenAI
import streamlit as st
import joblib
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load classifier pipeline
clf = joblib.load("emotion_classifier.pkl")

# Mapping labels to emotion names
label_map = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear"
}

# Streamlit UI
st.set_page_config(page_title="Emotion-Aware Chatbot", page_icon="🌈")
st.title("🌈 Emotion-Aware Mood Fix Chatbot 🧠")

st.write("I understand your emotions and reply warmly and positively.")

user_text = st.text_input("How are you feeling today?")

if st.button("Send"):
    if user_text.strip():

        # Emotion prediction
        pred = clf.predict([user_text])[0]
        emotion = label_map[pred]

        # Call OpenAI LLM
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"You are a short, warm, caring chatbot. "
                            f"The user feels {emotion}. "
                            f"Reply in only 1–2 sentences. Be supportive and positive."
                        )
                    },
                    {"role": "user", "content": user_text}
                ],
                max_tokens=80
            )

            # NEW API: access content like this
            bot_reply = response.choices[0].message.content

            st.markdown(f"### 💡 Detected Emotion: **_{emotion}_**")
            st.success(bot_reply)

        except Exception as e:
            st.error(f"Error contacting OpenAI: {str(e)}")
