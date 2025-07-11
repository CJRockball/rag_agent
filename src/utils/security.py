import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI


def validate_api_key():
    """Validate Google API key before app startup"""
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        st.error(
            "❌ Google API key is not configured. \
                Please set GOOGLE_API_KEY environment variable."
        )
        st.stop()

    if not api_key.startswith("AIza"):
        st.error(
            "❌ Invalid API key format. \
                Google API keys should start with 'AIza'."
        )
        st.stop()

    try:
        # Test API connectivity with minimal request
        test_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", temperature=0
        )
        test_llm.invoke("test")
        st.success("✅ API key validated successfully")
    except Exception as e:
        st.error(f"❌ API key validation failed: {str(e)}")
        st.stop()
