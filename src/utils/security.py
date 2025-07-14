import os
import streamlit as st

# from langchain_google_genai import ChatGoogleGenerativeAI


def validate_api_key():
    """Validate Google API key from environment variable"""
    # API key should now be in os.environ from startup initialization
    api_key = os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        st.error("❌ Google API key not found in environment variables")
        st.info("💡 Ensure environment is properly initialized at startup")
        st.stop()

    if not api_key.startswith("AIza"):
        st.error(
            "❌ Invalid API key format. \
            Google API keys should start with 'AIza'"
        )
        st.stop()

    # try:
    #     # Test API connectivity
    #     test_llm = ChatGoogleGenerativeAI(
    #         model="gemini-2.0-flash",
    #         temperature=0,
    #     )
    #     test_llm.invoke("test")
    #     st.success("✅ API key validated successfully")
    # except Exception as e:
    #     st.error(f"❌ API key validation failed: {str(e)}")
    #     st.stop()
