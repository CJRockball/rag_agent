# src/utils/rate_limit.py
import time
import streamlit as st

WINDOW = 60  # seconds
MAX_REQUEST = 10  # questions per window


def allow_request() -> bool:
    """Return True if the user can make another request in\
        the current window."""
    now = time.time()

    # first visit – initialise counters
    if "rl_start" not in st.session_state:
        st.session_state.rq_count = 0
        st.session_state.rl_start = now

    window_elapsed = now - st.session_state.rl_start

    # window expired → reset
    if window_elapsed > WINDOW:
        st.session_state.rq_count = 0
        st.session_state.rl_start = now
        return True

    # still in window → check quota
    if st.session_state.rq_count < MAX_REQUEST:
        return True

    return False


def record_request() -> None:
    """Increment the per-session counter after a successful call."""
    st.session_state.rq_count += 1
