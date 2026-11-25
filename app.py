import time
import cv2
import numpy as np
import streamlit as st
from detector import analyze_hosted_photo, load_known_profiles

st.set_page_config(page_title="Python Camera Demo", page_icon="🚨")
st.title("Python Camera Demo")
st.write("Take a photo with your camera to draw face boxes and show labels underneath.")

ALERT_TEXT = (
    " A kidnaper has been identified or spotted, sending his photo and its "
    "live location to the police department."
)

# Animated template styling for the police dispatch banner
st.markdown(
    """
    <style>
    @keyframes pulseAlert {
        0% { transform: scale(0.98); box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.55); }
        70% { transform: scale(1); box-shadow: 0 0 0 12px rgba(220, 38, 38, 0); }
        100% { transform: scale(0.98); box-shadow: 0 0 0 0 rgba(220, 38, 38, 0); }
    }
    @keyframes slideFadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .police-template {
        background: linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%);
        border: 2px solid #dc2626;
        border-left: 8px solid #b91c1c;
        border-radius: 12px;
        padding: 16px 18px;
        margin-top: 10px;
        color: #7f1d1d;
        animation: slideFadeIn 0.35s ease-out, pulseAlert 1.8s infinite;
    }
    .police-template-header {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .police-template-meta {
        font-size: 0.88rem;
        color: #991b1b;
        margin-top: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "known_profiles" not in st.session_state:
    st.session_state.known_profiles = load_known_profiles()

picture = st.camera_input("Take a photo")

if picture is not None:
    file_bytes = np.asarray(bytearray(picture.getvalue()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if frame is None:
        st.error("Could not decode camera image.")
    else:
        status, annotated_bgr = analyze_hosted_photo(
            frame, st.session_state.known_profiles
        )
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, caption="Detection Result", use_container_width=True)

        if status == "GOOD_GUY":
            st.success("Good Guy detected.")
        elif status == "COVERED":
            st.error("Bad Guy (Covered or no clear face found).")
        else:
            st.warning("Unknown face.")

        if status != "GOOD_GUY":
            st.divider()
            send_clicked = st.button(
                "🚨 Send Alert to Police Department",
                type="primary",
                use_container_width=True,
            )

            if send_clicked:
                typed_slot = st.empty()
                template_slot = st.empty()

                # Auto-type the alert message character by character
                typed_so_far = ""
                for ch in ALERT_TEXT:
                    typed_so_far += ch
                    typed_slot.code(typed_so_far, language=None)
                    time.sleep(0.025)

                # Animated template confirming the message was sent to police
                template_slot.markdown(
                    """
                    <div class="police-template">
                        <div class="police-template-header">
                            <span>🚔</span>
                            <span>MESSAGE SENT TO POLICE DEPARTMENT</span>
                        </div>
                        <div>Status: <strong>Dispatched</strong> — Snapshot &amp; live location attached.</div>
                        <div class="police-template-meta">Recipient: Local Police Dispatch Unit • Priority: High</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )