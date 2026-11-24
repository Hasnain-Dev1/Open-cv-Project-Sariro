import cv2
import numpy as np
import streamlit as st
from detector import analyze_hosted_photo, load_known_profiles

st.set_page_config(page_title="Python Camera Demo", page_icon="📷")
st.title("Python Camera Demo")
st.write("Take a photo with your camera to draw face boxes and show labels underneath.")

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