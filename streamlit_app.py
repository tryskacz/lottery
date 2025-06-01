import streamlit as st
import random
from ai_tip import vytvor_ai_tip, nacti_data
from collections import Counter

st.title("🎯 AI Tipovač - Sportka")

pocet_sad = st.slider("Kolik tipů chceš vygenerovat?", min_value=1, max_value=10, value=5)
if st.button("Vygeneruj tipy"):
    tahy = nacti_data()
    tipy = [vytvor_ai_tip(tahy) for _ in range(pocet_sad)]
    for i, tip in enumerate(tipy):
        st.success(f"Tip {i+1}: {tip}")
    spojena = sum(tipy, [])
    ult_tip = [cislo for cislo, _ in Counter(spojena).most_common(6)]
    st.subheader("💡 Ultimátní tip")
    st.info(f"{sorted(ult_tip)}")
