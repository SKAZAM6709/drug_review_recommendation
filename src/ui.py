import streamlit as st


def apply_theme():
    st.markdown(
        """
        <style>
            .stApp {
                background: #F4F7FB;
                color: #172B4D;
            }

            .block-container {
                max-width: 1250px;
                padding-top: 2rem;
                padding-bottom: 3rem;
            }

            [data-testid="stSidebar"] {
                background: #102A43;
            }

            [data-testid="stSidebar"] * {
                color: #F0F6FC;
            }

            [data-testid="stSidebar"] a:hover {
                background: #23445E;
            }

            h1, h2, h3 {
                color: #102A43;
            }

            [data-testid="stMetric"] {
                background: #FFFFFF;
                border: 1px solid #DCE5EF;
                border-top: 4px solid #168A8A;
                border-radius: 14px;
                padding: 20px;
                box-shadow: 0 4px 14px rgba(16, 42, 67, 0.05);
            }

            [data-testid="stMetricLabel"] {
                color: #52657A;
            }

            [data-testid="stMetricValue"] {
                color: #102A43;
            }

            div.stButton > button,
            div.stDownloadButton > button,
            div.stFormSubmitButton > button {
                background: #176B75;
                color: #FFFFFF;
                border: none;
                border-radius: 9px;
            }

            div.stButton > button:hover,
            div.stDownloadButton > button:hover,
            div.stFormSubmitButton > button:hover {
                background: #102A43;
                color: #FFFFFF;
            }

            .hero {
                background: linear-gradient(120deg, #102A43, #176B75);
                border-radius: 20px;
                padding: 36px;
                margin-bottom: 22px;
            }

            .hero-label {
                color: #B8F0E5;
                font-size: 13px;
                font-weight: 700;
                letter-spacing: 2px;
                margin-bottom: 12px;
            }

            .hero h1 {
                color: #FFFFFF;
                font-size: clamp(28px, 4vw, 42px);
                line-height: 1.2;
                margin-bottom: 14px;
            }

            .hero p {
                color: #E0EDF2;
                font-size: 17px;
                line-height: 1.6;
                max-width: 800px;
                margin: 0;
            }

            .feature-card {
                background: #FFFFFF;
                border: 1px solid #DCE5EF;
                border-radius: 14px;
                padding: 22px;
                margin-bottom: 10px;
            }

            .feature-card h3 {
                font-size: 20px;
                margin: 0 0 10px;
                color: #102A43;
            }

            .feature-card p {
                color: #52657A;
                font-size: 15px;
                line-height: 1.6;
                margin: 0;
            }

            .notice {
                background: #E8F5F3;
                border-left: 4px solid #168A8A;
                border-radius: 8px;
                padding: 15px 18px;
                color: #164E52;
                margin-bottom: 24px;
                line-height: 1.6;
            }

            .footer {
                border-top: 1px solid #DCE5EF;
                margin-top: 30px;
                padding-top: 16px;
                color: #52657A;
                font-size: 13px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )