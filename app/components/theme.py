"""Executive UI Theme & Mobile/Laptop Responsive Styling for NEXUS-9."""

import streamlit as st

def apply_custom_theme():
    """Inject high-fidelity responsive CSS for mobile phones, tablets, and laptops."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            -webkit-tap-highlight-color: transparent;
        }

        /* Responsive Hero Banner */
        .nexus-hero {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.95) 100%);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 16px;
            padding: 24px 32px;
            margin-bottom: 24px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.35);
            backdrop-filter: blur(14px);
        }

        .nexus-title {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #60a5fa 0%, #a855f7 50%, #ec4899 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
            line-height: 1.2;
        }

        .nexus-subtitle {
            font-size: 1.05rem;
            color: #94a3b8;
            margin-bottom: 16px;
            line-height: 1.5;
        }

        /* Metric Cards */
        .metric-card {
            background: rgba(30, 41, 59, 0.55);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 18px 22px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            transition: transform 0.2s ease, border-color 0.2s ease;
            height: 100%;
        }

        .metric-card:hover {
            transform: translateY(-2px);
            border-color: rgba(168, 85, 247, 0.45);
        }

        .metric-label {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94a3b8;
            font-weight: 600;
            margin-bottom: 6px;
        }

        .metric-value {
            font-size: 1.85rem;
            font-weight: 700;
            color: #f8fafc;
        }

        .metric-badge {
            display: inline-block;
            font-size: 0.75rem;
            padding: 3px 10px;
            border-radius: 12px;
            font-weight: 600;
            margin-top: 6px;
        }

        .badge-success { background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.4); }
        .badge-warning { background: rgba(234, 179, 8, 0.2); color: #facc15; border: 1px solid rgba(234, 179, 8, 0.4); }
        .badge-danger { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
        .badge-info { background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); }

        /* Feature Cards */
        .feature-card {
            background: rgba(30, 41, 59, 0.45);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 22px;
            height: 100%;
            transition: all 0.25s ease;
        }

        .feature-card:hover {
            border-color: rgba(96, 165, 250, 0.5);
            background: rgba(30, 41, 59, 0.75);
            transform: translateY(-2px);
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        /* ==================================================================== */
        /* BUTTON STYLING: Perfectly working on both Mobile and Laptop         */
        /* ==================================================================== */
        div.stButton > button, div.stDownloadButton > button {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            border-radius: 12px !important;
            padding: 10px 22px !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            letter-spacing: 0.02em !important;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35) !important;
            min-height: 44px !important;
            cursor: pointer !important;
        }

        div.stButton > button:hover, div.stDownloadButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(124, 58, 237, 0.5) !important;
            border-color: rgba(255, 255, 255, 0.4) !important;
            background: linear-gradient(135deg, #4338ca 0%, #6d28d9 100%) !important;
        }

        div.stButton > button:active, div.stDownloadButton > button:active {
            transform: translateY(0px) !important;
            box-shadow: 0 2px 8px rgba(79, 70, 229, 0.4) !important;
        }

        /* ==================================================================== */
        /* MOBILE VIEWPORT ADJUSTMENTS (Max Width 768px - Phones & Small Tablets)*/
        /* ==================================================================== */
        @media (max-width: 768px) {
            /* Full-width touch-optimized buttons on phones */
            div.stButton > button, div.stDownloadButton > button {
                width: 100% !important;
                padding: 14px 18px !important;
                font-size: 1rem !important;
                margin-top: 8px !important;
                margin-bottom: 8px !important;
                min-height: 48px !important;
            }

            .nexus-hero {
                padding: 16px 18px !important;
                border-radius: 14px !important;
            }

            .nexus-title {
                font-size: 1.5rem !important;
            }

            .nexus-subtitle {
                font-size: 0.88rem !important;
                margin-bottom: 12px !important;
            }

            .metric-card {
                padding: 14px 16px !important;
                margin-bottom: 10px !important;
            }

            .metric-value {
                font-size: 1.45rem !important;
            }

            .feature-card {
                padding: 16px !important;
                margin-bottom: 12px !important;
            }

            /* Responsive tabs */
            button[data-baseweb="tab"] {
                padding: 8px 12px !important;
                font-size: 0.85rem !important;
            }
        }

        /* ==================================================================== */
        /* LAPTOP & DESKTOP ADJUSTMENTS (Min Width 769px)                       */
        /* ==================================================================== */
        @media (min-width: 769px) {
            .nexus-hero {
                padding: 28px 36px !important;
            }

            .nexus-title {
                font-size: 2.25rem !important;
            }

            .metric-value {
                font-size: 1.85rem !important;
            }

            div.stButton > button, div.stDownloadButton > button {
                width: auto !important;
            }
        }
    </style>
    """, unsafe_allow_html=True)

def render_hero_banner(title: str, subtitle: str, badge_text: str = "PRODUCTION-READY v0.1.0"):
    """Render sleek hero header."""
    apply_custom_theme()
    st.markdown(f"""
    <div class="nexus-hero">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;">
            <div class="status-pill">
                <span style="width: 8px; height: 8px; background: #34d399; border-radius: 50%; display: inline-block;"></span>
                SYSTEM OPERATIONAL
            </div>
            <span class="metric-badge badge-info">{badge_text}</span>
        </div>
        <div class="nexus-title">{title}</div>
        <div class="nexus-subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)
