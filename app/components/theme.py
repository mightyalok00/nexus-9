"""Executive UI Theme & Styling for NEXUS-9 Dashboard."""

import streamlit as st

def apply_custom_theme():
    """Inject modern CSS styling: glassmorphism, gradient badges, and typography."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Hero Banner */
        .nexus-hero {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 24px 32px;
            margin-bottom: 24px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(12px);
        }

        .nexus-title {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #60a5fa 0%, #a855f7 50%, #ec4899 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }

        .nexus-subtitle {
            font-size: 1.05rem;
            color: #94a3b8;
            margin-bottom: 16px;
        }

        /* Metric Cards */
        .metric-card {
            background: rgba(30, 41, 59, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 18px 22px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }

        .metric-card:hover {
            transform: translateY(-2px);
            border-color: rgba(168, 85, 247, 0.4);
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
            font-size: 1.8rem;
            font-weight: 700;
            color: #f8fafc;
        }

        .metric-badge {
            display: inline-block;
            font-size: 0.75rem;
            padding: 3px 8px;
            border-radius: 12px;
            font-weight: 600;
            margin-top: 6px;
        }

        .badge-success { background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.4); }
        .badge-warning { background: rgba(234, 179, 8, 0.2); color: #facc15; border: 1px solid rgba(234, 179, 8, 0.4); }
        .badge-danger { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
        .badge-info { background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); }

        /* Feature Card */
        .feature-card {
            background: rgba(30, 41, 59, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 20px;
            height: 100%;
            transition: all 0.25s ease;
        }

        .feature-card:hover {
            border-color: rgba(96, 165, 250, 0.5);
            background: rgba(30, 41, 59, 0.7);
        }

        /* Pill status */
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
    </style>
    """, unsafe_allow_html=True)

def render_hero_banner(title: str, subtitle: str, badge_text: str = "PRODUCTION-READY v0.1.0"):
    """Render sleek hero header."""
    apply_custom_theme()
    st.markdown(f"""
    <div class="nexus-hero">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
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
