"""
role_manager.py — Three-role view system (Decision #15).

Roles:
  Operator        — Production floor: Digital Twin + alerts
  AI Engineer     — ML analysis: XAI, RCA, Model Forge, Drift
  Quality Manager — Compliance: ISO 9001, Reports, RCA summary

Usage in any page:
    from role_manager import render_role_selector, page_allowed
    render_role_selector()          # adds role pill to sidebar
    if not page_allowed("XAI Lab"): # gate content if needed
        st.stop()
"""
import streamlit as st

ROLES = {
    "👷 Operator": {
        "color":       "#00BFFF",
        "description": "Production floor — monitors live machine state",
        "pages":       {"Digital Twin", "Smart Reports", "Cycle History"},
        "can_train":   False,
    },
    "🤖 AI Engineer": {
        "color":       "#00FFCC",
        "description": "ML engineer — analyses models and XAI explanations",
        "pages":       {"Model Forge", "XAI Lab", "RCA Investigator",
                        "Drift Monitor", "Cycle History"},
        "can_train":   True,
    },
    "📋 Quality Manager": {
        "color":       "#FFA500",
        "description": "QA — oversees compliance and shift reporting",
        "pages":       {"ISO 9001 Dashboard", "Smart Reports",
                        "RCA Investigator", "Drift Monitor", "Cycle History"},
        "can_train":   False,
    },
}

DEFAULT_ROLE = "🤖 AI Engineer"


def get_role() -> str:
    return st.session_state.get("role", DEFAULT_ROLE)


def render_role_selector() -> str:
    """Render role selector in the sidebar. Returns the active role name."""
    with st.sidebar:
        st.markdown("---")
        role = st.selectbox(
            "👤 Active Role",
            list(ROLES.keys()),
            index=list(ROLES.keys()).index(
                st.session_state.get("role", DEFAULT_ROLE)
            ),
            key="role",
        )
        cfg = ROLES[role]
        st.markdown(
            f"<div style='padding:6px 12px; border-radius:12px; "
            f"background:{cfg['color']}22; border:1px solid {cfg['color']}; "
            f"color:{cfg['color']}; font-size:0.8em;'>"
            f"{cfg['description']}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")
    return role


def page_allowed(page_name: str) -> bool:
    """Return True if the current role can access page_name."""
    role = get_role()
    return page_name in ROLES.get(role, {}).get("pages", set())


def render_access_gate(page_name: str) -> bool:
    """
    Show a role-appropriate banner. If the page is outside the role's
    scope, show a warning banner (but don't hard-block — thesis demo mode).
    Returns True if fully allowed, False if outside role scope.
    """
    role = get_role()
    cfg  = ROLES.get(role, {})
    allowed = page_name in cfg.get("pages", set())

    if not allowed:
        st.warning(
            f"**{role}** role is primarily focused on other modules. "
            f"This page is available to: "
            f"{', '.join(r for r, c in ROLES.items() if page_name in c['pages'])}.",
            icon="⚠️",
        )
    else:
        st.sidebar.markdown(
            f"<div style='font-size:0.75em; color:{cfg['color']};'>"
            f"✓ This page is in your role's workflow</div>",
            unsafe_allow_html=True,
        )
    return allowed
