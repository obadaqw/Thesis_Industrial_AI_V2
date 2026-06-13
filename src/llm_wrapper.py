import os
from dotenv import load_dotenv
from groq import Groq
from config import get_secret

load_dotenv()


class LLMWrapper:
    def __init__(self):
        print("🤖 Initializing LLM Wrapper (Groq Adapter)...")

        self.api_key = get_secret("GROQ_API_KEY")

        self.client = None
        self.use_real_ai = False

        if self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
                self.use_real_ai = True
                print("   ✅ Connected to Groq Cloud")
            except Exception as e:
                print(f"   ⚠️ Client Init Failed: {e}. Switching to Mock Mode.")
                self.use_real_ai = False
        else:
            print("   ⚠️ No API Key found. Using Mock Mode.")

    def generate_smart_report(self, rca_diagnosis, oee_stats, cf_result=None):
        """
        Generates a technical shift report using Llama 3.3 on Groq.

        rca_diagnosis : dict from RCASurrogate.analyze_cycle  (z-score anomalies)
        oee_stats     : dict with key 'oee'
        cf_result     : optional dict from CounterfactualRCA.analyze (counterfactual)
        """
        critical_issues = [x['component'] for x in rca_diagnosis.get('critical_anomalies', [])]
        health_score    = rca_diagnosis.get('system_health_score', 100)
        oee_score       = oee_stats.get('oee', 0)

        # Build counterfactual section if available
        cf_section = ""
        if cf_result and cf_result.get('status') == 'resolved':
            tier = cf_result['tier']
            adjustments = cf_result.get('adjustments', [])
            vok = cf_result.get('validator_ok', False)
            adj_lines = "\n".join(
                f"  {a['direction']} {a['feature']}: {a['current']} → {a['suggested']} (Δ={a['delta']})"
                for a in adjustments[:5]
            )
            cf_section = f"""
[COUNTERFACTUAL RCA — Tier {tier}]:
  Validator Confirmed: {'YES' if vok else 'NO'}
  Parameter Adjustments:
{adj_lines}
"""
        elif cf_result and cf_result.get('status') == 'escalate':
            cf_section = "[COUNTERFACTUAL RCA]: Tier 3 — structurally uncorrectable. Manual inspection required.\n"

        prompt = f"""
Act as a Senior Process Engineer in an Injection Molding factory.
Analyze the following machine telemetry and generate a formal shift report.

[SYSTEM HEALTH]: {health_score}%
[OEE SCORE]: {oee_score:.2%}
[CRITICAL SENSOR FAULTS]: {", ".join(critical_issues) if critical_issues else "None"}
{cf_section}
Tasks:
1. Write a concise executive summary (2-3 sentences).
2. List 3 highly technical, actionable corrective actions using the counterfactual adjustments above where available.
3. State validator confirmation status where relevant.
4. If Health > 90% and no faults, confirm normal operation briefly.
Use professional industrial terminology. Be specific about parameter values.
"""

        # 3. Call AI
        if self.use_real_ai and self.client:
            try:
                # UPDATED MODEL NAME: Llama 3.3 70B (Versatile)
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    model="llama-3.3-70b-versatile",
                )
                return chat_completion.choices[0].message.content

            except Exception as e:
                print(f"   ❌ Groq Error: {e}")
                print("   ⚠️ Falling back to Mock.")
                return self._mock_response(health_score, critical_issues)
        else:
            return self._mock_response(health_score, critical_issues)

    def _mock_response(self, health, issues):
        """Backup Generator."""
        if health >= 90:
            return f"**OFFLINE REPORT**\n✅ System Optimal ({health}% Health). No actions needed."
        else:
            return f"**OFFLINE REPORT**\n⚠️ Anomalies Detected: {issues}. Inspect immediately."


if __name__ == "__main__":
    # Test
    llm = LLMWrapper()
    print("\n💬 Asking Llama 3.3...")
    try:
        report = llm.generate_smart_report(
            {"system_health_score": 45, "critical_anomalies": [{"component": "Hydraulic Pump"}]},
            {"oee": 0.60}
        )
        print("\n" + "=" * 40)
        print(report)
        print("=" * 40 + "\n")
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")