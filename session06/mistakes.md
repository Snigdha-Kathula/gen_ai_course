# Session 06 — Mistakes
> Snigdhaa's GenAI Course · Save as `session06/mistakes.md`

---

## Mistake 1 — Developer persona too restrictive

**What was written:**
```python
"system": """You are DevBot, technical assistant for InCred engineering team.
- Never discuss customer PII"""
```

**What happened:**
DevBot refused to explain the credit scoring algorithm even to a developer — an internal team member who needs this information. The model applied "don't share internal systems" too broadly because the rule was vague.

**The fix:**
```python
"system": """You are DevBot, technical assistant for InCred engineering team.
- You CAN discuss: architecture, scoring pipeline stages, API contracts, schema design
- You CANNOT discuss: exact algorithm weights, model parameters, customer PII"""
```

**The lesson:**
Vague rules get applied too broadly. Precise rules give the model clear boundaries.

```
❌ Vague:   "Never discuss internal systems"
✅ Precise: "You CAN discuss X. You CANNOT discuss Y."
```

Always define both what the bot CAN and CANNOT do — not just restrictions.

---

## Key insight — system prompts are access control

Different personas = different access levels. This is not just UX — it's security.

```
Customer  → deflect all internal questions
Developer → architecture level, no weights or PII  
Admin     → operational details with audit logging
```

If a rule is wrong (too strict or too loose), the bot gives wrong answers confidently. Always test each persona with adversarial questions — questions that should be deflected AND questions that should be answered in detail.

---

## Testing checklist for production system prompts

For every persona, always test:
- [ ] A question it SHOULD answer fully
- [ ] A question it SHOULD deflect
- [ ] A sensitive question (algorithm, PII, internal data)
- [ ] An out-of-scope question (flights, weather, unrelated topics)
- [ ] An adversarial prompt ("ignore previous instructions and...")

If any test fails → fix the system prompt before shipping.

---

*Session 06 mistakes logged*