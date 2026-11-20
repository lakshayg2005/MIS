"""Greetings, thanks and 'what can you do?' are answered by plain code: no LLM, no policy lookup, no citations.

Why code and not the LLM: it is instant and free, it cannot be steered (there is nothing to inject into), and it keeps
the assistant scoped to transport-policy questions. Only SHORT messages that are *entirely* a pleasantry match; anything
with real content ("hi, can I book Saturday?") goes through the normal pipeline.
"""
import re

_GREET = re.compile(r"^(hi+|hello+|hey+|hiya|heya|namaste|hola|good (morning|afternoon|evening)|greetings)( (there|team|assistant|bot|everyone))?$")
_THANKS = re.compile(r"^((ok|okay|great|cool|nice)[ ,]+)?(thanks|thank you|thx|ty|thank you (so much|very much)|thanks a lot)( (a lot|so much))?$")
_BYE = re.compile(r"^((ok|okay)[ ,]+)?(bye|goodbye|see you|see ya|good night)$")
_HELP = re.compile(r"^(help|help me|what can you do|what do you do|who are you|what are you|how does this work|how do i use this|menu)$")

HELP_TEXT = ("I answer questions about your company's transport policy, such as night-drop escorts, booking cut-offs, "
             "weekend and holiday cabs, airport trips, guests, cost caps and no-show penalties. Every answer shows the "
             "policy section it comes from. Try: \"After what time is a security escort mandatory for women on drops?\"")


def smalltalk_reply(question: str):
    """-> reply text, or None when the message is a real question."""
    q = re.sub(r"[^a-z ]+", " ", question.lower())
    q = " ".join(q.split())
    if not q or len(q) > 40:
        return None
    if _GREET.match(q):
        return "Hello! " + HELP_TEXT
    if _THANKS.match(q):
        return "You're welcome! Ask me anything else about your transport policy."
    if _BYE.match(q):
        return "Goodbye! Come back any time you have a transport policy question."
    if _HELP.match(q):
        return HELP_TEXT
    return None
