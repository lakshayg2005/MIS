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


_ENTITIES = ("human|person|real|robot|bot|chatbot|ai|assistant|gpt|chatgpt|claude|llm|machine|computer|program|dog|cat|animal|alive|"
             "man|woman|boy|girl|male|female|bird|monkey|parrot|alien|ghost|god|robo|bhai|sir|madam")
_IDENTITY = re.compile(rf"^(are|r) (you|u) (a |an |the |really |actually )*({_ENTITIES})( or not)?$|^(is this|am i talking to) (a |an )?(real )?(human|person|bot|robot|ai|chatbot)$")
_HOW_BUILT = re.compile(r"^(how (are|were|r) (you|u) (being |getting )?(implemented|built|made|created|developed|designed|trained|coded|programmed)|"
                        r"how do (you|u) work|how does (it|this|the assistant|the bot) work|what (model|llm|ai|tech|technology) (are you|do you use|is this|are you using)|"
                        r"(what|which) (are|r) (you|u) (built|made|running|powered) (with|on|by)|who (made|built|created|developed) (you|u|this))$")
_PRIVACY = re.compile(r"^(do|will|does) (you|u|this) (store|save|remember|keep|record|log|track) (my |our |the )?(chats?|questions?|conversations?|messages?|data|history|info|information)$")
_BOOK = re.compile(r"^(can|could|will) (you|u) (please )?(book|cancel|arrange|schedule|reserve|get)( me)?( a| my)? (cab|cabs|ride|trip|taxi|shuttle|seat)$")

IDENTITY_TEXT = ("I'm an AI assistant, not a person or an animal. " + HELP_TEXT)
HOW_TEXT = ("Here's how I work: when you ask a question, I look up your company's current transport policy, pick the most relevant "
            "sections (at most 3), and a language model writes a short answer using only those sections. Before you see it, my "
            "checks make sure it cites real sections and that every number in it appears in the policy text. If the policy doesn't "
            "cover your question I say so instead of guessing. I only ever use your own company's policy.")
PRIVACY_TEXT = ("I don't keep your questions on the server. This chat is stored only in your browser tab and disappears when you close "
                "it, switch company or press New chat. Each question is answered on its own, so I don't remember earlier messages.")
BOOK_TEXT = ("I can't book or cancel cabs. I can tell you what your policy says about booking, like cut-off times, approvals and "
             "penalties. To actually book, please use your usual booking channel or the transport desk.")


def smalltalk_reply(question: str):
    """-> reply text, or None when the message is a real question."""
    q = re.sub(r"[^a-z ]+", " ", question.lower())
    q = " ".join(q.split())
    if not q or len(q) > 70:
        return None
    if _GREET.match(q):
        return "Hello! " + HELP_TEXT
    if _THANKS.match(q):
        return "You're welcome! Ask me anything else about your transport policy."
    if _BYE.match(q):
        return "Goodbye! Come back any time you have a transport policy question."
    if _HELP.match(q):
        return HELP_TEXT
    if _IDENTITY.match(q):
        return IDENTITY_TEXT
    if _HOW_BUILT.match(q):
        return HOW_TEXT
    if _PRIVACY.match(q):
        return PRIVACY_TEXT
    if _BOOK.match(q):
        return BOOK_TEXT
    return None
