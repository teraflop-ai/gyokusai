import re2 as re

CODE_PATTERN = re.compile(
    r"\b(?:if|else|for|while|def|class|include|switch|case|default|const|static|"
    r"try|catch|exception|continue|open|close|import|var|None|null|true|True|"
    r"false|False|print|return|sudo|apt-get|wget|\+|-|\*|/|=)\b"
    r"|[{};]|\w+\s*\(.*\)|\w+\s*=\s*\w+"
)

AI_PHRASE_PATTERN = (
    r"(?im)(?:^|[.!?]\s+)[ \t]*(?:"
    r"(?:as an?|i(?: am|['’]m) an?) "
    r"(?:(?:ai(?: language)? model|large language model|language model|ai assistant|ai chatbot)\b"
    r"|ai(?:[,.!]| i\b))"
    r"|i(?: am|['’]m) chatgpt\b"
    r"|i (?:was|have been) (?:trained|developed|created) by (?:openai|anthropic|google)\b"
    r"|(?:as of my |my )(?:last |latest )?"
    r"(?:knowledge|training)(?: data)? (?:update|cut[- ]?off)\b"
    r"|my training data (?:only )?(?:goes up to|includes information up to)\b"
    r"|i (?:don['’]t|do not) have (?:access to )?real[- ]time (?:data|information|access)\b"
    r"|i (?:don['’]t|do not) have personal "
    r"(?:opinions|beliefs|experiences|feelings|preferences)\b"
    r"|i (?:don['’]t|do not) have (?:a physical body|consciousness|emotions)\b"
    r"|(?:i(?: am|['’]m) sorry|i apologize),? but i "
    r"(?:cannot|can['’]t|am unable to|am not able to) "
    r"(?:provide|generate|create|write|fulfill|comply with|predict|browse|access)\b"
    r"|i (?:cannot|can['’]t|am unable to) "
    r"(?:fulfill|assist with|comply with) (?:this|that|your) request\b"
    r"|(?:certainly[!,]?|sure!)\s+here(?:['’]s| is| are)\b"
    r"|here(?:['’]s| is) (?:a revised|an updated|a rewritten|an improved) version of your\b"
    r"|regenerate response[ \t]*$"
    r")"
)

AI_STYLE_PATTERN = (
    r"(?i)\b(?:"
    r"it(?: is|['’]s) (?:important|worth|essential) to (?:note|remember|understand)\b"
    r"|it(?: is|['’]s) worth noting\b"
    r"|in today['’]s (?:fast[- ]paced|ever[- ]evolving|digital) world\b"
    r"|in the (?:ever[- ]evolving|ever[- ]changing|rapidly evolving) (?:landscape|world) of\b"
    r"|navigat(?:e|es|ing) the (?:complexities|intricacies|landscape) of\b"
    r"|let['’]s (?:dive|delve) (?:right )?into\b"
    r"|delv(?:e|es|ing) into\b"
    r"|without further ado\b"
    r"|in the realm of\b"
    r"|a (?:rich |vibrant |intricate )?tapestry of\b"
    r"|a testament to\b"
    r"|at the heart of\b"
    r"|unlock (?:the |your )?(?:full )?potential\b"
    r"|harness the power of\b"
    r"|pave the way for\b"
    r"|plays? a (?:crucial|pivotal|vital) role in\b"
    r"|embark on (?:a|this|your) journey\b"
    r"|a game[- ]changer\b"
    r"|a (?:beacon of|journey of discovery)\b"
    r"|the possibilities are endless\b"
    r"|only time will tell\b"
    r")"
)

AI_TRACE_PATTERN = (
    r"(?i)utm_source(?:=|%3D)(?:chatgpt|openai|perplexity|copilot|claude|gemini|grok|deepseek|mistral)"
    r"|data-message-(?:model-slug|author-role|id)="
    r'|data-testid="conversation-turn'
    r'|data-start="\d+"[^>]*data-end="\d+"'
    r"|data-is-(?:last|only)-node"
)
