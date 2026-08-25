import re

def convert_to_competition_orthography(text: str) -> str:
    """
    Converts Nahuatl transcription to standard competition orthography (k/u/s)
    while preserving pure Spanish vocabulary words.
    """
    if not isinstance(text, str):
        return ""
        
    spanish_keep = {"que", "qui", "como", "quando", "porque", "quien", "con", "para", "por", "casa", "cosa", "de", "la", "el"}
    words = text.split()
    converted_words = []
    
    for word in words:
        w_lower = word.lower()
        if w_lower in spanish_keep:
            converted_words.append(word)
            continue
        
        w = word
        # 1. /k/ -> 'k'
        w = re.sub(r'qu([ei])', r'k\1', w, flags=re.IGNORECASE)
        w = re.sub(r'c([aou])', r'k\1', w, flags=re.IGNORECASE)
        w = re.sub(r'c\b', 'k', w, flags=re.IGNORECASE)
        
        # 2. /w/ -> 'u'
        w = re.sub(r'hu', 'u', w, flags=re.IGNORECASE)
        w = re.sub(r'\bw', 'u', w, flags=re.IGNORECASE)
        
        # 3. /s/ -> 's'
        w = re.sub(r'z', 's', w, flags=re.IGNORECASE)
        w = re.sub(r'c([ei])', r's\1', w, flags=re.IGNORECASE)
        
        converted_words.append(w)
        
    return " ".join(converted_words)
