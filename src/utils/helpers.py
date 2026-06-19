import re
import unicodedata

def clean_vietnamese_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = "".join(char for char in text if not unicodedata.category(char).startswith("C") or char in "\n\t")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def parse_focused_answer(text: str) -> str:
    text = (text or "").strip()
    if "[TRẢ LỜI]:" in text:
        answer = text.split("[TRẢ LỜI]:")[-1].strip()
    else:
        answer = text
    # Xóa các gạch đầu dòng dư thừa
    answer = re.sub(r"^\s*[\u2022\-\*]\s*", "", answer, flags=re.MULTILINE)
    answer = re.sub(r"\n+", " ", answer).strip()
    return answer