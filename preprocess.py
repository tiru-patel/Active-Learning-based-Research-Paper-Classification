import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import warnings
from re import sub, findall, I, MULTILINE
from nltk.tokenize import sent_tokenize
warnings.filterwarnings("ignore") 
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('wordnet', 'punkt')
from nltk.stem import WordNetLemmatizer

filter_chars = {'¥', '©', '¬', '®', '°', '±', '¼', 'Á', 'Å', 'Æ', 'Ç', 'É', 'Ñ', 'Ó', 'Ö', '×', 'ß', 'à', 'á', 'â', 'ã',
                'ä', 'å', 'æ', 'ç', 'è', 'é', 'ê', 'ë', 'í',
                'î', 'ï', 'ñ', 'ò', 'ó', 'ô', 'õ', 'ö', 'ø', 'ú', 'ü', 'ý', 'ā', 'ą', 'ć', 'Č', 'č', 'ē', 'ĕ', 'ė', 'ě',
                'ğ', 'ħ', 'ı', 'ĺ', 'Ł', 'ł', 'ń', 'ņ',
                'ň', 'ō', 'Ř', 'І', 'В', 'С', 'ř', 'ś', 'Ş', 'ş', 'Š', 'š', 'ť', 'ū', 'ŭ', 'ů', 'ų', 'Ż', 'Ž', 'ž', 'ǧ',
                'ǫ', 'ș', 'ə', 'ˆ', 'ˇ', '˙', '˜', '́', '̂', '̃', '̄',
                '̈', 'Γ', 'Δ', 'Θ', 'Λ', 'Π', 'Σ', 'Φ', 'Χ', 'Ψ', 'Ω', 'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'κ', 'λ',
                'μ', 'ν', 'ξ', 'π', 'ρ', 'σ', 'τ', 'υ',
                'φ', 'χ', 'ψ', 'ω', 'ϑ', 'ϕ', 'ϱ', 'ϵ', 'ḯ', '‖', '†', '…', '‰', '′', '″', '€', '⃖', '⃗', 'ℓ', 'ℜ', '™',
                '←', '↑', '→', '↓', '↦', '⇀', '⇒', '⇔',
                '⇢', '∀', '∂', '∃', '∅', '∆', '∇', '∈', '∉', '∏', '∑', '∘', '∙', '∝', '∞', '∠', '∣', '∥', '∧', '∨', '∩',
                '∪', '∫', '∭', '∼', '≃', '≅', '≈', '≔', '≜',
                '≠', '≡', '≤', '≥', '≪', '≫', '≲', '≳', '⊂', '⊆', '⊕', '⊖', '⊗', '⊙', '⋀', '⋁', '⋂', '⋃', '⋅', '⋆', '⋒',
                '⋮', '⋯', '⌊', '⌋', '□', '△', '▽', '♯',
                '✔', '➔', '⟶', '⟹', '⩽', '⩾', '⪰', '〈', '〉', '丙', '东', '作', '六', '务', '印', '厂', '合', '塘', '大',
                '子', '宁', '宅', '宝', '宫', '尚', '局',
                '峰', '府', '建', '承', '汪', '浦', '船', '药', '路', '辰', '铜', '食', '︷', '︸', '＋', '−', '{', '}',
                'ˆ', ':', '[', ']', '+', '=', '*', '<', '>', '^', '/', '-', '–', '&', '#'}


def filter_text(txt: str, keep_parenthesis=False) -> str:
    # Remove links, if any
    txt = sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*', '', txt, flags=MULTILINE)

    # Remove references, if any
    txt = sub(r'\s\([A-Z][a-z]+,\s[A-Z][a-z]?\.[^\)]*,\s\d{4}\)', '', txt)

    # Replace multiple spaces with a single space
    txt = sub(r' +', ' ', txt, flags=I)

    # '. singlelowercasechar' -> '.singlelowercasechar'
    txt = sub(r'(?<=[\.])\s+(?=(?:[a-z|[0-9]))', '', txt)

    # Converting patterns: ab.a -> ab a,   b.1 -> b 1,   A.F -> A F,   word.something -> word something
    for pattern in findall(r'[a-zA-Z]\.[a-zA-Z0-9]', txt):
        txt = txt.replace(pattern, f"{pattern.split('.')[0]} {pattern.split('.')[1]}")

    if not keep_parenthesis:
        # Removing parenthesis
        final_txt = sub(r'\(.*?\)', ' ', txt)
        
        punctuation = "\"#$&\'()-/:;@[\\]_`~'“”´ʼ‘’{|}+*^=><−"

    else:
        # Removing parenthesis with no alpha numeric character except (abcijkmnpqrtxy)
        final_txt = sub(r'\(([0-9+-/*^><=&$#@%.,!{} abcijkmnpqrtxyABCIJKMNPQRTXY]*)\)', ' ', txt)
        
        punctuation = "\"#$&\'-/:;@\\_`~'“”´ʼ‘’{|}[]+*^=><−"

    # Removes punctuation
    final_txt = ''.join([c if c not in punctuation and c not in filter_chars else ' ' for c in final_txt])
    # Removing extra spaces
    final_txt = sub(r' +', ' ', final_txt, flags=I)
    return final_txt

def lemma(df, col):
    tokens = df[col].apply(lambda x: x.split())

    lemmatizer = WordNetLemmatizer()
    tokens = tokens.apply(lambda x:[lemmatizer.lemmatize(i) for i in x])

    for i in range(len(tokens)):
        tokens[i] = " ".join(tokens[i])

    df[col] = tokens
    return df[col]

def preprocess_text(df, col):
    
    df[col] = df[col].str.replace('[^a-zA-Z]'," ", regex=True)
    # converting the text to lowercase
    df[col] = df[col].str.lower()
    # running the code for necessary pre processing
    df[col] = np.vectorize(filter_text)(df[col])
    df[col] = lemma(df, col)

    return df[col]