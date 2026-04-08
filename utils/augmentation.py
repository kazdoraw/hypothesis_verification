"""
Advanced Data Augmentation for dental clinic message classification.

Provides 10+ augmentation techniques:
1. Keyboard typos (Cyrillic layout)
2. Abbreviations and slang
3. Emoji insertion
4. Case variations
5. Punctuation variations
6. Colloquial transformations
7. Word deletion
8. Word swap
9. Synonym replacement
10. Character-level noise
11. Whitespace variations
"""

import random
import re
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass


# =============================================================================
# Keyboard Layout for Typos
# =============================================================================

# Russian keyboard layout neighbors
RUSSIAN_KEYBOARD_NEIGHBORS = {
    'й': ['ц', 'ф', 'ы'],
    'ц': ['й', 'у', 'ф', 'ы', 'в'],
    'у': ['ц', 'к', 'ы', 'в', 'а'],
    'к': ['у', 'е', 'в', 'а', 'п'],
    'е': ['к', 'н', 'а', 'п', 'р'],
    'н': ['е', 'г', 'п', 'р', 'о'],
    'г': ['н', 'ш', 'р', 'о', 'л'],
    'ш': ['г', 'щ', 'о', 'л', 'д'],
    'щ': ['ш', 'з', 'л', 'д', 'ж'],
    'з': ['щ', 'х', 'д', 'ж', 'э'],
    'х': ['з', 'ъ', 'ж', 'э'],
    'ъ': ['х', 'э'],
    'ф': ['й', 'ц', 'ы'],
    'ы': ['й', 'ц', 'у', 'ф', 'в', 'а'],
    'в': ['ц', 'у', 'к', 'ы', 'а', 'п'],
    'а': ['у', 'к', 'е', 'в', 'п', 'р'],
    'п': ['к', 'е', 'н', 'а', 'р', 'о'],
    'р': ['е', 'н', 'г', 'п', 'о', 'л'],
    'о': ['н', 'г', 'ш', 'р', 'л', 'д'],
    'л': ['г', 'ш', 'щ', 'о', 'д', 'ж'],
    'д': ['ш', 'щ', 'з', 'л', 'ж', 'э'],
    'ж': ['щ', 'з', 'х', 'д', 'э'],
    'э': ['з', 'х', 'ъ', 'ж'],
    'я': ['ч', 'ф', 'ы'],
    'ч': ['я', 'с', 'ы', 'в'],
    'с': ['ч', 'м', 'в', 'а'],
    'м': ['с', 'и', 'а', 'п'],
    'и': ['м', 'т', 'п', 'р'],
    'т': ['и', 'ь', 'р', 'о'],
    'ь': ['т', 'б', 'о', 'л'],
    'б': ['ь', 'ю', 'л', 'д'],
    'ю': ['б', 'д', 'ж'],
}

# =============================================================================
# Abbreviations and Slang
# =============================================================================

ABBREVIATIONS = {
    # Greetings
    "здравствуйте": ["здрасте", "здрасьте", "здраст", "здравст"],
    "добрый день": ["дд", "добр день", "добр ден"],
    "доброе утро": ["добр утр", "утра"],
    "добрый вечер": ["добр веч", "вечера"],
    "привет": ["прив", "хай", "хей"],
    
    # Thanks
    "спасибо": ["спс", "спсб", "спасиб", "сенкс", "благодарю"],
    "спасибо большое": ["спс большое", "спс бол", "большое спс"],
    "пожалуйста": ["пжлст", "пожста", "пож", "плз"],
    
    # Confirmations
    "хорошо": ["хор", "ок", "окей", "ладно", "лады"],
    "понятно": ["понял", "ясно", "понт", "поняла"],
    "да": ["ага", "угу", "ну да", "да-да"],
    "нет": ["неа", "не", "ну нет"],
    
    # Common words
    "сейчас": ["щас", "ща", "сейч"],
    "сегодня": ["сёдня", "сегодн"],
    "завтра": ["завтр"],
    "который": ["кот", "котор"],
    "потому что": ["потомучто", "птмч", "потому шо"],
    "что": ["чё", "чо", "шо"],
    "почему": ["почем", "пчм"],
    "какой": ["какой-то", "какойто"],
    
    # Medical
    "стоматолог": ["стомат", "дантист", "зубной"],
    "консультация": ["консульт", "конс"],
    "записаться": ["записатся", "записать"],
}

# Reverse mapping for restoration
ABBREVIATIONS_REVERSE = {}
for full, abbrs in ABBREVIATIONS.items():
    for abbr in abbrs:
        ABBREVIATIONS_REVERSE[abbr] = full

# =============================================================================
# Emoji Sets
# =============================================================================

EMOJI_SETS = {
    "positive": ["😊", "🙂", "👍", "😃", "😄", "🤗", "✨"],
    "negative": ["😔", "😢", "😞", "😕", "😣", "😩"],
    "medical": ["🦷", "💊", "🏥", "💉", "🩺"],
    "question": ["❓", "🤔", "❔"],
    "urgent": ["❗", "⚠️", "🚨"],
    "neutral": ["", "", "", ""],  # High chance of no emoji
}

# =============================================================================
# Colloquial Transformations
# =============================================================================

COLLOQUIAL_TRANSFORMS = [
    ("что", "чё"),
    ("что", "чо"),
    ("сейчас", "щас"),
    ("сегодня", "сёдня"),
    ("какой", "какой-то"),
    ("который", "который-то"),
    ("вообще", "вобще"),
    ("вообще", "ваще"),
    ("потому что", "потомучто"),
    ("как будто", "какбудто"),
    ("сколько", "скока"),
    ("только", "тока"),
    ("тоже", "тож"),
    ("можно", "можн"),
    ("нужно", "нужн"),
    ("наверное", "наверн"),
    ("конечно", "конеш"),
    ("кажется", "кажись"),
    ("по-моему", "помоему"),
]

# =============================================================================
# Punctuation Variations
# =============================================================================

PUNCTUATION_ENDINGS = [
    "",       # No punctuation
    ".",      # Period
    "!",      # Exclamation
    "?",      # Question (even for statements)
    "...",    # Ellipsis
    "!!!",    # Emotional exclamation
    "!?",     # Surprise
    "?!",     # Surprised question
]

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class AugmentationConfig:
    """Configuration for augmentation."""
    typo_prob: float = 0.1
    abbreviation_prob: float = 0.3
    emoji_prob: float = 0.15
    case_prob: float = 0.2
    punctuation_prob: float = 0.25
    colloquial_prob: float = 0.2
    word_delete_prob: float = 0.1
    word_swap_prob: float = 0.1
    char_noise_prob: float = 0.05
    whitespace_prob: float = 0.1


@dataclass
class AugmentedSample:
    """Result of augmentation."""
    text: str
    original: str
    augmentation_types: List[str]
    
    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "original": self.original,
            "augmentation_types": self.augmentation_types,
        }


# =============================================================================
# Augmentation Functions
# =============================================================================

def add_typo(text: str, prob: float = 0.1) -> str:
    """
    Add keyboard typos to text.
    
    Args:
        text: Input text
        prob: Probability of typo per character
        
    Returns:
        Text with typos
    """
    result = []
    for char in text:
        if random.random() < prob and char.lower() in RUSSIAN_KEYBOARD_NEIGHBORS:
            # Get neighbor key
            neighbors = RUSSIAN_KEYBOARD_NEIGHBORS[char.lower()]
            typo_char = random.choice(neighbors)
            # Preserve case
            if char.isupper():
                typo_char = typo_char.upper()
            result.append(typo_char)
        else:
            result.append(char)
    return "".join(result)


def apply_abbreviations(text: str, prob: float = 0.3) -> str:
    """
    Replace words with abbreviations/slang.
    
    Args:
        text: Input text
        prob: Probability of replacement per word
        
    Returns:
        Text with abbreviations
    """
    text_lower = text.lower()
    
    for full, abbrs in ABBREVIATIONS.items():
        if full in text_lower and random.random() < prob:
            abbr = random.choice(abbrs)
            # Case-insensitive replacement
            pattern = re.compile(re.escape(full), re.IGNORECASE)
            text = pattern.sub(abbr, text, count=1)
    
    return text


def add_emoji(text: str, tone: str = "neutral", prob: float = 0.15) -> str:
    """
    Add emoji to text.
    
    Args:
        text: Input text
        tone: Tone category (positive, negative, medical, question, urgent, neutral)
        prob: Probability of adding emoji
        
    Returns:
        Text with emoji
    """
    if random.random() > prob:
        return text
    
    emoji_set = EMOJI_SETS.get(tone, EMOJI_SETS["neutral"])
    emoji = random.choice(emoji_set)
    
    if emoji:
        # Add at end with space
        position = random.choice(["end", "start"])
        if position == "end":
            text = text.rstrip(".,!?") + " " + emoji
        else:
            text = emoji + " " + text
    
    return text


def vary_case(text: str) -> str:
    """
    Apply case variation to text.
    
    Options: lower, upper, title, random
    """
    variation = random.choice(["lower", "upper", "title", "random", "original"])
    
    if variation == "lower":
        return text.lower()
    elif variation == "upper":
        return text.upper()
    elif variation == "title":
        return text.title()
    elif variation == "random":
        return "".join(
            c.upper() if random.random() < 0.3 else c.lower()
            for c in text
        )
    else:
        return text


def vary_punctuation(text: str) -> str:
    """
    Vary punctuation at end of text.
    """
    # Remove existing ending punctuation
    text = text.rstrip(".,!?")
    
    # Add new punctuation
    ending = random.choice(PUNCTUATION_ENDINGS)
    return text + ending


def apply_colloquial(text: str, prob: float = 0.2) -> str:
    """
    Apply colloquial transformations.
    """
    for formal, colloquial in COLLOQUIAL_TRANSFORMS:
        if formal in text.lower() and random.random() < prob:
            pattern = re.compile(re.escape(formal), re.IGNORECASE)
            text = pattern.sub(colloquial, text, count=1)
            break  # Only one transformation per call
    
    return text


def delete_random_word(text: str, prob: float = 0.1) -> str:
    """
    Delete a random word from text.
    """
    words = text.split()
    if len(words) <= 2 or random.random() > prob:
        return text
    
    # Don't delete first or last word usually
    idx = random.randint(1, len(words) - 2)
    words.pop(idx)
    return " ".join(words)


def swap_adjacent_words(text: str, prob: float = 0.1) -> str:
    """
    Swap two adjacent words.
    """
    words = text.split()
    if len(words) <= 2 or random.random() > prob:
        return text
    
    idx = random.randint(0, len(words) - 2)
    words[idx], words[idx + 1] = words[idx + 1], words[idx]
    return " ".join(words)


def add_char_noise(text: str, prob: float = 0.05) -> str:
    """
    Add character-level noise (duplicate chars, missing chars).
    """
    result = []
    for i, char in enumerate(text):
        if random.random() < prob:
            action = random.choice(["duplicate", "skip", "keep"])
            if action == "duplicate":
                result.append(char)
                result.append(char)
            elif action == "skip" and len(result) > 0:
                pass  # Skip this character
            else:
                result.append(char)
        else:
            result.append(char)
    return "".join(result)


def vary_whitespace(text: str, prob: float = 0.1) -> str:
    """
    Vary whitespace (extra spaces, missing spaces).
    """
    if random.random() > prob:
        return text
    
    action = random.choice(["extra", "missing", "both"])
    
    if action == "extra":
        # Add extra spaces
        words = text.split()
        result = []
        for word in words:
            result.append(word)
            if random.random() < 0.3:
                result.append("")  # Creates double space
        return " ".join(result)
    
    elif action == "missing":
        # Remove some spaces
        words = text.split()
        if len(words) > 2:
            idx = random.randint(0, len(words) - 2)
            words[idx] = words[idx] + words[idx + 1]
            words.pop(idx + 1)
        return " ".join(words)
    
    return text


# =============================================================================
# Main Augmenter Class
# =============================================================================

class TextAugmenter:
    """
    Text augmenter for dental clinic messages.
    
    Applies multiple augmentation techniques to increase dataset variety.
    
    Example:
        >>> augmenter = TextAugmenter()
        >>> augmented = augmenter.augment("Хочу записаться на приём")
        >>> print(augmented)
        ['хочу записатся на приём', 'Хочу записаться на приём!', ...]
    """
    
    def __init__(self, config: Optional[AugmentationConfig] = None):
        """
        Initialize augmenter.
        
        Args:
            config: Augmentation configuration
        """
        self.config = config or AugmentationConfig()
        
        # Available augmentation functions
        self.augmentation_funcs: Dict[str, Callable[[str], str]] = {
            "typo": lambda t: add_typo(t, self.config.typo_prob),
            "abbreviation": lambda t: apply_abbreviations(t, self.config.abbreviation_prob),
            "emoji_positive": lambda t: add_emoji(t, "positive", self.config.emoji_prob),
            "emoji_negative": lambda t: add_emoji(t, "negative", self.config.emoji_prob),
            "emoji_medical": lambda t: add_emoji(t, "medical", self.config.emoji_prob),
            "case": vary_case,
            "punctuation": vary_punctuation,
            "colloquial": lambda t: apply_colloquial(t, self.config.colloquial_prob),
            "word_delete": lambda t: delete_random_word(t, self.config.word_delete_prob),
            "word_swap": lambda t: swap_adjacent_words(t, self.config.word_swap_prob),
            "char_noise": lambda t: add_char_noise(t, self.config.char_noise_prob),
            "whitespace": lambda t: vary_whitespace(t, self.config.whitespace_prob),
        }
    
    def augment(
        self,
        text: str,
        n_augmented: int = 3,
        techniques: Optional[List[str]] = None,
    ) -> List[AugmentedSample]:
        """
        Augment a single text.
        
        Args:
            text: Original text
            n_augmented: Number of augmented versions to generate
            techniques: Specific techniques to use (None = all)
            
        Returns:
            List of AugmentedSample objects
        """
        results = []
        available_techniques = techniques or list(self.augmentation_funcs.keys())
        
        for _ in range(n_augmented):
            augmented_text = text
            applied = []
            
            # Apply 1-3 random techniques
            n_techniques = random.randint(1, min(3, len(available_techniques)))
            selected = random.sample(available_techniques, n_techniques)
            
            for technique in selected:
                func = self.augmentation_funcs.get(technique)
                if func:
                    augmented_text = func(augmented_text)
                    applied.append(technique)
            
            # Only add if different from original
            if augmented_text != text:
                results.append(AugmentedSample(
                    text=augmented_text,
                    original=text,
                    augmentation_types=applied,
                ))
        
        return results
    
    def augment_single(
        self,
        text: str,
        technique: str,
    ) -> str:
        """
        Apply a single augmentation technique.
        
        Args:
            text: Original text
            technique: Technique name
            
        Returns:
            Augmented text
        """
        func = self.augmentation_funcs.get(technique)
        if func:
            return func(text)
        return text
    
    def augment_batch(
        self,
        texts: List[str],
        n_augmented_per_text: int = 2,
    ) -> List[AugmentedSample]:
        """
        Augment a batch of texts.
        
        Args:
            texts: List of original texts
            n_augmented_per_text: Number of augmented versions per text
            
        Returns:
            List of all AugmentedSample objects
        """
        all_results = []
        for text in texts:
            results = self.augment(text, n_augmented=n_augmented_per_text)
            all_results.extend(results)
        return all_results


# =============================================================================
# Dataset Augmentation Functions
# =============================================================================

def augment_dataset_advanced(
    texts: List[str],
    labels_l1: List[str],
    labels_l2: List[str],
    n_augmented: int = 2,
    seed: int = 42,
) -> tuple[List[str], List[str], List[str], List[str]]:
    """
    Augment a dataset with advanced techniques.
    
    Args:
        texts: List of original texts
        labels_l1: L1 labels for each text
        labels_l2: L2 labels for each text
        n_augmented: Number of augmented versions per text
        seed: Random seed
        
    Returns:
        Tuple of (augmented_texts, augmented_labels_l1, augmented_labels_l2, augmentation_types)
    """
    random.seed(seed)
    
    augmenter = TextAugmenter()
    
    aug_texts = []
    aug_l1 = []
    aug_l2 = []
    aug_types = []
    
    for text, l1, l2 in zip(texts, labels_l1, labels_l2):
        # Get augmented versions
        samples = augmenter.augment(text, n_augmented=n_augmented)
        
        for sample in samples:
            aug_texts.append(sample.text)
            aug_l1.append(l1)
            aug_l2.append(l2)
            aug_types.append(",".join(sample.augmentation_types))
    
    return aug_texts, aug_l1, aug_l2, aug_types


def create_augmented_variants(
    text: str,
    n_variants: int = 5,
    include_original: bool = True,
) -> List[Dict[str, Any]]:
    """
    Create multiple augmented variants of a single text.
    
    Args:
        text: Original text
        n_variants: Number of variants to create
        include_original: Whether to include original text
        
    Returns:
        List of dicts with 'text' and 'augmentation_type' keys
    """
    augmenter = TextAugmenter()
    results = []
    
    if include_original:
        results.append({
            "text": text,
            "augmentation_type": "original",
        })
    
    samples = augmenter.augment(text, n_augmented=n_variants)
    for sample in samples:
        results.append({
            "text": sample.text,
            "augmentation_type": ",".join(sample.augmentation_types),
        })
    
    return results


# =============================================================================
# Specific Augmentation Generators
# =============================================================================

def generate_typo_variants(text: str, n: int = 3) -> List[str]:
    """Generate variants with typos."""
    variants = []
    for _ in range(n):
        variant = add_typo(text, prob=0.15)
        if variant != text and variant not in variants:
            variants.append(variant)
    return variants


def generate_abbreviation_variants(text: str) -> List[str]:
    """Generate variants with abbreviations."""
    variants = []
    
    # Try each possible abbreviation
    text_lower = text.lower()
    for full, abbrs in ABBREVIATIONS.items():
        if full in text_lower:
            for abbr in abbrs:
                pattern = re.compile(re.escape(full), re.IGNORECASE)
                variant = pattern.sub(abbr, text, count=1)
                if variant not in variants:
                    variants.append(variant)
    
    return variants


def generate_case_variants(text: str) -> List[str]:
    """Generate all case variants."""
    return [
        text.lower(),
        text.upper(),
        text.capitalize(),
        text.title(),
    ]


def generate_punctuation_variants(text: str) -> List[str]:
    """Generate punctuation variants."""
    base = text.rstrip(".,!?")
    return [base + ending for ending in PUNCTUATION_ENDINGS]


# =============================================================================
# Singleton Instance
# =============================================================================

_default_augmenter: Optional[TextAugmenter] = None


def get_augmenter() -> TextAugmenter:
    """Get or create default augmenter instance."""
    global _default_augmenter
    if _default_augmenter is None:
        _default_augmenter = TextAugmenter()
    return _default_augmenter


if __name__ == "__main__":
    # Test augmentation
    augmenter = TextAugmenter()
    
    test_texts = [
        "Здравствуйте, хочу записаться на приём",
        "Болит зуб, что делать?",
        "Сколько стоит отбеливание?",
        "Спасибо большое!",
    ]
    
    print("Augmentation Test:\n")
    for text in test_texts:
        print(f"Original: {text}")
        samples = augmenter.augment(text, n_augmented=5)
        for i, sample in enumerate(samples, 1):
            print(f"  {i}. {sample.text} ({', '.join(sample.augmentation_types)})")
        print()
