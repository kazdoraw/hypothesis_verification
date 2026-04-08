"""
Entity Extraction module for dental clinic messages.

Extracts named entities such as:
- Doctors (names, surnames)
- Dates (relative and absolute)
- Times
- Procedures
- Teeth (numbers and names)
- Amounts (money)
- Durations
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field


# =============================================================================
# Entity Types and Vocabularies
# =============================================================================

# Doctor surnames (common Russian surnames)
DOCTOR_SURNAMES = [
    # А
    "Абрамов", "Алексеев", "Андреев", "Антонов",
    # Б
    "Белов", "Богданов", "Борисов",
    # В
    "Васильев", "Виноградов", "Волков", "Воробьёв",
    # Г
    "Голубев", "Григорьев",
    # Д
    "Дмитриев",
    # Е
    "Егоров",
    # Ж-З
    "Жуков", "Захаров",
    # И
    "Иванов", "Ильин",
    # К
    "Казаков", "Козлов", "Кузнецов",
    # Л
    "Лебедев",
    # М
    "Макаров", "Медведев", "Михайлов", "Морозов",
    # Н
    "Николаев", "Новиков",
    # О
    "Орлов",
    # П
    "Павлов", "Петров", "Попов",
    # Р
    "Романов",
    # С
    "Сергеев", "Сидоров", "Смирнов", "Соколов", "Соловьёв", "Степанов",
    # Т
    "Титов",
    # Ф
    "Фёдоров", "Филиппов",
    # Х-Ц-Ч
    "Хохлов", "Чернов",
    # Ш-Щ
    "Шевченко",
    # Э-Ю-Я
    "Яковлев",
]

# Female variants
DOCTOR_SURNAMES_FEMALE = [s[:-2] + "а" if s.endswith("ов") or s.endswith("ев") 
                          else s + "а" for s in DOCTOR_SURNAMES]

ALL_DOCTOR_SURNAMES = DOCTOR_SURNAMES + DOCTOR_SURNAMES_FEMALE

# Procedures vocabulary
PROCEDURES_VOCABULARY = {
    # Терапия
    "лечение кариеса": ["лечение кариеса", "кариес", "полечить кариес"],
    "пломбирование": ["пломбирование", "пломба", "поставить пломбу", "запломбировать"],
    "лечение пульпита": ["лечение пульпита", "пульпит"],
    "лечение периодонтита": ["лечение периодонтита", "периодонтит"],
    "лечение каналов": ["лечение каналов", "каналы", "пролечить каналы", "эндодонтия"],
    "реставрация": ["реставрация", "реставрировать", "восстановление зуба"],
    
    # Хирургия
    "удаление": ["удаление", "удалить", "вырвать"],
    "удаление зуба мудрости": ["зуб мудрости", "восьмёрка", "восьмерка", "мудрость"],
    "имплантация": ["имплантация", "имплант", "импланты", "поставить имплант"],
    "синус-лифтинг": ["синус-лифтинг", "синуслифтинг"],
    "костная пластика": ["костная пластика", "наращивание кости"],
    "резекция": ["резекция", "резекция верхушки"],
    
    # Ортодонтия
    "брекеты": ["брекеты", "установка брекетов", "поставить брекеты"],
    "элайнеры": ["элайнеры", "каппы", "капы", "прозрачные каппы"],
    "пластинки": ["пластинки", "пластинка"],
    "ретейнеры": ["ретейнеры", "ретейнер"],
    "исправление прикуса": ["исправление прикуса", "прикус", "неправильный прикус"],
    
    # Ортопедия
    "коронка": ["коронка", "коронки", "установка коронки"],
    "виниры": ["виниры", "винир", "поставить виниры"],
    "люминиры": ["люминиры"],
    "протезирование": ["протезирование", "протез", "протезы"],
    "мост": ["мост", "мостовидный протез"],
    "съёмный протез": ["съёмный протез", "съемный протез", "вставная челюсть"],
    
    # Гигиена
    "чистка": ["чистка", "профессиональная чистка", "почистить зубы", "гигиена"],
    "отбеливание": ["отбеливание", "отбелить", "отбеливание зубов"],
    "air-flow": ["air-flow", "эйр флоу", "airflow"],
    "профгигиена": ["профгигиена", "проф гигиена"],
    "удаление камня": ["удаление камня", "зубной камень"],
    
    # Диагностика
    "осмотр": ["осмотр", "посмотреть"],
    "консультация": ["консультация", "проконсультироваться"],
    "снимок": ["снимок", "рентген", "сделать снимок"],
    "кт": ["кт", "компьютерная томография", "3d снимок"],
    "панорамный снимок": ["панорамный снимок", "опг", "панорама"],
}

# Teeth vocabulary
TEETH_VOCABULARY = {
    # Numbers (FDI notation)
    "11": ["11", "верхняя единица", "правый верхний резец"],
    "12": ["12"],
    "21": ["21", "левый верхний резец"],
    "22": ["22"],
    "31": ["31", "левый нижний резец"],
    "32": ["32"],
    "41": ["41", "правый нижний резец"],
    "42": ["42"],
    
    # Colloquial names
    "единичка": ["единичка", "единица"],
    "двойка": ["двойка", "двоечка"],
    "тройка": ["тройка"],
    "четвёрка": ["четвёрка", "четверка"],
    "пятёрка": ["пятёрка", "пятерка"],
    "шестёрка": ["шестёрка", "шестерка", "шестой"],
    "семёрка": ["семёрка", "семерка", "седьмой"],
    "восьмёрка": ["восьмёрка", "восьмерка", "зуб мудрости", "мудрость"],
    
    # Anatomical terms
    "резец": ["резец", "резцы", "передний зуб", "передние зубы"],
    "клык": ["клык", "клыки"],
    "премоляр": ["премоляр", "премоляры", "малый коренной"],
    "моляр": ["моляр", "моляры", "коренной", "коренные", "жевательный"],
    
    # Location
    "верхняя челюсть": ["верхняя челюсть", "сверху", "верхний"],
    "нижняя челюсть": ["нижняя челюсть", "снизу", "нижний"],
    "справа": ["справа", "правый", "с правой стороны"],
    "слева": ["слева", "левый", "с левой стороны"],
}

# Date patterns
DATE_PATTERNS = {
    "relative": [
        (r"\bсегодня\b", "сегодня"),
        (r"\bзавтра\b", "завтра"),
        (r"\bпослезавтра\b", "послезавтра"),
        (r"\bвчера\b", "вчера"),
        (r"\bпозавчера\b", "позавчера"),
    ],
    "weekday": [
        (r"\b(?:в\s+)?понедельник\b", "понедельник"),
        (r"\b(?:во?\s+)?вторник\b", "вторник"),
        (r"\b(?:в\s+)?среду\b", "среда"),
        (r"\b(?:в\s+)?четверг\b", "четверг"),
        (r"\b(?:в\s+)?пятницу\b", "пятница"),
        (r"\b(?:в\s+)?субботу\b", "суббота"),
        (r"\b(?:в\s+)?воскресенье\b", "воскресенье"),
    ],
    "relative_week": [
        (r"\bна\s+этой\s+неделе\b", "на этой неделе"),
        (r"\bна\s+следующей\s+неделе\b", "на следующей неделе"),
        (r"\bна\s+прошлой\s+неделе\b", "на прошлой неделе"),
        (r"\bчерез\s+неделю\b", "через неделю"),
        (r"\bчерез\s+две\s+недели\b", "через две недели"),
    ],
    "absolute": [
        (r"(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)", None),
        (r"(\d{1,2})[./](\d{1,2})(?:[./](\d{2,4}))?", None),
    ],
}

# Time patterns
TIME_PATTERNS = [
    (r"\b(\d{1,2})[:\.](\d{2})\b", None),  # 10:00, 10.00
    (r"\b(\d{1,2})\s*час(?:ов|а)?\b", None),  # 10 часов
    (r"\bутром\b", "утром"),
    (r"\bднём\b", "днём"),
    (r"\bвечером\b", "вечером"),
    (r"\bпосле\s+обеда\b", "после обеда"),
    (r"\bдо\s+обеда\b", "до обеда"),
    (r"\bв\s+первой\s+половине\s+дня\b", "в первой половине дня"),
    (r"\bво\s+второй\s+половине\s+дня\b", "во второй половине дня"),
]

# Duration patterns
DURATION_PATTERNS = [
    (r"(\d+)\s*(?:минут[уы]?|мин)", "minutes"),
    (r"(\d+)\s*час(?:а|ов)?", "hours"),
    (r"(\d+)\s*(?:день|дня|дней)", "days"),
    (r"(\d+)\s*(?:неделю|недели|недель)", "weeks"),
    (r"(\d+)\s*(?:месяц|месяца|месяцев)", "months"),
    (r"\bуже\s+(\w+)", "already"),
    (r"\bс\s+(утра|вчера|позавчера)", "since"),
    (r"\bдавно\b", "давно"),
    (r"\bнедавно\b", "недавно"),
]

# Amount patterns (money)
AMOUNT_PATTERNS = [
    (r"(\d+(?:\s*\d{3})*)\s*(?:рублей|руб\.?|р\.?)\b", None),
    (r"(\d+)\s*(?:тысяч[иа]?|тыс\.?)\s*(?:рублей|руб\.?|р\.?)?\b", "thousands"),
]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Entity:
    """Extracted entity."""
    type: str
    value: str
    raw_text: str
    start: int
    end: int
    confidence: float = 1.0
    
    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "value": self.value,
            "raw_text": self.raw_text,
            "span": (self.start, self.end),
            "confidence": self.confidence,
        }


@dataclass
class ExtractionResult:
    """Result of entity extraction."""
    entities: List[Entity] = field(default_factory=list)
    
    def get_by_type(self, entity_type: str) -> List[Entity]:
        """Get entities of specific type."""
        return [e for e in self.entities if e.type == entity_type]
    
    def get_first(self, entity_type: str) -> Optional[Entity]:
        """Get first entity of specific type."""
        entities = self.get_by_type(entity_type)
        return entities[0] if entities else None
    
    def has_entity(self, entity_type: str) -> bool:
        """Check if entity type exists."""
        return any(e.type == entity_type for e in self.entities)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary grouped by type."""
        result = {}
        for entity in self.entities:
            if entity.type not in result:
                result[entity.type] = []
            result[entity.type].append(entity.value)
        return result
    
    def to_flat_dict(self) -> Dict[str, str]:
        """Convert to flat dict with first value per type."""
        return {
            etype: self.get_first(etype).value 
            for etype in set(e.type for e in self.entities)
        }


# =============================================================================
# Entity Extractor
# =============================================================================

class EntityExtractor:
    """
    Extract named entities from dental clinic messages.
    
    Supports extraction of:
    - doctor: Doctor names/surnames
    - date: Dates (relative and absolute)
    - time: Times
    - procedure: Dental procedures
    - tooth: Teeth (numbers and names)
    - amount: Money amounts
    - duration: Time durations
    
    Example:
        >>> extractor = EntityExtractor()
        >>> result = extractor.extract("Запишите к Иванову на завтра в 10:00")
        >>> print(result.to_dict())
        {'doctor': ['Иванову'], 'date': ['завтра'], 'time': ['10:00']}
    """
    
    def __init__(
        self,
        custom_doctors: Optional[List[str]] = None,
        custom_procedures: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Initialize entity extractor.
        
        Args:
            custom_doctors: Additional doctor surnames to recognize
            custom_procedures: Additional procedures vocabulary
        """
        self.doctors = set(s.lower() for s in ALL_DOCTOR_SURNAMES)
        if custom_doctors:
            self.doctors.update(s.lower() for s in custom_doctors)
        
        self.procedures = dict(PROCEDURES_VOCABULARY)
        if custom_procedures:
            self.procedures.update(custom_procedures)
        
        # Build reverse lookup for procedures
        self._procedure_lookup = {}
        for normalized, variants in self.procedures.items():
            for variant in variants:
                self._procedure_lookup[variant.lower()] = normalized
        
        # Build reverse lookup for teeth
        self._teeth_lookup = {}
        for normalized, variants in TEETH_VOCABULARY.items():
            for variant in variants:
                self._teeth_lookup[variant.lower()] = normalized
    
    def extract(self, text: str) -> ExtractionResult:
        """
        Extract all entities from text.
        
        Args:
            text: Input text
            
        Returns:
            ExtractionResult with all found entities
        """
        entities = []
        text_lower = text.lower()
        
        # Extract each entity type
        entities.extend(self._extract_doctors(text))
        entities.extend(self._extract_dates(text_lower))
        entities.extend(self._extract_times(text_lower))
        entities.extend(self._extract_procedures(text_lower))
        entities.extend(self._extract_teeth(text_lower))
        entities.extend(self._extract_amounts(text_lower))
        entities.extend(self._extract_durations(text_lower))
        
        return ExtractionResult(entities=entities)
    
    def _extract_doctors(self, text: str) -> List[Entity]:
        """Extract doctor names."""
        entities = []
        
        # Pattern: "у/к [Doctor]", "доктор/врач [Name]"
        patterns = [
            r"(?:у|к)\s+(\w+(?:ов[аы]?|ев[аы]?|ин[аы]?))",
            r"(?:доктор[ау]?|врач[ау]?|стоматолог[ау]?)\s+(\w+)",
            r"(\w+(?:ов|ев|ин))(?:\s+(?:лечил|принимал|смотрел))",
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                name = match.group(1)
                if name.lower().rstrip("ауы") in self.doctors or name.lower() in self.doctors:
                    entities.append(Entity(
                        type="doctor",
                        value=name,
                        raw_text=match.group(0),
                        start=match.start(),
                        end=match.end(),
                        confidence=0.9
                    ))
        
        # Direct surname match
        for surname in self.doctors:
            pattern = rf"\b{re.escape(surname)}[аыу]?\b"
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Avoid duplicates
                if not any(e.start == match.start() and e.type == "doctor" for e in entities):
                    entities.append(Entity(
                        type="doctor",
                        value=match.group(0),
                        raw_text=match.group(0),
                        start=match.start(),
                        end=match.end(),
                        confidence=0.8
                    ))
        
        return entities
    
    def _extract_dates(self, text: str) -> List[Entity]:
        """Extract date entities."""
        entities = []
        
        # Relative dates
        for pattern, value in DATE_PATTERNS["relative"]:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(Entity(
                    type="date",
                    value=value,
                    raw_text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    confidence=1.0
                ))
        
        # Weekdays
        for pattern, value in DATE_PATTERNS["weekday"]:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(Entity(
                    type="date",
                    value=value,
                    raw_text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    confidence=0.95
                ))
        
        # Relative week
        for pattern, value in DATE_PATTERNS["relative_week"]:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(Entity(
                    type="date",
                    value=value,
                    raw_text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    confidence=0.95
                ))
        
        # Absolute dates
        for pattern, _ in DATE_PATTERNS["absolute"]:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(Entity(
                    type="date",
                    value=match.group(0),
                    raw_text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9
                ))
        
        return entities
    
    def _extract_times(self, text: str) -> List[Entity]:
        """Extract time entities."""
        entities = []
        
        for pattern, value in TIME_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if value:
                    extracted_value = value
                else:
                    extracted_value = match.group(0)
                
                entities.append(Entity(
                    type="time",
                    value=extracted_value,
                    raw_text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    confidence=0.95
                ))
        
        return entities
    
    def _extract_procedures(self, text: str) -> List[Entity]:
        """Extract procedure entities."""
        entities = []
        
        for variant, normalized in self._procedure_lookup.items():
            pattern = rf"\b{re.escape(variant)}\b"
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Avoid duplicates (same span)
                if not any(e.start == match.start() and e.type == "procedure" for e in entities):
                    entities.append(Entity(
                        type="procedure",
                        value=normalized,
                        raw_text=match.group(0),
                        start=match.start(),
                        end=match.end(),
                        confidence=0.9
                    ))
        
        return entities
    
    def _extract_teeth(self, text: str) -> List[Entity]:
        """Extract teeth entities."""
        entities = []
        
        # Lookup-based extraction
        for variant, normalized in self._teeth_lookup.items():
            pattern = rf"\b{re.escape(variant)}\b"
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if not any(e.start == match.start() and e.type == "tooth" for e in entities):
                    entities.append(Entity(
                        type="tooth",
                        value=normalized,
                        raw_text=match.group(0),
                        start=match.start(),
                        end=match.end(),
                        confidence=0.85
                    ))
        
        # Pattern-based: "N-й зуб", "зуб N"
        tooth_patterns = [
            r"(\d{1,2})[-\s]*(?:й|ой|ый)?\s*зуб",
            r"зуб\s+(?:номер\s+)?(\d{1,2})",
        ]
        
        for pattern in tooth_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if not any(e.start == match.start() and e.type == "tooth" for e in entities):
                    entities.append(Entity(
                        type="tooth",
                        value=match.group(1),
                        raw_text=match.group(0),
                        start=match.start(),
                        end=match.end(),
                        confidence=0.8
                    ))
        
        return entities
    
    def _extract_amounts(self, text: str) -> List[Entity]:
        """Extract money amount entities."""
        entities = []
        
        for pattern, unit in AMOUNT_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                raw_value = match.group(1).replace(" ", "")
                
                if unit == "thousands":
                    value = f"{int(raw_value) * 1000} рублей"
                else:
                    value = f"{raw_value} рублей"
                
                entities.append(Entity(
                    type="amount",
                    value=value,
                    raw_text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9
                ))
        
        return entities
    
    def _extract_durations(self, text: str) -> List[Entity]:
        """Extract duration entities."""
        entities = []
        
        for pattern, unit in DURATION_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if unit in ("давно", "недавно"):
                    value = unit
                elif unit == "already":
                    value = f"уже {match.group(1)}"
                elif unit == "since":
                    value = f"с {match.group(1)}"
                elif match.groups():
                    num = match.group(1)
                    value = match.group(0)
                else:
                    value = match.group(0)
                
                entities.append(Entity(
                    type="duration",
                    value=value,
                    raw_text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    confidence=0.85
                ))
        
        return entities
    
    def extract_for_dataset(self, text: str) -> Dict[str, Any]:
        """
        Extract entities and return in dataset-friendly format.
        
        Returns:
            Dict with entity types as keys and first values
        """
        result = self.extract(text)
        return result.to_flat_dict()


# =============================================================================
# Utility Functions
# =============================================================================

def extract_entities(text: str) -> Dict[str, Any]:
    """
    Convenience function to extract entities from text.
    
    Args:
        text: Input text
        
    Returns:
        Dict with entity types and values
    """
    extractor = EntityExtractor()
    result = extractor.extract(text)
    return result.to_dict()


def has_doctor_mention(text: str) -> bool:
    """Check if text mentions a doctor."""
    extractor = EntityExtractor()
    result = extractor.extract(text)
    return result.has_entity("doctor")


def has_date_mention(text: str) -> bool:
    """Check if text mentions a date."""
    extractor = EntityExtractor()
    result = extractor.extract(text)
    return result.has_entity("date")


def get_mentioned_procedure(text: str) -> Optional[str]:
    """Get the first mentioned procedure from text."""
    extractor = EntityExtractor()
    result = extractor.extract(text)
    entity = result.get_first("procedure")
    return entity.value if entity else None


# =============================================================================
# Singleton Instance
# =============================================================================

_default_extractor: Optional[EntityExtractor] = None


def get_entity_extractor() -> EntityExtractor:
    """Get or create default entity extractor instance."""
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = EntityExtractor()
    return _default_extractor


if __name__ == "__main__":
    # Test entity extraction
    extractor = EntityExtractor()
    
    test_texts = [
        "Запишите к Иванову на завтра в 10:00",
        "Болит шестёрка уже неделю",
        "Сколько стоит отбеливание? Около 5000 рублей?",
        "Я был вчера у доктора Петровой, и после лечения болит",
        "Хочу перенести запись на понедельник",
        "У меня выпала пломба, нужно к терапевту на следующей неделе",
    ]
    
    print("Entity Extraction Test:\n")
    for text in test_texts:
        print(f"Text: {text}")
        result = extractor.extract(text)
        print(f"Entities: {result.to_dict()}")
        print()
