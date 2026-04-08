"""
Data generation and loading utilities for DS experiments.
"""

import json
import random
from pathlib import Path
from typing import Optional
import pandas as pd


# ============================================================================
# D1: Message Templates for Intent Classification
# ============================================================================

MESSAGE_TEMPLATES = {
    # =========================================================================
    # BOOKING - Запись на приём (80+ samples)
    # =========================================================================
    "booking": [
        # Прямые запросы на запись
        "Хочу записаться на приём к стоматологу",
        "Запишите меня на чистку зубов",
        "Можно записаться к терапевту на завтра?",
        "Есть свободное время на эту неделю?",
        "Хочу на приём к ортодонту",
        "Запись на отбеливание возможна?",
        "Мне нужно к хирургу, как записаться?",
        "Подскажите, когда можно прийти на осмотр?",
        "Хотела бы записаться к детскому стоматологу",
        "Можно ли записаться на имплантацию?",
        "Запишите меня, пожалуйста, на следующую неделю",
        "Когда есть окошки на приём?",
        "Нужна запись на протезирование",
        "Хочу попасть к врачу как можно скорее",
        "Запишите на профосмотр",
        # Дополнительные шаблоны
        "Как записаться к вам?",
        "Хочу прийти на консультацию",
        "Нужен приём у пародонтолога",
        "Запишите на снимок",
        "Можно ли попасть сегодня?",
        "Есть ли свободные слоты на завтра?",
        "Хочу записать ребёнка на осмотр",
        "Нужна запись к эндодонтисту",
        "Запишите на КТ зубов",
        "Хочу на профессиональную чистку",
        "Нужен осмотр перед лечением",
        "Записаться на установку брекетов",
        "Хочу прийти на примерку протеза",
        "Нужна запись к гигиенисту",
        "Запишите на коррекцию элайнеров",
        "Хочу к имплантологу на консультацию",
        "Есть ли приём в субботу?",
        "Можно записаться на раннее утро?",
        "Нужен срочный приём",
        "Запишите на повторный осмотр",
        "Хочу продолжить лечение",
        "Нужна запись на снятие швов",
        "Записаться на контрольный визит",
        "Хочу попасть к ортопеду",
        "Нужна консультация по винирам",
    ],
    
    # =========================================================================
    # COMPLAINT_PRIMARY - Первичная жалоба (80+ samples)
    # =========================================================================
    "complaint_primary": [
        # Боль
        "У меня сильно болит зуб",
        "Болит десна, опухла",
        "Ноет зуб уже неделю",
        "Зуб реагирует на холодное",
        "Кровоточат дёсны при чистке",
        "Откололся кусочек зуба",
        "Болит при жевании",
        "Появилась шишка на десне",
        "Зуб потемнел",
        "Чувствую пульсирующую боль",
        "Опухла щека, больно открывать рот",
        "Выпала пломба",
        "Болит под коронкой",
        "Зуб шатается",
        "Неприятный запах изо рта",
        # Дополнительные симптомы
        "Треснул зуб",
        "Болит при надавливании",
        "Зуб реагирует на горячее",
        "Десна отошла от зуба",
        "Появился флюс",
        "Гноится десна",
        "Болит челюсть",
        "Щёлкает при открывании рта",
        "Болит после удаления",
        "Не проходит онемение",
        "Зуб изменил цвет",
        "Появилась дырка в зубе",
        "Скололась эмаль",
        "Болит от сладкого",
        "Десна кровоточит постоянно",
        "Опухла губа",
        "Больно глотать",
        "Отёк под глазом",
        "Температура и болит зуб",
        "Не могу нормально жевать",
        "Зубы стали подвижными",
        "Появились язвочки во рту",
        "Болит после пломбирования",
        "Сломался протез",
        "Коронка шатается",
    ],
    
    # =========================================================================
    # PRICE_QUESTION - Вопросы о стоимости (75+ samples)
    # =========================================================================
    "price_question": [
        "Сколько стоит лечение кариеса?",
        "Какая цена на отбеливание?",
        "Прайс на имплантацию есть?",
        "Во сколько обойдётся чистка?",
        "Стоимость брекетов?",
        "Цена удаления зуба мудрости?",
        "Сколько стоит консультация?",
        "Есть ли рассрочка на лечение?",
        "Какие цены на виниры?",
        "Стоимость протезирования?",
        "Почём коронка из циркония?",
        "Сколько будет стоить всё лечение?",
        "Есть скидки пенсионерам?",
        "Какая цена на детский приём?",
        "Во сколько обойдётся снимок?",
        # Дополнительные
        "Сколько стоит удаление нерва?",
        "Цена на элайнеры?",
        "Стоимость Air-Flow?",
        "Почём КТ зубов?",
        "Сколько стоит пломба?",
        "Какая цена за один имплант?",
        "Стоимость костной пластики?",
        "Сколько стоит синус-лифтинг?",
        "Цена на металлокерамику?",
        "Почём временная коронка?",
        "Есть ли скидки для постоянных клиентов?",
        "Сколько стоит полное протезирование?",
        "Какая цена на ретейнеры?",
        "Стоимость лечения пульпита?",
        "Сколько стоит резекция?",
        "Цена на съёмный протез?",
        "Почём отбеливание Zoom?",
        "Стоимость профгигиены?",
        "Сколько стоит панорамный снимок?",
        "Какие есть акции?",
    ],
    
    # =========================================================================
    # RESCHEDULE_CANCEL - Перенос/отмена записи (70+ samples)
    # =========================================================================
    "reschedule_cancel": [
        "Хочу перенести запись",
        "Можно отменить приём на завтра?",
        "Не смогу прийти, перенесите пожалуйста",
        "Нужно изменить время записи",
        "Отмените мою запись",
        "Перенесите на другой день",
        "Не получается в это время, можно позже?",
        "Хочу отменить, заболел",
        "Можно перезаписаться на утро?",
        "Отмена записи на 15 число",
        # Дополнительные
        "Перенесите на следующую неделю",
        "Не смогу быть в назначенное время",
        "Нужно отменить визит",
        "Хочу изменить дату приёма",
        "Перезапишите на вечер",
        "Отмените запись, уезжаю",
        "Нужно перенести на пораньше",
        "Не получится прийти, заболел ребёнок",
        "Хочу сдвинуть время на час",
        "Перенесите приём на другое время",
        "Отмена записи у терапевта",
        "Не смогу завтра, занят на работе",
        "Перезапишите на субботу",
        "Отмените, пожалуйста, всё",
        "Хочу перенести чистку",
        "Нужно отменить консультацию",
        "Перезапишите на более позднее время",
        "Не получается в понедельник",
        "Хочу отменить повторный приём",
        "Перенос записи на протезирование",
    ],
    
    # =========================================================================
    # FOLLOWUP_QUESTION - Уточнения по лечению (75+ samples)
    # =========================================================================
    "followup_question": [
        "А это больно?",
        "Сколько времени займёт лечение?",
        "А что если не лечить?",
        "Какие есть альтернативы?",
        "Это точно поможет?",
        "А анестезия входит в стоимость?",
        "Сколько прослужит пломба?",
        "А если боль вернётся?",
        "Подскажите подробнее про эту процедуру",
        "А можно без удаления?",
        "Что лучше выбрать?",
        # Вопросы о процедурах
        "Как долго держатся виниры?",
        "Имплант приживётся?",
        "Сколько носить брекеты?",
        "Это навсегда?",
        "Какие гарантии?",
        "А если не подойдёт?",
        "Можно ли обойтись без анестезии?",
        "Будет ли синяк?",
        "Это безопасно?",
        "Какой срок службы коронки?",
        "А если аллергия на анестезию?",
        "Можно ли лечить беременным?",
        "Подходит ли для детей?",
        "Какой материал лучше?",
        "В чём разница между ними?",
        "Это современный метод?",
        "А есть противопоказания?",
        "Нужна ли подготовка?",
        "Какие риски?",
        "Как долго заживает?",
    ],
    
    # =========================================================================
    # CLINIC_FAQ - Вопросы о клинике (80+ samples) [НОВЫЙ КЛАСС]
    # =========================================================================
    "clinic_faq": [
        # Местоположение
        "Где вы находитесь?",
        "Как к вам добраться?",
        "Какой у вас адрес?",
        "Есть ли парковка рядом?",
        "Как добраться на общественном транспорте?",
        "Рядом есть метро?",
        "Далеко ли от центра?",
        "Есть ли бесплатная парковка?",
        # Часы работы
        "До скольки работаете?",
        "Работаете ли в выходные?",
        "Есть ли приём в субботу?",
        "Во сколько открываетесь?",
        "Работаете в праздники?",
        "Какой график работы?",
        "Работаете ли в воскресенье?",
        "До которого часа приём?",
        "Есть ли ночной приём?",
        # Оборудование
        "У вас есть рентген?",
        "Есть 3D-томограф?",
        "Делаете ли панорамные снимки?",
        "Есть детский кабинет?",
        "Работаете с наркозом?",
        "Есть ли лазер?",
        "Есть ли микроскоп?",
        # Оплата
        "Можно ли оплатить картой?",
        "Принимаете ли Apple Pay?",
        "Есть ли рассрочка?",
        "Работаете по ДМС?",
        "Принимаете полис ОМС?",
        "Можно оплатить частями?",
        "Есть ли кредит на лечение?",
        # Персонал
        "Какие врачи у вас работают?",
        "Есть ортодонт?",
        "Принимаете ли детей?",
        "Есть детский стоматолог?",
        "Кто главный врач?",
        "Есть ли имплантолог?",
        "Работает ли ортопед?",
        # Общее
        "Есть ли у вас сайт?",
        "Как с вами связаться?",
        "Есть ли WhatsApp?",
        "Можно позвонить?",
        "Есть ли личный кабинет?",
        "Можно записаться онлайн?",
        "Есть ли приложение?",
    ],
    
    # =========================================================================
    # VISIT_RECOMMENDATIONS - Рекомендации до/после (75+ samples) [НОВЫЙ КЛАСС]
    # =========================================================================
    "visit_recommendations": [
        # Подготовка к приёму
        "Как подготовиться к приёму?",
        "Нужно ли что-то делать перед лечением?",
        "Можно ли есть перед приёмом?",
        "За сколько нельзя есть?",
        "Нужно ли чистить зубы перед визитом?",
        "Какие документы взять?",
        "Нужно ли сдавать анализы?",
        "Что взять с собой?",
        "Нужно ли готовиться к имплантации?",
        "Что нужно перед удалением?",
        "Подготовка к операции?",
        "Нужна ли диета перед наркозом?",
        # После процедуры
        "Можно ли есть после процедуры?",
        "Что нельзя делать после лечения?",
        "Сколько нельзя есть после пломбы?",
        "Когда можно курить после удаления?",
        "Можно ли пить алкоголь после лечения?",
        "Как ухаживать за зубом после лечения?",
        "Что делать если болит после приёма?",
        "Нормально ли что болит после удаления?",
        "Будут ли какие-то ограничения?",
        "Что нельзя после отбеливания?",
        "Можно ли заниматься спортом после?",
        "Когда можно есть твёрдую пищу?",
        # Повторные визиты
        "Нужно ли приходить повторно?",
        "Когда прийти на осмотр?",
        "Через сколько снимать швы?",
        "Когда контрольный визит?",
        "Как часто нужно приходить на чистку?",
        # Уход
        "Как ухаживать за брекетами?",
        "Чем полоскать после удаления?",
        "Какую щётку использовать?",
        "Нужна ли специальная паста?",
        "Как чистить зубы с винирами?",
        "Чем полоскать рот после имплантации?",
        "Какие обезболивающие можно?",
        "Что делать если отёк?",
        "Как снять боль дома?",
        "Можно ли прикладывать холод?",
    ],
    
    # =========================================================================
    # OTHER - Общее (приветствия, благодарности) (65+ samples)
    # =========================================================================
    "other": [
        # Приветствия
        "Здравствуйте",
        "Добрый день",
        "Привет",
        "Доброе утро",
        "Добрый вечер",
        "Приветствую",
        "Алло",
        "Добрый день!",
        # Благодарности
        "Спасибо за информацию",
        "Благодарю",
        "Спасибо, до свидания",
        "Спасибо большое",
        "Большое спасибо",
        "Очень благодарен",
        "Спасибо за помощь",
        # Подтверждения
        "Понял, хорошо",
        "Ок, подумаю",
        "Хорошо, спасибо",
        "Ясно",
        "Понятно",
        "Да, конечно",
        "Хорошо",
        "Ладно",
        "Договорились",
        "Принято",
        "Записал",
        "Буду",
        "Приду",
        # Прощания
        "До свидания",
        "Всего доброго",
        "Пока",
        "До встречи",
        "Удачи",
        "Хорошего дня",
        # Неопределённые
        "Не знаю",
        "Подумаю",
        "Пока не решил",
        "Надо подумать",
        "Посоветуюсь",
    ],
}

# ============================================================================
# Variations templates for dynamic generation
# ============================================================================

VARIATIONS = {
    "booking": [
        "Мне нужно к {specialist}",
        "Запись к {specialist} на {day}",
        "Хочу прийти на {procedure}",
        "Есть ли свободное время на {day}?",
        "Запишите на {procedure} к {specialist}",
        "Когда можно попасть к {specialist}?",
        "Свободные окошки на {procedure}?",
        "Хочу к {specialist} на {day}",
        "Нужна запись на {procedure}",
        "Запишите на {day} на {procedure}",
    ],
    "complaint_primary": [
        "Болит {location} уже {duration}",
        "Беспокоит {symptom}",
        "{symptom}, что делать?",
        "У меня {symptom} {duration}",
        "{location} {pain_type} болит",
        "Появилась {symptom}",
        "{location} болит {duration}",
        "Сильно болит {location}",
    ],
    "price_question": [
        "Сколько стоит {procedure}?",
        "Какая цена на {procedure}?",
        "Почём {procedure}?",
        "Во сколько обойдётся {procedure}?",
        "Прайс на {procedure}?",
        "{procedure} у вас сколько стоит?",
        "Цена за {procedure}?",
    ],
    "reschedule_cancel": [
        "Перенесите запись на {day}",
        "Не смогу прийти, {reason}",
        "Можно перезаписаться на {day}?",
        "Отмените запись, {reason}",
        "Нужно перенести на {day}",
        "Не получится в {day}",
    ],
    "clinic_faq": [
        "Работаете ли в {day}?",
        "Есть ли у вас {equipment}?",
        "Принимаете {payment_method}?",
        "Есть приём в {day}?",
        "Работаете ли с {equipment}?",
    ],
    "visit_recommendations": [
        "Как подготовиться к {procedure}?",
        "Что нельзя после {procedure}?",
        "Сколько нельзя {restriction} после {procedure}?",
        "Нужна ли подготовка к {procedure}?",
        "Что делать после {procedure}?",
    ],
    "followup_question": [
        "А {procedure} это больно?",
        "Сколько длится {procedure}?",
        "Какие риски у {procedure}?",
        "А можно без {procedure}?",
    ],
}

# ============================================================================
# Generation parameters (expanded)
# ============================================================================

# Специалисты
SPECIALISTS = [
    "терапевту", "хирургу", "ортодонту", "пародонтологу", 
    "имплантологу", "ортопеду", "гигиенисту", "эндодонтисту",
    "детскому стоматологу", "челюстно-лицевому хирургу",
]

# Дни
DAYS = [
    "понедельник", "вторник", "среду", "четверг", "пятницу", 
    "субботу", "воскресенье", "завтра", "послезавтра",
    "на этой неделе", "на следующей неделе", "сегодня",
    "в ближайшее время", "утро", "вечер",
]

# Процедуры
PROCEDURES = [
    # Терапия
    "лечение кариеса", "лечение пульпита", "пломбирование",
    "лечение каналов", "реставрация",
    # Хирургия
    "удаление", "удаление зуба мудрости", "имплантацию",
    "синус-лифтинг", "костную пластику", "резекцию",
    # Ортодонтия
    "брекеты", "элайнеры", "пластинки", "ретейнеры",
    # Ортопедия
    "коронку", "виниры", "протезирование", "мост",
    # Гигиена
    "чистку", "отбеливание", "Air-Flow", "профгигиену",
    # Общее
    "осмотр", "консультацию", "снимок", "КТ", "панорамный снимок",
]

# Локализация
LOCATIONS = [
    "зуб", "передний зуб", "коренной зуб", "верхний зуб", 
    "нижний зуб", "зуб мудрости", "шестёрка", "семёрка",
    "десна", "верхняя десна", "нижняя десна",
    "челюсть", "верхняя челюсть", "нижняя челюсть",
]

# Длительность
DURATIONS = [
    "час", "два часа", "день", "два дня", "три дня",
    "неделю", "две недели", "месяц", "полгода", "давно",
    "с утра", "со вчера", "несколько дней",
]

# Симптомы
SYMPTOMS = [
    "боль при жевании", "боль от холодного", "боль от горячего",
    "чувствительность", "пульсирующая боль", "ноющая боль",
    "острая боль", "отёк", "опухоль", "шишка",
    "кровоточивость", "запах изо рта", "подвижность зуба",
]

# Тип боли
PAIN_TYPES = [
    "сильно", "немного", "очень", "постоянно", 
    "периодически", "ночью", "при надавливании",
]

# Оборудование
EQUIPMENT = [
    "рентген", "3D-томограф", "микроскоп", "лазер",
    "детский кабинет", "наркоз", "седация",
]

# Способы оплаты
PAYMENT_METHODS = [
    "карту", "наличные", "Apple Pay", "рассрочку",
    "ДМС", "ОМС", "кредит",
]

# Причины отмены
CANCEL_REASONS = [
    "заболел", "не смогу", "уезжаю", "занят на работе",
    "форс-мажор", "болею", "не получается",
]

# Ограничения
RESTRICTIONS = [
    "есть", "пить", "курить", "пить алкоголь",
    "заниматься спортом", "есть твёрдое",
]

# Backward compatibility aliases
SPECS = SPECIALISTS


def _generate_variation(intent: str) -> Optional[str]:
    """Generate a random variation for an intent."""
    if intent not in VARIATIONS:
        return None
    
    template = random.choice(VARIATIONS[intent])
    
    try:
        text = template.format(
            specialist=random.choice(SPECIALISTS),
            spec=random.choice(SPECIALISTS),  # backward compat
            day=random.choice(DAYS),
            procedure=random.choice(PROCEDURES),
            location=random.choice(LOCATIONS),
            duration=random.choice(DURATIONS),
            symptom=random.choice(SYMPTOMS),
            pain_type=random.choice(PAIN_TYPES),
            equipment=random.choice(EQUIPMENT),
            payment_method=random.choice(PAYMENT_METHODS),
            reason=random.choice(CANCEL_REASONS),
            restriction=random.choice(RESTRICTIONS),
        )
        return text
    except KeyError:
        return None


# Intent to Scenario mapping (8 classes)
INTENT_TO_SCENARIO = {
    "booking": "booking_flow",
    "complaint_primary": "anamnesis_flow",
    "reschedule_cancel": "reschedule_flow",
    "price_question": "faq_flow",
    "followup_question": "faq_flow",
    "clinic_faq": "faq_flow",
    "visit_recommendations": "faq_flow",
    "other": "faq_flow",
}

# All intent labels (8 classes)
INTENT_LABELS = list(MESSAGE_TEMPLATES.keys())


def generate_d1_dataset(n: int = 600, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic dataset for D1 (intent classification).
    
    Supports 8 intent classes:
    - booking, complaint_primary, price_question, reschedule_cancel,
    - followup_question, clinic_faq, visit_recommendations, other
    
    Args:
        n: Number of samples to generate (default 600 for 8 classes)
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with columns: id, text, label_intent, label_scenario, source
    """
    random.seed(seed)
    
    intents = list(MESSAGE_TEMPLATES.keys())
    samples_per_intent = n // len(intents)
    extra = n % len(intents)
    
    data = []
    id_counter = 1
    used_texts = set()  # Track used texts to avoid duplicates
    
    for intent in intents:
        templates = MESSAGE_TEMPLATES[intent]
        count = samples_per_intent + (1 if extra > 0 else 0)
        extra -= 1 if extra > 0 else 0
        
        generated = 0
        attempts = 0
        max_attempts = count * 3  # Prevent infinite loop
        
        while generated < count and attempts < max_attempts:
            attempts += 1
            
            # 60% from templates, 40% variations (more variety)
            if random.random() < 0.6 or intent not in VARIATIONS:
                text = random.choice(templates)
                # Add slight variations
                if random.random() < 0.3:
                    text = text.lower()
                if random.random() < 0.2:
                    text = text.rstrip("?!.") + random.choice(["", "?", "!", "."])
            else:
                text = _generate_variation(intent)
                if text is None:
                    text = random.choice(templates)
            
            # Skip duplicates
            text_normalized = text.lower().strip()
            if text_normalized in used_texts:
                continue
            used_texts.add(text_normalized)
            
            data.append({
                "id": id_counter,
                "text": text,
                "label_intent": intent,
                "label_scenario": INTENT_TO_SCENARIO[intent],
                "source": "synthetic",
            })
            id_counter += 1
            generated += 1
    
    # Shuffle
    random.shuffle(data)
    
    return pd.DataFrame(data)


# ============================================================================
# Easy Data Augmentation (EDA)
# ============================================================================

def _init_augmenters():
    """
    Initialize nlpaug augmenters lazily.
    Returns dict of augmenters or None if nlpaug not installed.
    """
    try:
        import nlpaug.augmenter.word as naw
        import nlpaug.augmenter.char as nac
        
        augmenters = {
            # Random word operations
            'random_delete': naw.RandomWordAug(action='delete', aug_p=0.1),
            'random_swap': naw.RandomWordAug(action='swap', aug_p=0.1),
            # Character-level augmentation (typos)
            'char_keyboard': nac.KeyboardAug(aug_char_p=0.05, aug_word_p=0.1),
            # Context word insertion (optional - slower)
            # 'contextual': naw.ContextualWordEmbsAug(model_path='bert-base-multilingual-cased', action='insert'),
        }
        return augmenters
    except ImportError:
        return None


def augment_text(text: str, augmenters: dict, n_aug: int = 2) -> list[str]:
    """
    Apply augmentation to a single text.
    
    Args:
        text: Original text
        augmenters: Dict of nlpaug augmenters
        n_aug: Number of augmented versions to generate
        
    Returns:
        List of augmented texts (not including original)
    """
    if augmenters is None:
        return []
    
    augmented = []
    aug_keys = list(augmenters.keys())
    
    for i in range(n_aug):
        try:
            aug_name = aug_keys[i % len(aug_keys)]
            aug = augmenters[aug_name]
            result = aug.augment(text)
            
            # nlpaug returns list or string depending on version
            if isinstance(result, list):
                aug_text = result[0] if result else text
            else:
                aug_text = result
            
            # Skip if augmentation didn't change anything
            if aug_text and aug_text.lower().strip() != text.lower().strip():
                augmented.append(aug_text)
        except Exception:
            continue
    
    return augmented


def augment_dataset(
    df: pd.DataFrame,
    n_aug: int = 2,
    text_col: str = 'text',
    label_col: str = 'label_intent',
    seed: int = 42
) -> pd.DataFrame:
    """
    Augment D1 dataset using EDA techniques.
    
    This function applies Easy Data Augmentation (EDA) to increase
    dataset size and variety:
    - Random word deletion
    - Random word swap
    - Character-level keyboard typos
    
    Args:
        df: Original DataFrame with text and labels
        n_aug: Number of augmented samples per original sample
        text_col: Name of text column
        label_col: Name of label column
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with original + augmented samples
        
    Example:
        >>> df = generate_d1_dataset(n=600)
        >>> df_aug = augment_dataset(df, n_aug=2)
        >>> print(f"Original: {len(df)}, Augmented: {len(df_aug)}")
        Original: 600, Augmented: ~1800
    """
    random.seed(seed)
    
    augmenters = _init_augmenters()
    if augmenters is None:
        print("Warning: nlpaug not installed, skipping augmentation")
        return df
    
    augmented_rows = []
    
    for _, row in df.iterrows():
        original_text = row[text_col]
        label = row[label_col]
        
        # Generate augmented versions
        aug_texts = augment_text(original_text, augmenters, n_aug=n_aug)
        
        for aug_text in aug_texts:
            new_row = row.to_dict()
            new_row[text_col] = aug_text
            new_row['source'] = 'augmented'
            # Increment ID if exists
            if 'id' in new_row:
                new_row['id'] = None  # Will be re-assigned
            augmented_rows.append(new_row)
    
    # Combine original and augmented
    df_augmented = pd.DataFrame(augmented_rows)
    result = pd.concat([df, df_augmented], ignore_index=True)
    
    # Re-assign IDs
    result['id'] = range(1, len(result) + 1)
    
    # Shuffle
    result = result.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    return result


def generate_d1_with_augmentation(
    n_base: int = 600,
    n_aug: int = 2,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate D1 dataset with automatic augmentation.
    
    Args:
        n_base: Number of base samples to generate
        n_aug: Number of augmented samples per original
        seed: Random seed
        
    Returns:
        DataFrame with base + augmented samples
    """
    df_base = generate_d1_dataset(n=n_base, seed=seed)
    df_aug = augment_dataset(df_base, n_aug=n_aug, seed=seed)
    return df_aug


def load_or_generate_d1(path: str | Path, n: int = 120, seed: int = 42) -> pd.DataFrame:
    """
    Load D1 dataset from file or generate if not exists.
    
    Args:
        path: Path to CSV file
        n: Number of samples to generate
        seed: Random seed
        
    Returns:
        DataFrame
    """
    path = Path(path)
    
    if path.exists():
        return pd.read_csv(path)
    
    df = generate_d1_dataset(n=n, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    
    return df


# ============================================================================
# D2: Case Templates for Anamnesis Collection
# ============================================================================

CASE_TEMPLATES = {
    "acute_pain": [
        {
            "initial_complaint": "У меня очень сильно болит зуб, не могу спать",
            "gold_fields": {
                "chief_complaint": "Сильная зубная боль",
                "localization": "Нижний правый моляр",
                "duration": "2 дня",
                "intensity": 8,
                "onset": "Началось после еды",
                "triggers": "Холодное, горячее",
                "chronic_or_allergies": "Нет",
            },
            "patient_answers": {
                "symptoms": "Зуб просто убивает, всю ночь промучился, ни спать ни есть не могу",
                "localization": "Справа внизу, дальний зуб, ну который для жевания",
                "duration": "Позавчера вечером началось, а сейчас вообще невыносимо",
                "chronic_or_allergies": "Да нет вроде, ничего такого. Здоров в целом",
            },
        },
        {
            "initial_complaint": "Острая боль в верхней челюсти",
            "gold_fields": {
                "chief_complaint": "Острая боль в верхней челюсти",
                "localization": "Верхний левый премоляр",
                "duration": "3 дня",
                "intensity": 7,
                "onset": "Постепенно усиливалась",
                "triggers": "Жевание, надавливание",
                "chronic_or_allergies": "Аллергия на лидокаин",
            },
            "patient_answers": {
                "symptoms": "Резкая такая боль наверху, как прострел, потом отпускает и снова",
                "localization": "Наверху слева, ближе к щеке, четвёртый или пятый если считать",
                "duration": "Уже третий день мучаюсь, причём с каждым днём хуже становится",
                "chronic_or_allergies": "Когда-то укол делали лидокаин — плохо стало, опухло всё",
            },
        },
        {
            "initial_complaint": "Невыносимая боль, зуб пульсирует",
            "gold_fields": {
                "chief_complaint": "Пульсирующая зубная боль",
                "localization": "Нижний левый моляр, 36 зуб",
                "duration": "1 день",
                "intensity": 9,
                "onset": "Внезапно ночью",
                "triggers": "Горячее, лёжа",
                "chronic_or_allergies": "Диабет 2 типа",
            },
            "patient_answers": {
                "symptoms": "Пульсирует прямо, как сердце бьётся в зубе, ночью вообще кошмар был",
                "localization": "Внизу слева, не самый дальний, а предпоследний кажется",
                "duration": "Со вчерашнего вечера, вдруг как начало стрелять ни с того ни с сего",
                "chronic_or_allergies": "Сахарный есть, второго типа. Таблетки пью каждый день",
            },
        },
        {
            "initial_complaint": "Резко заболел зуб после лечения",
            "gold_fields": {
                "chief_complaint": "Боль после пломбирования",
                "localization": "Верхний правый премоляр",
                "duration": "1 день",
                "intensity": 6,
                "onset": "После визита к стоматологу вчера",
                "triggers": "При накусывании",
                "chronic_or_allergies": "Нет аллергий",
            },
            "patient_answers": {
                "symptoms": "Вчера пломбу поставили, а к вечеру разболелось! Накусить вообще не могу",
                "localization": "Наверху справа, который вчера лечили, номер не помню какой",
                "duration": "Со вчера, как из кресла встал — сначала нормально, потом к ночи началось",
                "chronic_or_allergies": "Нет, ничего нет, здоров",
            },
        },
        {
            "initial_complaint": "Сильная боль и опухла щека",
            "gold_fields": {
                "chief_complaint": "Острая боль с отёком щеки",
                "localization": "Нижняя челюсть справа",
                "duration": "2 дня",
                "intensity": 8,
                "onset": "Начало ныть, потом усилилось",
                "triggers": "Любое прикосновение",
                "chronic_or_allergies": "Гипертония, принимаю эналаприл",
            },
            "patient_answers": {
                "symptoms": "Всё распухло, щека как подушка стала, больно даже рот открыть",
                "localization": "Справа вся челюсть нижняя болит, и в ухо как-то отдаёт",
                "duration": "Два дня назад появилось, сначала чуть ныло, а сегодня раздуло",
                "chronic_or_allergies": "Давление высокое, пью эналаприл каждый день утром",
            },
        },
    ],
    "chronic_pain": [
        {
            "initial_complaint": "Зуб ноет уже месяц, то сильнее, то слабее",
            "gold_fields": {
                "chief_complaint": "Ноющая боль в зубе",
                "localization": "Нижний левый моляр",
                "duration": "1 месяц",
                "intensity": 4,
                "onset": "Постепенно",
                "relievers": "Обезболивающие помогают",
                "chronic_or_allergies": "Нет",
            },
            "patient_answers": {
                "symptoms": "Ноет и ноет, бывает терпимо, а бывает прямо на стенку лезу",
                "localization": "Внизу слева, жевательный, давно уже с ним проблемы были",
                "duration": "Ну уже месяц точно, а может и подольше, сначала не обращал внимания",
                "chronic_or_allergies": "Нет, вроде всё нормально, ничем не болею",
            },
        },
        {
            "initial_complaint": "Периодически болит один и тот же зуб",
            "gold_fields": {
                "chief_complaint": "Периодическая боль",
                "localization": "Верхний передний зуб",
                "duration": "несколько месяцев",
                "intensity": 3,
                "onset": "Не помню точно",
                "relievers": "Проходит само",
                "chronic_or_allergies": "Аллергия на пенициллин",
            },
            "patient_answers": {
                "symptoms": "То болит, то не болит, непонятно, зуб как будто живёт своей жизнью",
                "localization": "Передний сверху, большой такой, ну резец наверное называется",
                "duration": "Да уже несколько месяцев так, точно не вспомню когда началось",
                "chronic_or_allergies": "На антибиотики пенициллиновые — опухаю от них, врачи говорили",
            },
        },
        {
            "initial_complaint": "Давно побаливает зуб под старой коронкой",
            "gold_fields": {
                "chief_complaint": "Боль под коронкой",
                "localization": "Верхний правый моляр, под коронкой",
                "duration": "3 месяца",
                "intensity": 3,
                "onset": "Постепенно нарастала",
                "relievers": "Полоскание содой",
                "chronic_or_allergies": "Нет хронических заболеваний",
            },
            "patient_answers": {
                "symptoms": "Под коронкой что-то ноет, особенно когда горячее пью, неприятно",
                "localization": "Наверху справа, там коронка стоит уже лет пять наверное",
                "duration": "Месяца три как начал замечать, раньше вообще нормально было",
                "chronic_or_allergies": "Ничего такого нет, хронических болячек вроде не имею",
            },
        },
        {
            "initial_complaint": "Чувствительность зубов к холодному и горячему",
            "gold_fields": {
                "chief_complaint": "Повышенная чувствительность зубов",
                "localization": "Несколько зубов, передние нижние",
                "duration": "2 месяца",
                "intensity": 4,
                "onset": "После профессиональной чистки",
                "relievers": "Зубная паста для чувствительных зубов",
                "chronic_or_allergies": "Гастрит",
            },
            "patient_answers": {
                "symptoms": "Как мороженое ем или чай горячий — аж передёргивает всего",
                "localization": "Нижние передние в основном, но иногда и другие тоже реагируют",
                "duration": "После чистки в клинике началось, месяца два назад делали ультразвуком",
                "chronic_or_allergies": "Желудок слабый, гастрит хронический, на учёте стою",
            },
        },
        {
            "initial_complaint": "Тупая боль в зубе, то появляется, то исчезает",
            "gold_fields": {
                "chief_complaint": "Тупая периодическая боль",
                "localization": "Нижний правый премоляр",
                "duration": "2 недели",
                "intensity": 3,
                "onset": "После простуды",
                "relievers": "Нурофен снимает",
                "chronic_or_allergies": "Нет",
            },
            "patient_answers": {
                "symptoms": "Тупая такая боль, как будто давит что-то изнутри, потом проходит",
                "localization": "Справа внизу, но какой именно — не разберу, может и десна рядом",
                "duration": "Недели две примерно, после простуды что-то началось такое",
                "chronic_or_allergies": "Нет, ничего нет, здоров",
            },
        },
    ],
    "esthetics": [
        {
            "initial_complaint": "Хочу красивую улыбку, зубы жёлтые",
            "gold_fields": {
                "chief_complaint": "Желтизна зубов",
                "localization": "Все передние зубы",
                "desired_outcome": "Белоснежные зубы",
                "chronic_or_allergies": "Нет",
            },
            "patient_answers": {
                "symptoms": "Зубы жёлтые какие-то, некрасивые, стесняюсь улыбаться на фото",
                "localization": "Ну все передние, которые видно когда улыбаюсь, и верхние и нижние",
                "duration": "Давно уже так, но в последнее время хуже стало от кофе наверное",
                "chronic_or_allergies": "Нет, здоров, ничем не болею",
            },
        },
        {
            "initial_complaint": "Хочу поставить виниры",
            "gold_fields": {
                "chief_complaint": "Установка виниров",
                "localization": "Зона улыбки",
                "desired_outcome": "Ровные красивые зубы",
                "chronic_or_allergies": "Нет аллергий",
            },
            "patient_answers": {
                "symptoms": "Хочу ровные красивые зубы, как у звёзд, чтобы идеально было",
                "localization": "Ну зону улыбки, верхние передние штук шесть-восемь наверное",
                "duration": "Давно хочу, наконец решился, деньги накопил",
                "chronic_or_allergies": "Нет аллергий, ничем серьёзным не болел",
            },
        },
        {
            "initial_complaint": "Зубы потемнели, хочу отбеливание",
            "gold_fields": {
                "chief_complaint": "Потемнение зубов",
                "localization": "Верхние и нижние передние",
                "desired_outcome": "Осветление на 3-4 тона",
                "chronic_or_allergies": "Аллергия на перекись водорода",
            },
            "patient_answers": {
                "symptoms": "Потемнели от кофе наверное, раньше белее были, сейчас серые какие-то",
                "localization": "И верхние и нижние передние, но верхние особенно заметно",
                "duration": "Последний год примерно, быстро как-то потемнели",
                "chronic_or_allergies": "На перекись плохо реагирую, дёсны воспаляются и жжёт",
            },
        },
        {
            "initial_complaint": "Щель между передними зубами, хочу убрать",
            "gold_fields": {
                "chief_complaint": "Диастема (щель между зубами)",
                "localization": "Верхние центральные резцы",
                "desired_outcome": "Закрыть щель, ровная улыбка",
                "chronic_or_allergies": "Нет",
            },
            "patient_answers": {
                "symptoms": "Щербинка между зубами, еда постоянно застревает и некрасиво выглядит",
                "localization": "Верхние два передних, самых больших, между ними дырка такая",
                "duration": "С детства так было, но сейчас решил наконец исправить",
                "chronic_or_allergies": "Ничего нет, здоров",
            },
        },
        {
            "initial_complaint": "Скол на переднем зубе, некрасиво",
            "gold_fields": {
                "chief_complaint": "Скол эмали на переднем зубе",
                "localization": "Верхний центральный резец",
                "desired_outcome": "Реставрация, чтобы не было видно",
                "chronic_or_allergies": "Нет хронических",
            },
            "patient_answers": {
                "symptoms": "Откололся кусочек, теперь неровный край, язык цепляется постоянно",
                "localization": "Верхний передний, самый большой, ну по центру который",
                "duration": "На прошлой неделе случилось, орех неудачно раскусил",
                "chronic_or_allergies": "Нет, ничего такого, здоров в целом",
            },
        },
    ],
    "ortho": [
        {
            "initial_complaint": "У ребёнка кривые зубы",
            "gold_fields": {
                "chief_complaint": "Неровные зубы у ребёнка",
                "localization": "Верхняя челюсть",
                "bite_issues": "Скученность, неправильный прикус",
                "chronic_or_allergies": "Нет",
            },
            "patient_answers": {
                "symptoms": "Зубы криво растут, один на другой залезает, страшно смотреть уже",
                "localization": "Наверху в основном, и клык один торчит в сторону сильно",
                "duration": "Года два как начали постоянные расти, и пошло-поехало всё",
                "chronic_or_allergies": "Нет, ребёнок здоровый, ничем не болеет",
            },
        },
        {
            "initial_complaint": "Хочу исправить прикус",
            "gold_fields": {
                "chief_complaint": "Неправильный прикус",
                "localization": "Обе челюсти",
                "bite_issues": "Глубокий прикус",
                "chronic_or_allergies": "Нет аллергий",
            },
            "patient_answers": {
                "symptoms": "Зубы не смыкаются нормально, верхние сильно перекрывают нижние",
                "localization": "И сверху и снизу, вся челюсть наверное, не знаю как объяснить",
                "duration": "С детства так, но раньше не парился, а сейчас решил исправить",
                "chronic_or_allergies": "Нет, ничего нет, здоров",
            },
        },
        {
            "initial_complaint": "Зубы растут неровно, нужны брекеты",
            "gold_fields": {
                "chief_complaint": "Неровный рост зубов",
                "localization": "Верхняя и нижняя челюсть",
                "bite_issues": "Скученность передних зубов",
                "chronic_or_allergies": "Аллергия на никель",
            },
            "patient_answers": {
                "symptoms": "Зубы кривые стоят, скученные какие-то, в кучу сбились передние",
                "localization": "И верхние и нижние, передние больше всего кривые",
                "duration": "Лет с двенадцати примерно, мама привела наконец",
                "chronic_or_allergies": "На никель бывает реакция, серёжки носить не могу — краснеет",
            },
        },
        {
            "initial_complaint": "Нижняя челюсть выдвинута вперёд",
            "gold_fields": {
                "chief_complaint": "Мезиальный прикус",
                "localization": "Нижняя челюсть",
                "bite_issues": "Нижняя челюсть выступает, зубы не смыкаются",
                "chronic_or_allergies": "Нет",
            },
            "patient_answers": {
                "symptoms": "Нижняя челюсть вперёд торчит, зубы не сходятся, жевать неудобно бывает",
                "localization": "Ну нижняя вся челюсть как бы выступает вперёд",
                "duration": "С подросткового возраста, но в последнее время стало заметнее",
                "chronic_or_allergies": "Нет, здоров, ничем не болею",
            },
        },
        {
            "initial_complaint": "Хочу элайнеры вместо брекетов",
            "gold_fields": {
                "chief_complaint": "Желание выровнять зубы элайнерами",
                "localization": "Обе челюсти, передний отдел",
                "bite_issues": "Лёгкая скученность, ротация резцов",
                "chronic_or_allergies": "Астма",
            },
            "patient_answers": {
                "symptoms": "Зубы кривоваты, хочу без железок выровнять, есть прозрачные штуки такие",
                "localization": "Передние в основном кривые, и сверху и снизу, повёрнутые немного",
                "duration": "Давно так, но сейчас хочу исправить, железные брекеты не хочу",
                "chronic_or_allergies": "Астма есть, ингалятор с собой ношу, иногда пользуюсь",
            },
        },
    ],
    "therapy": [
        {
            "initial_complaint": "Нужно вылечить кариес",
            "gold_fields": {
                "chief_complaint": "Кариес",
                "localization": "Несколько зубов",
                "last_visit": "Полгода назад",
                "chronic_or_allergies": "Нет",
            },
            "patient_answers": {
                "symptoms": "Дырки в зубах вижу, потемнело, один побаливает когда сладкое ем",
                "localization": "Несколько штук, и справа и слева есть проблемные вроде",
                "duration": "Ну давно уже заметил, полгода как минимум, руки не доходили",
                "chronic_or_allergies": "Нет, здоров, ничего такого",
            },
        },
        {
            "initial_complaint": "Давно не был у стоматолога, нужен осмотр",
            "gold_fields": {
                "chief_complaint": "Профилактический осмотр",
                "localization": "Все зубы",
                "last_visit": "Больше года назад",
                "chronic_or_allergies": "Нет аллергий",
            },
            "patient_answers": {
                "symptoms": "Ну в целом вроде нормально, но давно не проверялся, мало ли что там",
                "localization": "Не знаю, всё проверить надо бы, может что-то незаметное есть",
                "duration": "Года полтора не ходил точно, а может и больше, не помню уже",
                "chronic_or_allergies": "Нет, ничем не болею, аллергий не знаю",
            },
        },
        {
            "initial_complaint": "Заметил тёмное пятно на зубе",
            "gold_fields": {
                "chief_complaint": "Тёмное пятно, возможно кариес",
                "localization": "Верхний правый моляр",
                "last_visit": "Год назад",
                "chronic_or_allergies": "Аллергия на анестетики артикаинового ряда",
            },
            "patient_answers": {
                "symptoms": "Пятнышко тёмное появилось, щёткой не оттирается, кариес наверное",
                "localization": "Наверху справа, дальний зуб, в зеркало увидел случайно",
                "duration": "Недавно заметил, может неделю назад, а может раньше было просто не видел",
                "chronic_or_allergies": "Когда-то на укол реакция была у стоматолога, не помню на что точно, плохо стало",
            },
        },
        {
            "initial_complaint": "Нужна профессиональная чистка зубов",
            "gold_fields": {
                "chief_complaint": "Профессиональная гигиена",
                "localization": "Все зубы, особенно нижние передние",
                "last_visit": "8 месяцев назад",
                "chronic_or_allergies": "Нет",
            },
            "patient_answers": {
                "symptoms": "Камень скопился наверное, дёсны кровят когда чищу, налёт желтоватый виден",
                "localization": "Внизу передние больше всего видно, но и сзади тоже наверное есть",
                "duration": "Давно чистку не делал, месяцев восемь наверное прошло с последней",
                "chronic_or_allergies": "Нет, всё нормально со здоровьем",
            },
        },
        {
            "initial_complaint": "Выпала старая пломба, нужно поставить новую",
            "gold_fields": {
                "chief_complaint": "Выпавшая пломба",
                "localization": "Нижний левый моляр, 37 зуб",
                "last_visit": "3 месяца назад",
                "chronic_or_allergies": "Диабет, принимаю метформин",
            },
            "patient_answers": {
                "symptoms": "Пломба выскочила, там дырка теперь, еда забивается и неприятно",
                "localization": "Внизу слева, большой жевательный, тридцать какой-то, не помню номер",
                "duration": "На днях выпала, пломбе лет пять было наверное, давно ставили",
                "chronic_or_allergies": "Диабет есть, таблетки пью, метформин каждый день",
            },
        },
    ],
    # =========================================================================
    # ADVERSARIAL — Сложные, неоднозначные, шумные кейсы (5 шаблонов)
    # cooperation_mode: adversarial — симулятор может дать шумный ответ
    # =========================================================================
    "adversarial": [
        {
            "initial_complaint": "Что-то не так с зубами, не знаю что именно",
            "mapped_type": "chronic_pain",
            "cooperation_mode": "ambiguous",
            "gold_fields": {
                "chief_complaint": "Неопределённый дискомфорт",
                "localization": "Не определена",
                "duration": "Несколько недель",
                "chronic_or_allergies": "Нет",
            },
            "patient_answers": {
                "symptoms": "Ну как вам сказать... что-то не то, дискомфорт какой-то, не боль прямо, а непонятно",
                "localization": "Где-то болит, точно не скажу... может справа, а может слева, когда трогаю языком — вроде всё норм",
                "duration": "Давно уже, не помню когда началось, может месяц, может два, я внимания не обращал",
                "chronic_or_allergies": "Да вроде нет ничего... хотя не помню, давно не проверялся, может и есть что-то",
            },
        },
        {
            "initial_complaint": "Зуб болит, и ещё у меня вопрос по расписанию и ценам",
            "mapped_type": "acute_pain",
            "cooperation_mode": "topic_switch",
            "gold_fields": {
                "chief_complaint": "Зубная боль",
                "localization": "Верхний передний зуб",
                "duration": "1 неделя",
                "chronic_or_allergies": "Гипертония",
            },
            "patient_answers": {
                "symptoms": "Болит зуб, а сколько кстати консультация стоит? И ещё у меня спина побаливает, это к вам или нет?",
                "localization": "Передний наверху... а вы по выходным работаете? Мне бы в субботу удобнее",
                "duration": "Неделю примерно, а кстати можно ли рассрочку оформить на лечение?",
                "chronic_or_allergies": "Давление повышенное, таблетки пью... а у вас парковка есть рядом?",
            },
        },
        {
            "initial_complaint": "Нужна помощь с зубами",
            "mapped_type": "acute_pain",
            "cooperation_mode": "refusal",
            "gold_fields": {
                "chief_complaint": "Зубная боль",
                "localization": "Нижний правый моляр",
                "duration": "5 дней",
                "chronic_or_allergies": "Отказ отвечать",
            },
            "patient_answers": {
                "symptoms": "Болит, а что конкретнее — на приёме скажу, не хочу в чат писать",
                "localization": "Внизу справа, жевательный",
                "duration": "Дней пять уже. А зачем вам это всё? Я просто записаться хочу",
                "chronic_or_allergies": "Это личная информация, не хочу здесь обсуждать. На приёме скажу врачу",
            },
        },
        {
            "initial_complaint": "Так, значит, у меня тут вот какая ситуация — зуб начал болеть, мы с женой как раз из отпуска вернулись, в Турции были, жарко было, много холодного пили, и вот после мороженого как-то началось",
            "mapped_type": "acute_pain",
            "cooperation_mode": "info_dump",
            "gold_fields": {
                "chief_complaint": "Боль после холодного",
                "localization": "Верхний левый премоляр",
                "duration": "1 неделя",
                "chronic_or_allergies": "Нет",
            },
            "patient_answers": {
                "symptoms": "Ну вот говорю же, от мороженого началось, потом и от воды холодной, а горячий чай нормально, хотя нет, вру, тоже немного, жена говорит я зубами скриплю ночью, может от этого",
                "localization": "Наверху где-то слева, четвёртый кажется или пятый, я зеркалом смотрел но плохо видно, может и не тот, стоматолог разберётся",
                "duration": "Неделю назад из Турции прилетели, вот тогда и началось, хотя может раньше чуть-чуть было просто не замечал за отпуском",
                "chronic_or_allergies": "Нет, здоров, ТТТ, ничего нет, ни аллергий ни болезней, разве что спина иногда, но это не по вашей части",
            },
        },
        {
            "initial_complaint": "У меня зубы болят, не знаю какой именно",
            "mapped_type": "acute_pain",
            "cooperation_mode": "contradictory",
            "gold_fields": {
                "chief_complaint": "Боль неопределённой локализации",
                "localization": "Нижняя челюсть справа",
                "duration": "Около недели",
                "chronic_or_allergies": "Аллергия на пенициллин",
            },
            "patient_answers": {
                "symptoms": "Болит... ну не сильно, хотя иногда сильно, ночью вообще невыносимо было, а сейчас терпимо вроде",
                "localization": "Внизу справа... или слева? Нет, справа точно. Хотя когда жую — слева тоже отдаёт. Наверное справа всё-таки",
                "duration": "Дня два назад... а нет, наверное неделю уже. Хотя может и больше, просто я сначала не обращал внимания. Короче давно",
                "chronic_or_allergies": "На пенициллин аллергия. Или на амоксициллин? Один из них, точно не помню, в детстве было. Сыпь была и отёк",
            },
        },
    ],
}


def generate_d2_cases(n: int = 30, seed: int = 42) -> list[dict]:
    """
    Generate synthetic cases for D2 (anamnesis collection).
    
    Args:
        n: Number of cases to generate
        seed: Random seed
        
    Returns:
        List of case dicts
    """
    random.seed(seed)
    
    complaint_types = list(CASE_TEMPLATES.keys())
    cases_per_type = n // len(complaint_types)
    extra = n % len(complaint_types)
    
    cases = []
    case_id = 1
    
    for ctype in complaint_types:
        templates = CASE_TEMPLATES[ctype]
        count = cases_per_type + (1 if extra > 0 else 0)
        extra -= 1 if extra > 0 else 0
        
        for _ in range(count):
            template = random.choice(templates)
            
            # adversarial шаблоны имеют mapped_type для совместимости с COMPLAINT_TYPES
            effective_type = template.get("mapped_type", ctype)
            case = {
                "case_id": case_id,
                "initial_user_message": template["initial_complaint"],
                "target_complaint_type": effective_type,
                "gold_required_fields": list(template["gold_fields"].keys()),
                "gold_values": template["gold_fields"].copy(),
                "patient_answers": template.get("patient_answers", {}),
                "cooperation_mode": template.get("cooperation_mode", "cooperative"),
                "source": "synthetic",
            }
            
            cases.append(case)
            case_id += 1
    
    random.shuffle(cases)
    return cases


def load_or_generate_d2(path: str | Path, n: int = 30, seed: int = 42) -> list[dict]:
    """
    Load D2 cases from file or generate if not exists.
    
    Args:
        path: Path to JSONL file
        n: Number of cases
        seed: Random seed
        
    Returns:
        List of case dicts
    """
    path = Path(path)
    
    if path.exists():
        cases = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    cases.append(json.loads(line))
        return cases
    
    cases = generate_d2_cases(n=n, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    
    return cases


if __name__ == "__main__":
    # Test data generation
    print("Generating D1 dataset...")
    df = generate_d1_dataset(n=20, seed=42)
    print(df.head(10))
    print(f"\nIntent distribution:\n{df['label_intent'].value_counts()}")
    
    print("\n" + "="*50)
    print("Generating D2 cases...")
    cases = generate_d2_cases(n=10, seed=42)
    for case in cases[:3]:
        print(f"\nCase {case['case_id']} ({case['target_complaint_type']}):")
        print(f"  Complaint: {case['initial_user_message']}")
        print(f"  Required fields: {case['gold_required_fields']}")
