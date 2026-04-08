"""
Advanced classifiers for intent classification.

This module provides:
- EmbeddingClassifier: sentence-transformers + sklearn
- SetFitClassifier: few-shot learning with SetFit
- RuleBasedClassifier: regex-based with priorities
- CascadeClassifier: Rule → ML → LLM cascade

Author: AI Dentist Team
"""

import re
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any
import numpy as np
import joblib

from .taxonomy import CONFIDENCE_THRESHOLDS

logger = logging.getLogger(__name__)

# SSoT thresholds from taxonomy
_ML_THRESHOLD = CONFIDENCE_THRESHOLDS["high"]
_LLM_THRESHOLD = CONFIDENCE_THRESHOLDS["low"]


# ============================================================================
# Embedding Classifier (Sentence-Transformers)
# ============================================================================

class EmbeddingClassifier:
    """
    Intent classifier using sentence-transformers embeddings.
    
    Uses pre-trained Russian BERT model for semantic embeddings
    combined with a sklearn classifier.
    
    Example:
        >>> clf = EmbeddingClassifier(model_name='cointegrated/rubert-tiny2')
        >>> clf.fit(train_texts, train_labels)
        >>> predictions = clf.predict(test_texts)
        >>> probas = clf.predict_proba(test_texts)
    """
    
    # Available Russian models (by quality/size)
    MODELS = {
        'rubert-tiny2': 'cointegrated/rubert-tiny2',  # 45MB, fast
        'rubert-tiny-sts': 'sergeyzh/rubert-tiny-sts',  # 45MB, very fast
        'e5-base': 'intfloat/multilingual-e5-base',  # 278MB, medium
        'e5-large': 'intfloat/multilingual-e5-large',  # 560MB, best quality
    }
    
    def __init__(
        self,
        model_name: str = 'cointegrated/rubert-tiny2',
        classifier_type: str = 'logistic',
        classifier_params: Optional[dict] = None,
        device: Optional[str] = None
    ):
        """
        Initialize EmbeddingClassifier.
        
        Args:
            model_name: Name of sentence-transformer model or alias
            classifier_type: 'logistic', 'svc', or 'rf'
            classifier_params: Custom params for sklearn classifier
            device: 'cpu', 'cuda', or None (auto)
        """
        # Resolve model alias
        if model_name in self.MODELS:
            model_name = self.MODELS[model_name]
        
        self.model_name = model_name
        self.classifier_type = classifier_type
        self.device = device
        
        # Lazy load encoder
        self._encoder = None
        self._classifier = None
        self._classes = None
        
        # Default classifier params
        default_params = {
            'logistic': {'max_iter': 1000, 'C': 10.0, 'class_weight': 'balanced'},
            'svc': {'C': 10.0, 'kernel': 'rbf', 'probability': True, 'class_weight': 'balanced'},
            'rf': {'n_estimators': 100, 'max_depth': 10, 'class_weight': 'balanced'},
        }
        
        self.classifier_params = classifier_params or default_params.get(classifier_type, {})
    
    @property
    def encoder(self):
        """Lazy load sentence transformer."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self.model_name, device=self.device)
                logger.info(f"Loaded encoder: {self.model_name}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
        return self._encoder
    
    def _init_classifier(self):
        """Initialize sklearn classifier."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier
        
        classifiers = {
            'logistic': LogisticRegression,
            'svc': SVC,
            'rf': RandomForestClassifier,
        }
        
        clf_class = classifiers.get(self.classifier_type, LogisticRegression)
        return clf_class(**self.classifier_params)
    
    def encode(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of text strings
            show_progress: Show progress bar
            
        Returns:
            Numpy array of shape (n_texts, embedding_dim)
        """
        return self.encoder.encode(
            texts, 
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
    
    def fit(self, texts: List[str], labels: List[str]) -> 'EmbeddingClassifier':
        """
        Fit classifier on training data.
        
        Args:
            texts: Training texts
            labels: Training labels
            
        Returns:
            self
        """
        logger.info(f"Encoding {len(texts)} training samples...")
        embeddings = self.encode(texts)
        
        self._classifier = self._init_classifier()
        self._classifier.fit(embeddings, labels)
        self._classes = self._classifier.classes_
        
        logger.info(f"Trained classifier with {len(self._classes)} classes")
        return self
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Predict labels for texts.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            Array of predicted labels
        """
        if self._classifier is None:
            raise ValueError("Classifier not fitted. Call fit() first.")
        
        embeddings = self.encode(texts, show_progress=False)
        return self._classifier.predict(embeddings)
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            texts: List of texts
            
        Returns:
            Array of shape (n_texts, n_classes) with probabilities
        """
        if self._classifier is None:
            raise ValueError("Classifier not fitted. Call fit() first.")
        
        embeddings = self.encode(texts, show_progress=False)
        return self._classifier.predict_proba(embeddings)
    
    def predict_with_confidence(
        self, texts: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Predict with confidence scores.
        
        Args:
            texts: List of texts
            
        Returns:
            List of (label, confidence) tuples
        """
        probas = self.predict_proba(texts)
        results = []
        
        for proba in probas:
            max_idx = np.argmax(proba)
            label = self._classes[max_idx]
            confidence = proba[max_idx]
            results.append((label, float(confidence)))
        
        return results
    
    @property
    def classes_(self) -> np.ndarray:
        """Get class labels."""
        return self._classes
    
    def save(self, path: str | Path):
        """
        Save classifier to disk.
        
        Args:
            path: Directory path to save model
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save sklearn classifier
        joblib.dump(self._classifier, path / 'classifier.joblib')
        
        # Save metadata
        meta = {
            'model_name': self.model_name,
            'classifier_type': self.classifier_type,
            'classifier_params': self.classifier_params,
            'classes': list(self._classes) if self._classes is not None else None,
        }
        with open(path / 'meta.json', 'w') as f:
            json.dump(meta, f, indent=2)
        
        logger.info(f"Saved classifier to {path}")
    
    @classmethod
    def load(cls, path: str | Path) -> 'EmbeddingClassifier':
        """
        Load classifier from disk.
        
        Args:
            path: Directory path with saved model
            
        Returns:
            Loaded EmbeddingClassifier
        """
        path = Path(path)
        
        # Load metadata
        with open(path / 'meta.json') as f:
            meta = json.load(f)
        
        # Create instance
        instance = cls(
            model_name=meta['model_name'],
            classifier_type=meta['classifier_type'],
            classifier_params=meta.get('classifier_params'),
        )
        
        # Load classifier
        instance._classifier = joblib.load(path / 'classifier.joblib')
        instance._classes = np.array(meta['classes']) if meta['classes'] else None
        
        logger.info(f"Loaded classifier from {path}")
        return instance


# ============================================================================
# SetFit Classifier (Few-Shot Learning)
# ============================================================================

class SetFitClassifier:
    """
    Few-shot learning classifier using SetFit.
    
    SetFit achieves SOTA results with 8-50 samples per class.
    Uses contrastive learning to fine-tune sentence transformers.
    
    Example:
        >>> clf = SetFitClassifier(model_name='cointegrated/rubert-tiny2')
        >>> clf.fit(train_texts, train_labels, num_epochs=3)
        >>> predictions = clf.predict(test_texts)
    """
    
    def __init__(
        self,
        model_name: str = 'cointegrated/rubert-tiny2',
        labels: Optional[List[str]] = None
    ):
        """
        Initialize SetFitClassifier.
        
        Args:
            model_name: Base sentence-transformer model
            labels: List of all possible labels (optional)
        """
        self.model_name = model_name
        self.labels = labels
        self._model = None
        self._trainer = None
    
    def fit(
        self,
        texts: List[str],
        labels: List[str],
        num_epochs: int = 3,
        batch_size: int = 16,
        eval_texts: Optional[List[str]] = None,
        eval_labels: Optional[List[str]] = None
    ) -> 'SetFitClassifier':
        """
        Train SetFit model.
        
        Args:
            texts: Training texts
            labels: Training labels
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            eval_texts: Optional evaluation texts
            eval_labels: Optional evaluation labels
            
        Returns:
            self
        """
        try:
            from setfit import SetFitModel, Trainer, TrainingArguments
            from datasets import Dataset
        except ImportError:
            raise ImportError(
                "setfit not installed. Run: pip install setfit"
            )
        
        # Determine all labels
        if self.labels is None:
            self.labels = sorted(list(set(labels)))
        
        # Create model
        self._model = SetFitModel.from_pretrained(
            self.model_name,
            labels=self.labels
        )
        
        # Create datasets
        train_dataset = Dataset.from_dict({
            'text': texts,
            'label': labels
        })
        
        eval_dataset = None
        if eval_texts and eval_labels:
            eval_dataset = Dataset.from_dict({
                'text': eval_texts,
                'label': eval_labels
            })
        
        # Training arguments
        args = TrainingArguments(
            batch_size=batch_size,
            num_epochs=num_epochs,
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
        )
        
        # Create trainer
        self._trainer = Trainer(
            model=self._model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        # Train
        logger.info(f"Training SetFit with {len(texts)} samples, {num_epochs} epochs...")
        self._trainer.train()
        
        return self
    
    def predict(self, texts: List[str]) -> List[str]:
        """
        Predict labels for texts.
        
        Args:
            texts: List of texts
            
        Returns:
            List of predicted labels
        """
        if self._model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self._model.predict(texts).tolist()
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            texts: List of texts
            
        Returns:
            Array of probabilities
        """
        if self._model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self._model.predict_proba(texts)
    
    def save(self, path: str | Path):
        """Save model to disk."""
        if self._model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        path = Path(path)
        self._model.save_pretrained(str(path))
        
        # Save labels
        with open(path / 'labels.json', 'w') as f:
            json.dump(self.labels, f)
        
        logger.info(f"Saved SetFit model to {path}")
    
    @classmethod
    def load(cls, path: str | Path) -> 'SetFitClassifier':
        """Load model from disk."""
        try:
            from setfit import SetFitModel
        except ImportError:
            raise ImportError("setfit not installed")
        
        path = Path(path)
        
        # Load labels
        with open(path / 'labels.json') as f:
            labels = json.load(f)
        
        instance = cls(labels=labels)
        instance._model = SetFitModel.from_pretrained(str(path))
        
        logger.info(f"Loaded SetFit model from {path}")
        return instance


# ============================================================================
# Rule-Based Classifier (Improved)
# ============================================================================

# Sub-patterns for refining rule → L2 mapping
_CANCEL_RE = re.compile(r'отмен|не\s*смогу\s*прийти', re.IGNORECASE)
_GREETING_RE = re.compile(r'^(здравствуйте|добрый|привет|приветствую|алло)', re.IGNORECASE)
_GRATITUDE_RE = re.compile(r'^(спасибо|благодар)', re.IGNORECASE)
_FAREWELL_RE = re.compile(r'^(пока|до\s*свидания|всего\s*доброго)', re.IGNORECASE)
_CONFIRM_RE = re.compile(r'^(понял|ок|хорошо|ясно|понятно|ладно|да|нет)$', re.IGNORECASE)


class RuleBasedClassifier:
    """
    Improved rule-based classifier with priorities and negative lookbehind.
    
    Features:
    - Priority-based rule matching
    - Negative lookbehind patterns
    - Confidence estimation
    - Pattern coverage tracking
    - Returns L2 taxonomy labels (v5.0)
    
    Example:
        >>> clf = RuleBasedClassifier()
        >>> intent, confidence, matched = clf.classify("Хочу записаться на приём")
        >>> print(f"{intent}: {confidence:.2f}")
        new_appointment: 0.88
    """
    
    # Rule key → L2 taxonomy label (v5.0: alignment with taxonomy.py)
    RULE_TO_L2 = {
        'reschedule_cancel': 'reschedule',
        'booking': 'new_appointment',
        'complaint_primary': 'symptom',
        'price_question': 'price',
        'clinic_faq': 'clinic_info',
        'visit_recommendations': 'visit_prep',
        'followup_question': 'procedure',
        'negative_feedback': 'negative_general',
        'positive_feedback': 'positive_general',
        'other': 'unclear',
    }
    
    # Default rules with priorities (higher = checked first)
    DEFAULT_RULES = {
        # Priority 1: Reschedule/Cancel (must be checked before booking)
        'reschedule_cancel': {
            'priority': 100,
            'patterns': [
                r'перенес',
                r'перезапис',
                r'отмен',
                r'не\s*смогу\s*прийти',
                r'изменить\s*время',
                r'другой\s*день',
                r'другое\s*время',
                r'сдвинуть',
                r'передвинуть',
            ],
            'negative_patterns': [],  # No negative patterns
        },
        
        # Priority 2: Booking (after reschedule)
        'booking': {
            'priority': 90,
            'patterns': [
                r'(?<!пере)запис',  # Negative lookbehind: not "перезапис"
                r'хочу\s*(на\s*)?приём',
                r'нужен\s*приём',
                r'записаться',
                r'свободн\w*\s*(врем|окош|слот)',
                r'когда\s*можно\s*прийти',
                r'нужен\s*осмотр',
                r'попасть\s*к\s*врачу',
                r'попасть\s*на\s*приём',
            ],
            'negative_patterns': [
                r'перенес',
                r'отмен',
            ],
        },
        
        # Priority 3: Complaint (symptoms)
        'complaint_primary': {
            'priority': 85,
            'patterns': [
                r'бол[иеюя]т',
                r'ноет',
                r'опух[ло]',
                r'отёк',
                r'кровоточ',
                r'шата[еюя]тся',
                r'откол[оа]',
                r'выпал[аои]',
                r'трещин',
                r'чувствител',
                r'реагирует\s*на',
                r'воспал',
                r'гной',
                r'флюс',
                r'температур\w*\s*(и|с)\s*бол',
            ],
            'negative_patterns': [
                r'это\s*больно',  # This is a question, not complaint
                r'будет\s*больно',
            ],
        },
        
        # Priority 4: Price questions
        'price_question': {
            'priority': 80,
            'patterns': [
                r'скольк\w*\s*стои',
                r'цен[аыу]',
                r'прайс',
                r'стоимость',
                r'почём',
                r'во\s*сколько\s*обойд',
                r'рассрочк',
                r'скидк',
                r'акци[яию]',
            ],
            'negative_patterns': [],
        },
        
        # Priority 5: Clinic FAQ (about clinic)
        'clinic_faq': {
            'priority': 75,
            'patterns': [
                r'где\s*(вы\s*)?находи',
                r'как\s*добраться',
                r'адрес',
                r'парковк',
                r'работаете\s*(ли\s*)?(в|до)',
                r'график\s*работ',
                r'открыт[ыаое]',
                r'закрыт[ыаое]',
                r'выходн',
                r'есть\s*(ли\s*)?(рентген|томограф|микроскоп|лазер)',
                r'оплат\w*\s*карт',
                r'apple\s*pay',
                r'дмс|омс',
                r'врач\w*\s*работ',
                r'какие\s*врачи',
                r'есть\s*(ли\s*)?сайт',
                r'whatsapp',
                r'телефон',
            ],
            'negative_patterns': [],
        },
        
        # Priority 6: Visit recommendations
        'visit_recommendations': {
            'priority': 70,
            'patterns': [
                r'как\s*подготов',
                r'что\s*(нужно|взять)\s*перед',
                r'можно\s*(ли\s*)?есть\s*(перед|после)',
                r'нельзя\s*(после|перед)',
                r'ограничени[яе]',
                r'как\s*ухаживать',
                r'чем\s*полоск',
                r'через\s*сколько\s*(можно|снимать)',
                r'контрольн\w*\s*визит',
                r'повторн\w*\s*(визит|приём|осмотр)',
                r'когда\s*прийти\s*на\s*осмотр',
                r'что\s*делать\s*если',
                r'нормально\s*(ли\s*)?что',
            ],
            'negative_patterns': [],
        },
        
        # Priority 7: Followup questions
        'followup_question': {
            'priority': 65,
            'patterns': [
                r'это\s*больно',
                r'будет\s*больно',
                r'сколько\s*времени\s*займ',
                r'как\s*долго',
                r'что\s*если\s*не',
                r'альтернатив',
                r'гаранти[яию]',
                r'приживётся',
                r'прослужит',
                r'безопасн',
                r'рис[кн]',
                r'противопоказан',
                r'современн\w*\s*метод',
                r'в\s*чём\s*разниц',
                r'что\s*лучше',
                r'можно\s*(ли\s*)?без',
                r'обязательн\w*\s*ли',
            ],
            'negative_patterns': [],
        },
        
        # Priority 8: Negative Feedback (v4.0: high priority - don't miss complaints!)
        'negative_feedback': {
            'priority': 95,  # High priority for complaints
            'patterns': [
                r'недовол',
                r'разочаров',
                r'жалоб[аыу]',
                r'претензи',
                r'возмутител',
                r'безобраз',
                r'недопустим',
                r'ужасн\w*\s*(сервис|обслуживание)',
                r'плох\w*\s*(сервис|клиника|врач|работа)',
                r'стало\s*хуже',
                r'не\s*помогло',
                r'боль\s*не\s*прошла',
                r'снова\s*болит',
                r'пришлось\s*переделывать',
                r'нужно\s*переделать',
                r'осложнени',
                r'груб\w*(о|ость)?',
                r'нахам',
                r'невеж',
                r'непрофессионал',
                r'не\s*перезвонили',
                r'потеряли\s*запись',
                r'долго\s*ждал',
                r'опоздал',
                r'хочу\s*(оставить\s*)?жалобу',
                r'верните\s*деньги',
                r'требую\s*компенсац',
                r'хочу\s*возврат',
                r'руководств\w*',
                r'главврач',
                r'больше\s*не\s*приду',
                r'не\s*рекоменду',
            ],
            'negative_patterns': [],
        },
        
        # Priority 9: Positive Feedback (v4.0: NEW - thank you, recommendations)
        'positive_feedback': {
            'priority': 60,  # Lower than negative to prioritize complaints
            'patterns': [
                r'спасибо\s+(большое|огромное|вам)',
                r'благодар\w+\s+(вас|за)',
                r'очень\s+довол',
                r'довол\w+\s+(клиник|врач|результат|работ)',
                r'рекоменду\w+\s+(всем|друзья|знаком)',
                r'лучш\w+\s+(клиника|стоматолог|врач)',
                r'отличн\w+\s+(работ|результат|качеств)',
                r'прекрасн\w+\s+(работ|врач|результат)',
                r'профессионал\w+\s+(работ|подход)',
                r'внимательн\w+\s+(врач|персонал|отношен)',
                r'золот\w+\s+руки',
                r'буду\s+обращаться',
                r'приду\s+(к\s+вам|ещё|снова)',
                r'на\s+высш\w+\s+уровн',
                r'результат\s+превзош',
                r'качественн\w+\s+(работ|лечени)',
                r'удобн\w+\s+(запис|сервис)',
                r'быстро\s+записали',
                r'сразу\s+перезвонили',
                r'вежлив\w+\s+(персонал|администратор)',
                r'очень\s+рад',
                r'супер\s+(клиника|врач|сервис)',
            ],
            'negative_patterns': [
                r'не\s+рекоменду',  # Negative recommendation
                r'недовол',  # Dissatisfied
            ],
        },
        
        # Priority 10: Other (greetings, thanks, etc.)
        'other': {
            'priority': 10,
            'patterns': [
                r'^(здравствуйте|добрый\s*(день|вечер|утро)|привет|приветствую|алло)[\s!.]*$',
                r'^(спасибо|благодар|пока|до\s*свидания|всего\s*доброго|хорошего\s*дня)[\s!.]*',
                r'^(понял[аои]?|ок|хорошо|ясно|понятно|ладно|договорились)[\s!.,]*$',
                r'^(да|нет|не\s*знаю|подумаю)[\s!.,]*$',
            ],
            'negative_patterns': [],
        },
    }
    
    def __init__(self, rules: Optional[Dict] = None):
        """
        Initialize RuleBasedClassifier.
        
        Args:
            rules: Custom rules dict. If None, uses DEFAULT_RULES.
        """
        self.rules = rules or self.DEFAULT_RULES.copy()
        self._compile_rules()
        
        # Statistics
        self.stats = {
            'total_calls': 0,
            'matched': 0,
            'not_matched': 0,
            'by_intent': {},
        }
    
    def _compile_rules(self):
        """Compile regex patterns for efficiency."""
        self._compiled = {}
        
        for intent, config in self.rules.items():
            self._compiled[intent] = {
                'priority': config.get('priority', 50),
                'patterns': [
                    re.compile(p, re.IGNORECASE | re.UNICODE)
                    for p in config['patterns']
                ],
                'negative_patterns': [
                    re.compile(p, re.IGNORECASE | re.UNICODE)
                    for p in config.get('negative_patterns', [])
                ],
            }
    
    def classify(self, text: str) -> Tuple[Optional[str], float, bool]:
        """
        Classify single text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (intent, confidence, matched)
            - intent: Predicted intent or None if no match
            - confidence: Confidence score (0.0-1.0)
            - matched: Whether a rule matched
        """
        self.stats['total_calls'] += 1
        text_lower = text.lower().strip()
        
        # Sort rules by priority (descending)
        sorted_intents = sorted(
            self._compiled.items(),
            key=lambda x: x[1]['priority'],
            reverse=True
        )
        
        for intent, config in sorted_intents:
            # Check negative patterns first (exclusions)
            excluded = False
            for neg_pattern in config['negative_patterns']:
                if neg_pattern.search(text_lower):
                    excluded = True
                    break
            
            if excluded:
                continue
            
            # Check positive patterns
            for pattern in config['patterns']:
                if pattern.search(text_lower):
                    self.stats['matched'] += 1
                    
                    # Map rule key → L2 taxonomy label
                    l2_label = self._resolve_l2(intent, text_lower)
                    
                    self.stats['by_intent'][l2_label] = \
                        self.stats['by_intent'].get(l2_label, 0) + 1
                    
                    # Confidence based on priority and pattern specificity
                    confidence = min(0.95, 0.7 + config['priority'] / 500)
                    return l2_label, confidence, True
        
        self.stats['not_matched'] += 1
        return None, 0.0, False
    
    def _resolve_l2(self, rule_key: str, text_lower: str) -> str:
        """Resolve rule key to L2 taxonomy label with sub-pattern refinement."""
        if rule_key == 'reschedule_cancel':
            return 'cancel' if _CANCEL_RE.search(text_lower) else 'reschedule'
        if rule_key == 'other':
            if _GREETING_RE.search(text_lower):
                return 'greeting'
            if _GRATITUDE_RE.search(text_lower):
                return 'gratitude'
            if _FAREWELL_RE.search(text_lower):
                return 'farewell'
            if _CONFIRM_RE.search(text_lower):
                return 'confirmation'
            return 'unclear'
        return self.RULE_TO_L2.get(rule_key, rule_key)

    def classify_batch(
        self, texts: List[str]
    ) -> List[Tuple[Optional[str], float, bool]]:
        """Classify multiple texts."""
        return [self.classify(text) for text in texts]
    
    def get_coverage(self) -> float:
        """Get rule coverage (% of texts matched)."""
        if self.stats['total_calls'] == 0:
            return 0.0
        return self.stats['matched'] / self.stats['total_calls']
    
    def get_stats(self) -> Dict:
        """Get classification statistics."""
        return {
            **self.stats,
            'coverage': self.get_coverage(),
        }
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            'total_calls': 0,
            'matched': 0,
            'not_matched': 0,
            'by_intent': {},
        }


# ============================================================================
# Cascade Classifier (Rule → ML → LLM)
# ============================================================================

class CascadeClassifier:
    """
    Cascade classifier combining Rule-based, ML, and LLM.
    
    Architecture:
    1. Rule-based: Fast, high-confidence matches (~50% coverage)
    2. ML (Embedding): Medium confidence matches (~40% remaining)
    3. LLM: Fallback for complex cases (~10% remaining)
    
    Example:
        >>> rule_clf = RuleBasedClassifier()
        >>> ml_clf = EmbeddingClassifier()
        >>> ml_clf.fit(train_texts, train_labels)
        >>> cascade = CascadeClassifier(rule_clf, ml_clf, llm_client)
        >>> intent, source, confidence = cascade.classify("Болит зуб")
    """
    
    def __init__(
        self,
        rule_classifier: RuleBasedClassifier,
        ml_classifier: EmbeddingClassifier,
        llm_client: Optional[Any] = None,
        ml_threshold: float = _ML_THRESHOLD,
        llm_threshold: float = _LLM_THRESHOLD,
        calibrate: bool = False,  # v4.0: добавлен флаг калибровки
        calibration_cv: int = 5,  # v4.0: кол-во фолдов для калибровки
    ):
        """
        Initialize CascadeClassifier.
        
        Args:
            rule_classifier: Rule-based classifier
            ml_classifier: Trained embedding classifier
            llm_client: LLM client (optional, for fallback)
            ml_threshold: Confidence threshold for ML predictions (SSoT: taxonomy high)
            llm_threshold: Confidence threshold below which to use LLM (SSoT: taxonomy low)
            calibrate: If True, apply CalibratedClassifierCV to ML layer (v4.0)
            calibration_cv: Number of CV folds for calibration (default: 5)
        """
        self.rules = rule_classifier
        self.ml = ml_classifier
        self.llm = llm_client
        self.ml_threshold = ml_threshold
        self.llm_threshold = llm_threshold
        self.calibrate = calibrate
        self.calibration_cv = calibration_cv
        self._calibrated_ml = None  # Will store calibrated model if enabled
        
        # Statistics
        self.stats = {
            'total': 0,
            'rule_hits': 0,
            'ml_hits': 0,
            'llm_hits': 0,
            'llm_calls': 0,
            'calibrated': calibrate,  # v4.0: track if calibration is used
        }
    
    def fit_calibration(self, X_cal: list, y_cal: list):
        """
        Fit calibration layer on calibration data (v4.0).
        
        Uses CalibratedClassifierCV with isotonic regression for better
        probability estimates.
        
        Args:
            X_cal: Calibration texts
            y_cal: Calibration labels
        """
        if not self.calibrate:
            logger.warning("Calibration disabled, skipping fit_calibration()")
            return
        
        try:
            from sklearn.calibration import CalibratedClassifierCV
            
            # Get base classifier from EmbeddingClassifier
            if hasattr(self.ml, 'classifier') and self.ml.classifier is not None:
                base_clf = self.ml.classifier
                # Apply isotonic calibration
                self._calibrated_ml = CalibratedClassifierCV(
                    base_clf,
                    method='isotonic',
                    cv=self.calibration_cv
                )
                
                # Need embeddings for calibration
                if hasattr(self.ml, '_embed'):
                    X_emb = self.ml._embed(X_cal)
                    self._calibrated_ml.fit(X_emb, y_cal)
                    logger.info(f"Calibrated ML layer with {len(X_cal)} samples, cv={self.calibration_cv}")
                else:
                    logger.warning("EmbeddingClassifier has no _embed method, calibration skipped")
            else:
                logger.warning("ML classifier not found, calibration skipped")
        except ImportError:
            logger.warning("sklearn.calibration not available, calibration disabled")
        except Exception as e:
            logger.warning(f"Calibration failed: {e}")
    
    def _classify_with_llm(self, text: str) -> Tuple[str, float]:
        """Call LLM for classification."""
        if self.llm is None:
            return 'other', 0.5
        
        self.stats['llm_calls'] += 1
        
        # v4.0: updated prompt with feedback categories
        prompt = f"""Классифицируй сообщение пациента стоматологии в одну из категорий:
- booking: запись на приём
- complaint_primary: жалоба, симптомы
- price_question: вопрос о цене
- reschedule_cancel: перенос/отмена записи
- clinic_faq: вопросы о клинике
- visit_recommendations: рекомендации до/после
- followup_question: уточняющий вопрос о лечении
- negative_feedback: жалоба на сервис/качество/персонал
- positive_feedback: благодарность, положительный отзыв
- other: приветствие, прочее

Сообщение: "{text}"

Ответь ТОЛЬКО названием категории (одно слово, например: booking)."""

        try:
            response = self.llm.generate(prompt, max_tokens=20)
            intent = response.strip().lower()
            
            # Validate intent (v4.0: added feedback)
            valid_intents = [
                'booking', 'complaint_primary', 'price_question',
                'reschedule_cancel', 'clinic_faq', 'visit_recommendations',
                'followup_question', 'negative_feedback', 'positive_feedback', 'other'
            ]
            if intent not in valid_intents:
                intent = 'other'
            
            return intent, 0.8
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
            return 'other', 0.5
    
    def classify(self, text: str) -> Tuple[str, str, float]:
        """
        Classify text using cascade approach.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (intent, source, confidence)
            - intent: Predicted intent
            - source: 'rule', 'ml', or 'llm'
            - confidence: Confidence score
        """
        self.stats['total'] += 1
        
        # Layer 1: Rule-based
        intent, confidence, matched = self.rules.classify(text)
        if matched and confidence >= 0.7:
            self.stats['rule_hits'] += 1
            return intent, 'rule', confidence
        
        # Layer 2: ML (Embedding)
        results = self.ml.predict_with_confidence([text])
        ml_intent, ml_confidence = results[0]
        
        if ml_confidence >= self.ml_threshold:
            self.stats['ml_hits'] += 1
            return ml_intent, 'ml', ml_confidence
        
        # Layer 3: LLM fallback (if available and confidence is low)
        if self.llm is not None and ml_confidence < self.llm_threshold:
            llm_intent, llm_confidence = self._classify_with_llm(text)
            self.stats['llm_hits'] += 1
            return llm_intent, 'llm', llm_confidence
        
        # Use ML result if no LLM or above llm_threshold
        self.stats['ml_hits'] += 1
        return ml_intent, 'ml', ml_confidence
    
    def classify_batch(
        self, texts: List[str]
    ) -> List[Tuple[str, str, float]]:
        """Classify multiple texts."""
        return [self.classify(text) for text in texts]
    
    def get_stats(self) -> Dict:
        """Get cascade statistics."""
        total = max(1, self.stats['total'])
        return {
            **self.stats,
            'rule_pct': self.stats['rule_hits'] / total * 100,
            'ml_pct': self.stats['ml_hits'] / total * 100,
            'llm_pct': self.stats['llm_hits'] / total * 100,
            'llm_savings': (total - self.stats['llm_calls']) / total * 100,
        }
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            'total': 0,
            'rule_hits': 0,
            'ml_hits': 0,
            'llm_hits': 0,
            'llm_calls': 0,
        }
        self.rules.reset_stats()


# ============================================================================
# GridSearch utilities
# ============================================================================

def create_tfidf_pipeline(classifier_type: str = 'svc'):
    """
    Create TF-IDF + Classifier pipeline for GridSearchCV.
    
    Args:
        classifier_type: 'svc', 'logistic', or 'rf'
        
    Returns:
        sklearn Pipeline
    """
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import RandomForestClassifier
    
    classifiers = {
        'svc': LinearSVC(dual='auto'),
        'logistic': LogisticRegression(max_iter=1000),
        'rf': RandomForestClassifier(),
    }
    
    return Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', classifiers.get(classifier_type, LinearSVC(dual='auto'))),
    ])


def get_tfidf_param_grid(classifier_type: str = 'svc') -> Dict:
    """
    Get parameter grid for TF-IDF pipeline.
    
    Args:
        classifier_type: 'svc', 'logistic', or 'rf'
        
    Returns:
        Parameter grid dict
    """
    base_grid = {
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'tfidf__max_features': [1000, 3000, 5000],
        'tfidf__max_df': [0.75, 0.9, 1.0],
        'tfidf__min_df': [1, 2],
        'tfidf__sublinear_tf': [True, False],
    }
    
    classifier_grids = {
        'svc': {
            'clf__C': [0.1, 1.0, 10.0],
            'clf__class_weight': [None, 'balanced'],
        },
        'logistic': {
            'clf__C': [0.1, 1.0, 10.0],
            'clf__class_weight': [None, 'balanced'],
        },
        'rf': {
            'clf__n_estimators': [50, 100],
            'clf__max_depth': [5, 10, None],
            'clf__class_weight': [None, 'balanced'],
        },
    }
    
    return {**base_grid, **classifier_grids.get(classifier_type, {})}


def run_gridsearch(
    X_train: List[str],
    y_train: List[str],
    classifier_type: str = 'svc',
    cv: int = 5,
    scoring: str = 'f1_macro',
    n_jobs: int = -1
) -> Tuple[Any, Dict]:
    """
    Run GridSearchCV for TF-IDF pipeline.
    
    Args:
        X_train: Training texts
        y_train: Training labels
        classifier_type: 'svc', 'logistic', or 'rf'
        cv: Number of cross-validation folds
        scoring: Scoring metric
        n_jobs: Number of parallel jobs
        
    Returns:
        Tuple of (best_estimator, results_dict)
    """
    from sklearn.model_selection import GridSearchCV
    
    pipeline = create_tfidf_pipeline(classifier_type)
    param_grid = get_tfidf_param_grid(classifier_type)
    
    logger.info(f"Starting GridSearchCV with {len(param_grid)} param combinations...")
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    results = {
        'best_score': grid_search.best_score_,
        'best_params': grid_search.best_params_,
        'cv_results': {
            'mean_test_score': grid_search.cv_results_['mean_test_score'].tolist(),
            'std_test_score': grid_search.cv_results_['std_test_score'].tolist(),
            'params': grid_search.cv_results_['params'],
        }
    }
    
    logger.info(f"Best score: {grid_search.best_score_:.4f}")
    logger.info(f"Best params: {grid_search.best_params_}")
    
    return grid_search.best_estimator_, results


if __name__ == "__main__":
    # Quick test
    print("Testing RuleBasedClassifier...")
    rule_clf = RuleBasedClassifier()
    
    test_cases = [
        "Хочу записаться на приём",
        "Перенесите мою запись",
        "Сколько стоит чистка?",
        "Болит зуб уже неделю",
        "Где вы находитесь?",
        "Как подготовиться к удалению?",
        "А это больно?",
        "Здравствуйте",
    ]
    
    for text in test_cases:
        intent, conf, matched = rule_clf.classify(text)
        print(f"  '{text[:40]}...' -> {intent or 'NO MATCH'} ({conf:.2f})")
    
    print(f"\nCoverage: {rule_clf.get_coverage():.1%}")
    print(f"Stats: {rule_clf.get_stats()}")
