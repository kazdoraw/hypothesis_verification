"""
Production utilities for deploying trained classifiers.

This module provides:
- Model loading utilities
- Production-ready classification functions
- Model versioning and metadata
"""

import json
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from datetime import datetime

from .taxonomy import L2_TO_L1, L2_TO_SCENARIO, INTENT_LABELS_L2

logger = logging.getLogger(__name__)


class ProductionClassifier:
    """
    Production-ready wrapper for intent classification.
    
    Supports multiple model types:
    - TF-IDF (joblib)
    - EmbeddingClassifier
    - SetFitClassifier
    - CascadeClassifier
    
    Example:
        >>> clf = ProductionClassifier.load('models/d1_best_classifier')
        >>> intent, confidence = clf.classify("Хочу записаться на приём")
        >>> print(f"{intent}: {confidence:.2f}")
    """
    
    @staticmethod
    def get_flow(intent_l2: str) -> str:
        """Get production flow for L2 intent (SSoT: taxonomy.L2_TO_SCENARIO)."""
        return L2_TO_SCENARIO.get(intent_l2, "faq_flow")
    
    def __init__(self, model, model_type: str, metadata: Optional[Dict] = None):
        """
        Initialize production classifier.
        
        Args:
            model: Loaded model object
            model_type: Type of model ('tfidf', 'embedding', 'setfit', 'cascade')
            metadata: Optional metadata dict
        """
        self.model = model
        self.model_type = model_type
        self.metadata = metadata or {}
        self._load_time = datetime.now()
    
    def classify(self, text: str) -> Tuple[str, float]:
        """
        Classify single text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (intent, confidence)
        """
        if self.model_type == 'tfidf':
            intent = self.model.predict([text])[0]
            # TF-IDF models may not have predict_proba
            try:
                proba = self.model.predict_proba([text])[0]
                confidence = max(proba)
            except AttributeError:
                confidence = 0.8  # Default for SVC
            return intent, confidence
        
        elif self.model_type == 'embedding':
            results = self.model.predict_with_confidence([text])
            return results[0]
        
        elif self.model_type == 'setfit':
            intent = self.model.predict([text])[0]
            try:
                proba = self.model.predict_proba([text])[0]
                confidence = max(proba)
            except:
                confidence = 0.8
            return intent, confidence
        
        elif self.model_type == 'cascade':
            intent, source, confidence = self.model.classify(text)
            return intent, confidence
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def classify_with_scenario(self, text: str) -> Dict[str, Any]:
        """
        Classify text and return full result with scenario.
        
        Args:
            text: Input text
            
        Returns:
            Dict with intent, scenario, confidence, model_type
        """
        intent, confidence = self.classify(text)
        scenario = self.get_flow(intent)
        
        return {
            'intent': intent,
            'scenario': scenario,
            'confidence': confidence,
            'model_type': self.model_type,
        }
    
    def batch_classify(self, texts: list) -> list:
        """
        Classify multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of (intent, confidence) tuples
        """
        return [self.classify(text) for text in texts]
    
    def get_info(self) -> Dict[str, Any]:
        """Get model info and metadata."""
        return {
            'model_type': self.model_type,
            'load_time': self._load_time.isoformat(),
            'metadata': self.metadata,
            'supported_intents': INTENT_LABELS_L2,
        }
    
    @classmethod
    def load(cls, path: str | Path) -> 'ProductionClassifier':
        """
        Load classifier from disk.
        
        Automatically detects model type from files.
        
        Args:
            path: Path to model directory or file
            
        Returns:
            ProductionClassifier instance
        """
        path = Path(path)
        
        # Detect model type
        if path.suffix == '.joblib' or (path / 'model.joblib').exists():
            return cls._load_tfidf(path)
        elif (path / 'meta.json').exists():
            return cls._load_embedding(path)
        elif (path / 'labels.json').exists():
            return cls._load_setfit(path)
        else:
            raise ValueError(f"Cannot detect model type at {path}")
    
    @classmethod
    def _load_tfidf(cls, path: Path) -> 'ProductionClassifier':
        """Load TF-IDF model."""
        import joblib
        
        if path.suffix == '.joblib':
            model = joblib.load(path)
        else:
            model = joblib.load(path / 'model.joblib')
        
        metadata = {}
        meta_path = path.parent / 'meta.json' if path.suffix else path / 'meta.json'
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)
        
        logger.info(f"Loaded TF-IDF model from {path}")
        return cls(model, 'tfidf', metadata)
    
    @classmethod
    def _load_embedding(cls, path: Path) -> 'ProductionClassifier':
        """Load EmbeddingClassifier."""
        from .classifiers import EmbeddingClassifier
        
        model = EmbeddingClassifier.load(path)
        
        with open(path / 'meta.json') as f:
            metadata = json.load(f)
        
        logger.info(f"Loaded EmbeddingClassifier from {path}")
        return cls(model, 'embedding', metadata)
    
    @classmethod
    def _load_setfit(cls, path: Path) -> 'ProductionClassifier':
        """Load SetFitClassifier."""
        from .classifiers import SetFitClassifier
        
        model = SetFitClassifier.load(path)
        
        metadata = {}
        if (path / 'labels.json').exists():
            with open(path / 'labels.json') as f:
                metadata['labels'] = json.load(f)
        
        logger.info(f"Loaded SetFitClassifier from {path}")
        return cls(model, 'setfit', metadata)


def export_model_for_production(
    model,
    model_type: str,
    output_dir: str | Path,
    model_name: str = 'd1_classifier',
    metrics: Optional[Dict] = None,
    config: Optional[Dict] = None
) -> Path:
    """
    Export trained model for production use.
    
    Creates a directory with:
    - Model files
    - Metadata JSON
    - Config JSON
    - README with usage instructions
    
    Args:
        model: Trained model object
        model_type: 'tfidf', 'embedding', 'setfit', or 'cascade'
        output_dir: Output directory
        model_name: Name for the exported model
        metrics: Optional metrics dict from training
        config: Optional config dict
        
    Returns:
        Path to exported model directory
    """
    import joblib
    
    output_dir = Path(output_dir)
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model based on type
    if model_type == 'tfidf':
        model_path = model_dir / 'model.joblib'
        joblib.dump(model, model_path)
    elif model_type == 'embedding':
        model.save(model_dir)
    elif model_type == 'setfit':
        model.save(model_dir)
    else:
        # Generic joblib save
        model_path = model_dir / 'model.joblib'
        joblib.dump(model, model_path)
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'model_type': model_type,
        'export_time': datetime.now().isoformat(),
        'metrics': metrics or {},
        'config': config or {},
        'intent_labels': INTENT_LABELS_L2,
    }
    
    with open(model_dir / 'meta.json', 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Create README
    readme = f"""# {model_name}

## Model Type
{model_type}

## Export Time
{metadata['export_time']}

## Supported Intents
{', '.join(metadata['intent_labels'])}

## Usage

```python
from utils.production import ProductionClassifier

# Load model
clf = ProductionClassifier.load('{model_dir}')

# Classify text
intent, confidence = clf.classify("Хочу записаться на приём")
print(f"Intent: {{intent}}, Confidence: {{confidence:.2f}}")

# Get full result with scenario
result = clf.classify_with_scenario("Сколько стоит чистка?")
print(result)
# {{'intent': 'price_question', 'scenario': 'faq_flow', 'confidence': 0.95}}
```

## Metrics
{json.dumps(metrics, indent=2, ensure_ascii=False) if metrics else 'Not available'}
"""
    
    with open(model_dir / 'README.md', 'w', encoding='utf-8') as f:
        f.write(readme)
    
    logger.info(f"Exported model to {model_dir}")
    return model_dir


def validate_production_model(model_dir: str | Path) -> Dict[str, Any]:
    """
    Validate exported production model.
    
    Checks:
    - Model can be loaded
    - Classification works
    - All intents are recognized
    
    Args:
        model_dir: Path to model directory
        
    Returns:
        Validation results dict
    """
    model_dir = Path(model_dir)
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'checks': {}
    }
    
    # Check model can be loaded
    try:
        clf = ProductionClassifier.load(model_dir)
        results['checks']['load'] = True
    except Exception as e:
        results['valid'] = False
        results['errors'].append(f"Failed to load model: {e}")
        results['checks']['load'] = False
        return results
    
    # Check classification works
    test_texts = [
        ("Хочу записаться на приём", "new_appointment"),
        ("Болит зуб", "symptom"),
        ("Сколько стоит чистка?", "price"),
        ("Здравствуйте", "greeting"),
    ]
    
    correct = 0
    for text, expected in test_texts:
        try:
            intent, conf = clf.classify(text)
            if intent == expected:
                correct += 1
        except Exception as e:
            results['errors'].append(f"Classification failed for '{text}': {e}")
    
    results['checks']['classification'] = correct == len(test_texts)
    results['checks']['accuracy'] = correct / len(test_texts)
    
    if correct < len(test_texts):
        results['warnings'].append(f"Only {correct}/{len(test_texts)} test cases passed")
    
    # Check metadata
    meta_path = model_dir / 'meta.json'
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        results['checks']['metadata'] = True
        results['model_info'] = meta
    else:
        results['warnings'].append("No metadata file found")
        results['checks']['metadata'] = False
    
    return results


if __name__ == "__main__":
    # Quick test
    print("Testing production utilities...")
    
    # Test with a simple example
    test_path = Path("models/d1_best_classifier.joblib")
    if test_path.exists():
        clf = ProductionClassifier.load(test_path)
        print(f"Loaded: {clf.get_info()}")
        
        result = clf.classify_with_scenario("Хочу записаться на чистку")
        print(f"Result: {result}")
    else:
        print(f"Test model not found at {test_path}")
