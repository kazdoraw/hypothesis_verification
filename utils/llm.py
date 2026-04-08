"""
Together AI LLM Client for baseline experiments.
"""

import os
import json
import random
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import yaml


def load_together_config(config_path: Optional[str | Path] = None) -> dict:
    """
    Load Together AI config from yaml file.
    
    Args:
        config_path: Path to config file. Defaults to configs/together_config.yaml
        
    Returns:
        Config dict with api_key and models
    """
    if config_path is None:
        # Try to find config relative to this file or cwd
        possible_paths = [
            Path(__file__).parent.parent / "configs" / "together_config.yaml",
            Path("configs/together_config.yaml"),
            Path("study/configs/together_config.yaml"),
        ]
        for p in possible_paths:
            if p.exists():
                config_path = p
                break
    
    if config_path is None or not Path(config_path).exists():
        return {}
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config or {}


def get_api_key(config_path: Optional[str | Path] = None) -> Optional[str]:
    """
    Get Together AI API key from config or environment.
    
    Priority:
    1. Config file (api_key_env field - can be direct key or env var name)
    2. TOGETHER_API_KEY environment variable
    """
    config = load_together_config(config_path)
    
    # Check config file
    api_key_value = config.get("api_key_env", "")
    
    if api_key_value:
        # If it looks like an API key (starts with tgp_), use directly
        if api_key_value.startswith("tgp_"):
            return api_key_value
        # Otherwise treat as env var name
        env_key = os.getenv(api_key_value)
        if env_key:
            return env_key
    
    # Fallback to environment variable
    return os.getenv("TOGETHER_API_KEY")


@dataclass
class LLMResponse:
    """Response from LLM call."""
    content: str
    tokens_used: int = 0
    model: str = ""
    raw_response: Optional[dict] = None


@dataclass
class LLMStats:
    """Statistics for LLM usage."""
    total_calls: int = 0
    total_tokens: int = 0
    calls_log: list = field(default_factory=list)
    
    def log_call(self, tokens: int, model: str):
        self.total_calls += 1
        self.total_tokens += tokens
        self.calls_log.append({"tokens": tokens, "model": model})


class TogetherLLM:
    """
    Together AI LLM Client.
    
    Loads API key from configs/together_config.yaml or environment.
    Falls back to simulator only if explicitly requested or no key found.
    """
    
    INTENT_LABELS = [
        "booking", "complaint_primary", "followup_question",
        "price_question", "reschedule_cancel", "other"
    ]
    
    def __init__(
        self, 
        model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        api_key: Optional[str] = None,
        use_simulator: bool = False,
        config_path: Optional[str | Path] = None
    ):
        self.model = model
        self.config = load_together_config(config_path)
        
        # Get API key: explicit > config > env
        self.api_key = api_key or get_api_key(config_path)
        
        # Only use simulator if explicitly requested
        self.use_simulator = use_simulator
        self.stats = LLMStats()
        self.client = None
        
        if not self.use_simulator:
            if not self.api_key:
                print("Warning: No API key found. Set use_simulator=True or add key to config.")
                print("Falling back to simulator.")
                self.use_simulator = True
            else:
                # Using requests directly (Together SDK has pydantic bugs)
                try:
                    import requests
                    self._requests = requests
                    print(f"Together AI client initialized (via requests). Model: {self.model}")
                except ImportError:
                    print("Warning: requests package not installed.")
                    print("Falling back to simulator.")
                    self.use_simulator = True
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (~4 chars per token for Russian)."""
        return max(1, len(text) // 4)
    
    def _call_api(self, messages: list, temperature: float = 0.1, max_tokens: int = 256) -> LLMResponse:
        """Call Together AI API via direct HTTP request."""
        if self.use_simulator:
            raise RuntimeError("API not available, use simulator methods")
        
        response = self._requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            },
            timeout=60
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"API error {response.status_code}: {response.text}")
        
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {}).get("total_tokens", self._estimate_tokens(content))
        
        self.stats.log_call(tokens, self.model)
        
        return LLMResponse(
            content=content,
            tokens_used=tokens,
            model=self.model,
            raw_response=data
        )
    
    # =========================================================================
    # D1: Intent Classification
    # =========================================================================
    
    def classify_intent(self, text: str, labels: Optional[list] = None) -> dict:
        """
        Classify intent using LLM (Baseline A for D1).
        
        Returns:
            dict with keys: intent, confidence, tokens_used
        """
        labels = labels or self.INTENT_LABELS
        
        if self.use_simulator:
            return self._simulate_classify_intent(text, labels)
        
        prompt = f"""Классифицируй сообщение пациента стоматологической клиники.

Возможные классы: {', '.join(labels)}

Сообщение: "{text}"

Верни JSON: {{"intent": "класс", "confidence": 0.0-1.0, "reasoning": "краткое обоснование"}}"""
        
        response = self._call_api(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=128
        )
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[^}]+\}', response.content)
            if json_match:
                result = json.loads(json_match.group())
                result["tokens_used"] = response.tokens_used
                return result
        except (json.JSONDecodeError, AttributeError):
            pass
        
        return {
            "intent": "other",
            "confidence": 0.5,
            "tokens_used": response.tokens_used,
            "raw": response.content
        }
    
    def _simulate_classify_intent(self, text: str, labels: list) -> dict:
        """
        Simulate LLM classification with realistic error patterns.
        
        Simulates F1 ~0.75-0.80 with typical confusions:
        - complaint_primary <-> other
        - booking <-> followup_question
        """
        text_lower = text.lower()
        
        # Deterministic patterns (high confidence)
        if any(w in text_lower for w in ["записаться", "запишите", "хочу на приём", "свободное время"]):
            true_intent = "booking"
        elif any(w in text_lower for w in ["сколько стоит", "цена", "стоимость", "прайс"]):
            true_intent = "price_question"
        elif any(w in text_lower for w in ["перенести", "отменить", "отмена"]):
            true_intent = "reschedule_cancel"
        elif any(w in text_lower for w in ["болит", "боль", "ноет", "опухл", "кровоточ"]):
            true_intent = "complaint_primary"
        elif any(w in text_lower for w in ["а если", "а что", "уточните", "подскажите"]):
            true_intent = "followup_question"
        else:
            true_intent = "other"
        
        # Simulate LLM errors (~20-25% error rate)
        error_matrix = {
            "complaint_primary": {"other": 0.15, "followup_question": 0.05},
            "other": {"complaint_primary": 0.10, "followup_question": 0.08},
            "booking": {"followup_question": 0.08},
            "followup_question": {"booking": 0.05, "other": 0.07},
            "price_question": {"other": 0.03},
            "reschedule_cancel": {"booking": 0.05},
        }
        
        predicted_intent = true_intent
        if true_intent in error_matrix:
            rand = random.random()
            cumulative = 0
            for wrong_intent, prob in error_matrix[true_intent].items():
                cumulative += prob
                if rand < cumulative:
                    predicted_intent = wrong_intent
                    break
        
        confidence = random.uniform(0.65, 0.95) if predicted_intent == true_intent else random.uniform(0.45, 0.75)
        tokens = self._estimate_tokens(text) + random.randint(20, 50)
        
        self.stats.log_call(tokens, "simulator")
        
        return {
            "intent": predicted_intent,
            "confidence": round(confidence, 2),
            "tokens_used": tokens,
            "_true_intent": true_intent,  # For evaluation
            "_is_simulated": True
        }
    
    # =========================================================================
    # D2: Anamnesis Collection
    # =========================================================================
    
    def collect_anamnesis_free(self, history: list, patient_complaint: str) -> dict:
        """
        Collect anamnesis using free-form LLM dialog (Baseline A for D2).
        
        Args:
            history: List of {"role": "user"|"assistant", "text": str}
            patient_complaint: Initial complaint text
            
        Returns:
            dict with next_question or filled_data
        """
        if self.use_simulator:
            return self._simulate_anamnesis_free(history, patient_complaint)
        
        history_text = "\n".join([f"{m['role']}: {m['text']}" for m in history])
        
        prompt = f"""Ты ассистент стоматологической клиники. Собери анамнез у пациента.

Начальная жалоба: "{patient_complaint}"

История диалога:
{history_text}

Задай следующий уточняющий вопрос или, если информации достаточно, верни JSON с собранными данными:
{{"is_complete": true/false, "next_question": "вопрос" или null, "collected_data": {{...}}}}"""
        
        response = self._call_api(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=256
        )
        
        try:
            json_match = re.search(r'\{[^}]+\}', response.content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                result["tokens_used"] = response.tokens_used
                return result
        except (json.JSONDecodeError, AttributeError):
            pass
        
        return {
            "is_complete": False,
            "next_question": response.content,
            "tokens_used": response.tokens_used
        }
    
    def _simulate_anamnesis_free(self, history: list, patient_complaint: str) -> dict:
        """
        Simulate free-form anamnesis collection.
        
        Simulates ~30-40% field skip rate (baseline behavior).
        """
        # Standard questions pool (LLM might ask in any order, skip some)
        question_pool = [
            "Где именно болит? Можете показать или описать расположение?",
            "Как давно это беспокоит?",
            "Насколько сильная боль по шкале от 1 до 10?",
            "Есть ли у вас аллергии на лекарства?",
            "Принимаете ли вы какие-то препараты?",
            "Есть ли хронические заболевания?",
        ]
        
        turn = len(history)
        tokens = random.randint(30, 80)
        self.stats.log_call(tokens, "simulator")
        
        # Simulate stopping too early (30% chance after 3 turns)
        if turn >= 3 and random.random() < 0.3:
            return {
                "is_complete": True,
                "next_question": None,
                "collected_data": {"partial": True},
                "tokens_used": tokens,
                "_is_simulated": True
            }
        
        # Simulate skipping questions (40% chance to skip a logical question)
        if turn < len(question_pool):
            # 40% chance to skip to a random later question
            if random.random() < 0.4 and turn < len(question_pool) - 1:
                question = random.choice(question_pool[turn+1:])
            else:
                question = question_pool[turn]
        else:
            return {
                "is_complete": True,
                "next_question": None,
                "collected_data": {},
                "tokens_used": tokens,
                "_is_simulated": True
            }
        
        return {
            "is_complete": False,
            "next_question": question,
            "tokens_used": tokens,
            "_is_simulated": True
        }
    
    def formulate_question(self, field: str, context: dict) -> str:
        """
        Generate natural language question for a specific field (for State Machine).
        
        Args:
            field: Field name to ask about
            context: Current intake context
            
        Returns:
            Natural language question string
        """
        field_questions = {
            "chief_complaint": "Что вас беспокоит? Опишите вашу проблему.",
            "localization": "Где именно болит или беспокоит? Укажите зуб или область.",
            "duration": "Как давно это беспокоит?",
            "intensity": "Оцените интенсивность боли от 1 до 10, где 10 — невыносимая боль.",
            "onset": "Когда это началось? Было ли что-то, что спровоцировало?",
            "triggers": "Что усиливает боль? (холодное, горячее, жевание)",
            "relievers": "Что облегчает состояние?",
            "temperature": "Есть ли повышенная температура?",
            "swelling": "Есть ли отёк или припухлость?",
            "bleeding": "Есть ли кровоточивость?",
            "allergies": "Есть ли у вас аллергии на лекарства или материалы?",
            "chronic_conditions": "Есть ли хронические заболевания?",
            "medications": "Принимаете ли вы сейчас какие-либо лекарства?",
            "pregnancy": "Есть ли беременность? (для женщин)",
            "desired_outcome": "Какой результат вы хотели бы получить?",
            "last_visit": "Когда вы последний раз были у стоматолога?",
            "bite_issues": "Есть ли проблемы с прикусом?",
        }
        
        return field_questions.get(field, f"Расскажите подробнее о {field}.")
    
    def extract_field_value(self, response: str, field: str) -> Optional[str]:
        """
        Extract field value from user response.
        
        Args:
            response: User's response text
            field: Field name being extracted
            
        Returns:
            Extracted value or None if not found
        """
        response = response.strip()
        
        if not response or response.lower() in ["не знаю", "не помню", "нет", "-"]:
            return None
        
        # Intensity: extract number 1-10
        if field == "intensity":
            numbers = re.findall(r'\d+', response)
            if numbers:
                val = int(numbers[0])
                if 1 <= val <= 10:
                    return str(val)
        
        # Boolean fields
        if field in ["temperature", "swelling", "bleeding", "pregnancy"]:
            if any(w in response.lower() for w in ["да", "есть", "имеется"]):
                return "да"
            elif any(w in response.lower() for w in ["нет", "отсутствует"]):
                return "нет"
        
        # Default: return as-is if non-empty
        if len(response) > 1:
            return response
        
        return None
    
    def reset_stats(self):
        """Reset usage statistics."""
        self.stats = LLMStats()
    
    def get_stats(self) -> dict:
        """Get usage statistics."""
        return {
            "total_calls": self.stats.total_calls,
            "total_tokens": self.stats.total_tokens,
            "avg_tokens_per_call": (
                self.stats.total_tokens / self.stats.total_calls 
                if self.stats.total_calls > 0 else 0
            ),
            "calls_log": self.stats.calls_log
        }
