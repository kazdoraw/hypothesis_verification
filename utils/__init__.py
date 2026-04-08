"""
Utils module for DS experiments.

Version 5.0: Two-level classification (L1/L2) with 12,000 samples, 24 L2 classes.
"""

# Original data module (backward compatibility)
from .data import (
    generate_d1_dataset, generate_d2_cases, load_or_generate_d1, load_or_generate_d2,
    augment_dataset, generate_d1_with_augmentation, INTENT_LABELS, INTENT_TO_SCENARIO
)

# New v2 data module with L1/L2 support
from .data_v2 import (
    generate_d1_dataset_v2,
    generate_stratified_split,
    save_dataset,
    load_dataset_v2,
    get_dataset_stats,
    MESSAGE_TEMPLATES_L2,
    VARIATION_TEMPLATES,
)

# Taxonomy (L1/L2 structure)
from .taxonomy import (
    INTENT_LABELS_L1,
    INTENT_LABELS_L2,
    L2_TO_L1,
    L1_TO_PRODUCTION_NODE,
    L2_TO_SCENARIO as L2_TO_SCENARIO_NEW,
    CLASS_WEIGHTS_L1,
    CLASS_WEIGHTS_L2,
    CONFIDENCE_THRESHOLDS,
    ClassificationResult,
    TaxonomyConfig,
    get_l1_from_l2,
    get_production_node,
    get_scenario,
    get_l2_classes_for_l1,
    load_taxonomy,
    get_sklearn_class_weights,
)

# Entity extraction
from .entity_extractor import (
    EntityExtractor,
    Entity,
    ExtractionResult,
    extract_entities,
    has_doctor_mention,
    has_date_mention,
    get_mentioned_procedure,
    get_entity_extractor,
)

# Augmentation
from .augmentation import (
    TextAugmenter,
    AugmentationConfig,
    AugmentedSample,
    augment_dataset_advanced,
    create_augmented_variants,
    generate_typo_variants,
    generate_abbreviation_variants,
    generate_case_variants,
    generate_punctuation_variants,
    get_augmenter,
)

# LLM generation
from .llm_generator import (
    LLMDataGenerator,
    GeneratedSample,
    QualityChecker,
    generate_llm_samples,
)

# Context-aware prompts
from .context_prompts import (
    DialogContext,
    EnrichedContext,
    PromptBuilder,
    ContextInterpreter,
    CONTEXT_PROMPTS,
    get_prompt_builder,
    get_context_interpreter,
)

# Smart Router
from .smart_router import (
    SmartRouter,
    RouterResult,
    RouterStats,
    RoutingAction,
    TextPreprocessor,
    ModifierDetector,
    ConfidenceRouter,
    process_batch,
    evaluate_router,
    get_smart_router,
)

# Metrics
from .metrics import compute_classification_metrics, compute_intake_metrics, compute_economics

# Visualization
from .viz import (
    plot_confusion_matrix, plot_f1_by_class, plot_completion_distribution, 
    plot_economics_comparison, plot_turns_vs_completion
)

# Schemas
from .schemas import INTAKE_SCHEMA, COMPLAINT_TYPES, get_required_fields, validate_intake, export_schema

# LLM client
from .llm import TogetherLLM

# Classifiers
from .classifiers import (
    EmbeddingClassifier, SetFitClassifier, RuleBasedClassifier, 
    CascadeClassifier, create_tfidf_pipeline, get_tfidf_param_grid, run_gridsearch
)

# Production utilities
from .production import (
    ProductionClassifier, export_model_for_production, validate_production_model
)


__all__ = [
    # =========================================================================
    # Data (v1 - backward compatibility)
    # =========================================================================
    "generate_d1_dataset",
    "generate_d2_cases", 
    "load_or_generate_d1",
    "load_or_generate_d2",
    "augment_dataset",
    "generate_d1_with_augmentation",
    "INTENT_LABELS",
    "INTENT_TO_SCENARIO",
    
    # =========================================================================
    # Data v2 (L1/L2 support)
    # =========================================================================
    "generate_d1_dataset_v2",
    "generate_stratified_split",
    "save_dataset",
    "load_dataset_v2",
    "get_dataset_stats",
    "MESSAGE_TEMPLATES_L2",
    "VARIATION_TEMPLATES",
    
    # =========================================================================
    # Taxonomy
    # =========================================================================
    "INTENT_LABELS_L1",
    "INTENT_LABELS_L2",
    "L2_TO_L1",
    "L1_TO_PRODUCTION_NODE",
    "L2_TO_SCENARIO_NEW",
    "CLASS_WEIGHTS_L1",
    "CLASS_WEIGHTS_L2",
    "CONFIDENCE_THRESHOLDS",
    "ClassificationResult",
    "TaxonomyConfig",
    "get_l1_from_l2",
    "get_production_node",
    "get_scenario",
    "get_l2_classes_for_l1",
    "load_taxonomy",
    "get_sklearn_class_weights",
    
    # =========================================================================
    # Entity Extraction
    # =========================================================================
    "EntityExtractor",
    "Entity",
    "ExtractionResult",
    "extract_entities",
    "has_doctor_mention",
    "has_date_mention",
    "get_mentioned_procedure",
    "get_entity_extractor",
    
    # =========================================================================
    # Augmentation
    # =========================================================================
    "TextAugmenter",
    "AugmentationConfig",
    "AugmentedSample",
    "augment_dataset_advanced",
    "create_augmented_variants",
    "generate_typo_variants",
    "generate_abbreviation_variants",
    "generate_case_variants",
    "generate_punctuation_variants",
    "get_augmenter",
    
    # =========================================================================
    # LLM Generation
    # =========================================================================
    "LLMDataGenerator",
    "GeneratedSample",
    "QualityChecker",
    "generate_llm_samples",
    
    # =========================================================================
    # Context-Aware Prompts
    # =========================================================================
    "DialogContext",
    "EnrichedContext",
    "PromptBuilder",
    "ContextInterpreter",
    "CONTEXT_PROMPTS",
    "get_prompt_builder",
    "get_context_interpreter",
    
    # =========================================================================
    # Smart Router
    # =========================================================================
    "SmartRouter",
    "RouterResult",
    "RouterStats",
    "RoutingAction",
    "TextPreprocessor",
    "ModifierDetector",
    "ConfidenceRouter",
    "process_batch",
    "evaluate_router",
    "get_smart_router",
    
    # =========================================================================
    # Metrics
    # =========================================================================
    "compute_classification_metrics",
    "compute_intake_metrics",
    "compute_economics",
    
    # =========================================================================
    # Visualization
    # =========================================================================
    "plot_confusion_matrix",
    "plot_f1_by_class",
    "plot_completion_distribution",
    "plot_economics_comparison",
    "plot_turns_vs_completion",
    
    # =========================================================================
    # Schemas
    # =========================================================================
    "INTAKE_SCHEMA",
    "COMPLAINT_TYPES",
    "get_required_fields",
    "validate_intake",
    "export_schema",
    
    # =========================================================================
    # LLM Client
    # =========================================================================
    "TogetherLLM",
    
    # =========================================================================
    # Classifiers
    # =========================================================================
    "EmbeddingClassifier",
    "SetFitClassifier",
    "RuleBasedClassifier",
    "CascadeClassifier",
    "create_tfidf_pipeline",
    "get_tfidf_param_grid",
    "run_gridsearch",
    
    # =========================================================================
    # Production
    # =========================================================================
    "ProductionClassifier",
    "export_model_for_production",
    "validate_production_model",
]
