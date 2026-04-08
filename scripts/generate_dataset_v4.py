#!/usr/bin/env python3
"""
Script to generate D1 dataset v4 with 10,000 samples.

Two-level classification:
- L1 (5 classes): anamnesis, booking, faq, negative_feedback, conversational
- L2 (20 classes): subtypes for each L1

Usage:
    python scripts/generate_dataset_v4.py --n 10000 --output data/
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_v2 import (
    generate_d1_dataset_v2,
    save_dataset,
    get_dataset_stats,
)


def main():
    parser = argparse.ArgumentParser(description="Generate D1 dataset v4")
    parser.add_argument("--n", type=int, default=10000, help="Number of samples")
    parser.add_argument("--output", type=str, default="data", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--name", type=str, default="d1_messages_v4", help="Dataset name")
    parser.add_argument("--no-splits", action="store_true", help="Don't create train/val/test splits")
    
    args = parser.parse_args()
    
    print(f"Generating D1 dataset v4 with {args.n} samples...")
    print(f"Output directory: {args.output}")
    print(f"Random seed: {args.seed}")
    print()
    
    # Generate dataset
    df = generate_d1_dataset_v2(n=args.n, seed=args.seed)
    
    # Print statistics
    stats = get_dataset_stats(df)
    print("Dataset Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Unique texts: {stats['unique_texts']}")
    print(f"  L1 classes: {stats['l1_classes']}")
    print(f"  L2 classes: {stats['l2_classes']}")
    print()
    
    print("L1 Distribution:")
    for l1, count in sorted(stats['l1_distribution'].items()):
        print(f"  {l1}: {count} ({count/stats['total_samples']*100:.1f}%)")
    print()
    
    print("L2 Distribution:")
    for l2, count in sorted(stats['l2_distribution'].items()):
        print(f"  {l2}: {count}")
    print()
    
    print("Source Distribution:")
    for source, count in sorted(stats['source_distribution'].items()):
        print(f"  {source}: {count}")
    print()
    
    # Save dataset
    output_dir = Path(args.output)
    paths = save_dataset(
        df=df,
        output_dir=output_dir,
        name=args.name,
        create_splits=not args.no_splits,
    )
    
    print("Saved files:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    
    print()
    print("Done!")


if __name__ == "__main__":
    main()
