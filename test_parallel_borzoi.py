#!/usr/bin/env python3
"""
Test script for the parallel MultiBorzoiGeneRegions implementation.
"""

import time

from bolero.tl.model.borzoi.utils import MultiBorzoiGeneRegions


def test_parallel_vs_sequential():
    """Test that parallel and sequential processing give the same results."""
    # Create a simple test case with multiple datasets
    key_to_genome = {"dataset1": "hg38", "dataset2": "hg38", "dataset3": "hg38"}

    # Initialize the multi-regions handler
    multi_regions = MultiBorzoiGeneRegions(key_to_genome)

    # Test parameters
    split_id = 0
    deg_list_dict = None

    print("Testing parallel vs sequential processing...")

    # Test sequential processing
    print("Running sequential processing...")
    start_time = time.time()
    train_seq, valid_seq, test_seq = multi_regions.get_train_valid_test_regions(
        split_id=split_id, deg_list_dict=deg_list_dict, use_parallel=False
    )
    sequential_time = time.time() - start_time
    print(f"Sequential processing took {sequential_time:.2f} seconds")

    # Test parallel processing
    print("Running parallel processing...")
    start_time = time.time()
    train_par, valid_par, test_par = multi_regions.get_train_valid_test_regions(
        split_id=split_id, deg_list_dict=deg_list_dict, use_parallel=True
    )
    parallel_time = time.time() - start_time
    print(f"Parallel processing took {parallel_time:.2f} seconds")

    # Compare results
    print("\nComparing results...")

    # Check if results are identical
    train_equal = train_seq.equals(train_par)
    valid_equal = valid_seq.equals(valid_seq)
    test_equal = test_seq.equals(test_par)

    print(f"Train regions identical: {train_equal}")
    print(f"Valid regions identical: {valid_equal}")
    print(f"Test regions identical: {test_equal}")

    if train_equal and valid_equal and test_equal:
        print(
            "✅ SUCCESS: Parallel and sequential processing produce identical results!"
        )
        speedup = sequential_time / parallel_time if parallel_time > 0 else float("inf")
        print(f"Speedup: {speedup:.2f}x")
    else:
        print("❌ ERROR: Results differ between parallel and sequential processing!")

    # Print some basic statistics
    print("\nResults summary:")
    print(f"Train regions: {len(train_seq)} rows")
    print(f"Valid regions: {len(valid_seq)} rows")
    print(f"Test regions: {len(test_seq)} rows")

    # Clean up Ray resources
    multi_regions.shutdown_ray()
    print("\nRay resources cleaned up.")


if __name__ == "__main__":
    test_parallel_vs_sequential()
