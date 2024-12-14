import numpy as np


def sample_dna_one_hot(pwm_df, num_sequences):
    """Sample one-hot encoding from PWM."""
    # Convert PWM to cumulative probabilities
    cdf = np.cumsum(pwm_df.values, axis=1)  # Convert to CDF

    # Generate random numbers for each position in each sequence
    random_vals = np.random.rand(
        num_sequences, cdf.shape[0]
    )  # Shape (num_sequences, num_positions)

    # Compare random values with CDF to find the one-hot index
    sampled_indices = (random_vals[:, :, None] < cdf[None, :, :]).argmax(
        axis=2
    )  # Shape (num_sequences, num_positions)

    # Convert indices to one-hot encoding
    one_hot = np.zeros(
        (num_sequences, cdf.shape[0], cdf.shape[1]), dtype=bool
    )  # Shape (num_sequences, num_positions, 4)
    np.put_along_axis(one_hot, sampled_indices[:, :, None], True, axis=2)
    return one_hot


def one_hot_to_sequence(one_hot, bases):
    """Convert one-hot encoding to DNA sequence."""
    indices = np.argmax(one_hot, axis=2)  # Find the index of the 1 in each position
    return ["".join(bases[i] for i in row) for row in indices]
