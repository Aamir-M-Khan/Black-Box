import numpy as np

def conv1d(signal, kernel, padding=0):
    """
    Perform 1D convolution on a signal using a kernel.

    Arguements:
    signal -- 1D numpy array, representing the input signal.
    kernel -- 1D numpy array, representing the concolutional kernel.
    padding -- integer, amount of padding to add.

    Returns:
    result -- 1D numpy array, result of convolution
    """
    signal_length = len(signal)
    kernel_length = len(kernel)

    # Calculate the output length
    output_length = signal_length - kernel_length + 1 + 2 * padding

    # Initialize the result
    result = np.zeros(output_length)

    # Add padding to the signal
    padded_signal = np.pad(signal, pad_width=padding, mode='constant', constant_values=0)

    # Perform the convolution
    for i in range(output_length):
        result[i] = np.sum(padded_signal[i:i+kernel_length] * kernel)

    return result

