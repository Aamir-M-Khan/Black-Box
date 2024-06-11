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

def transposed_conv1d(signal, kernel, stride=1, padding=0):
    """
    Perform 1D transposed convolution on a signal using a kernel.

    Arguments:
    signal -- 1D numpy array, representing the input signal.
    kernel -- 1D numpy array, representing the transposed convolution kernel.
    stride -- integer, stride of the transposed convolution.
    padding -- integer, amount of padding to add.

    Returns:
    result -- 1D numpy array, result of the transposed convolution.
    """
    signal_length = len(signal)
    kernel_length = len(kernel)

    # Calculate the output length 
    output_length = (signal_length - 1) * stride + kernel_length - 2 * padding

    # Initialize the result
    result = np.zeros(output_length)

    # Add padding to the result
    padded_result = np.pad(result, pad_width=padding, mode='constant', constant_values=0)

    # Perform the transposed convolution
    for i in range(signal_length):
        padded_result[i*stride: i*stride + kernel_length] += signal[i] * kernel

    # Remove padding
    if padding > 0:
        result = padded_result[padding:-padding]
    else:
        result = padded_result

    return result

def conv2d(image, kernel, padding=0):
    """
    Perform 2D convolution on an image using a kernel with padding,

    Arguments:
    image -- 2D numpy array, representing the input image
    kernel -- 2D numpy array, representing the convolutional kernel.
    padding -- integer, amount of padding to add around the image.

    Returns:
    result -- 2D numpy array, result of the convolution.
    
    """
    # Get the dimension of the image and the kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate the output dimensions after convolution with padding
    output_height = image_height - kernel_height + 1 + 2 * padding
    output_width = image_width - kernel_width + 1 + 2*padding

    # Initialize the result after convolution
    result_conv = np.zeros(output_height, output_width)

    # Add padding to the image
    padded_image = np.pad(image, pad_width=padding, mode='constant', constant_values=0)

    # Perform the convolution
    for y in range(output_height):
        for x in range(output_width):
            result_conv[y, x] = np.sum(padded_image[y:y+kernel_height, x:x+kernel_width]*kernel)

    return result_conv

