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

def transposed_conv2d(image, kernel, stride=1, padding=0):
    """
    Perform 2D transposed convolution on an image using a kernel.

    Arguments:
    image -- 2D numpy array, representing the input image.
    kernel -- 2D numpy array, represnting the transposed convolution kernel.
    stride -- integer, stride of the transposed convolution.
    padding -- integer, amount of padding to add around the image.

    Returns:
    result -- 2D numpy array, result of the transposed convolution.
    """
    # Get dimension of the image and the kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate the output dimension
    output_height = (image_height - 1) * stride + kernel_height -2*padding
    output_width = (image_width - 1) * stride + kernel_width - 2 *padding

    # Initialize the result
    result = np.zeros(output_height, output_width)

    # Add padding to the result
    padded_result = np.pad(result, pad_width=padding, mode='constant', constant_values=0)

    # Perform the transposed convolution
    for y in range(image_height):
        for x in range(image_width):
            padded_result[y*stride:y*stride+kernel_height, x*stride:x*stride+kernel_width] += image[y, x] * kernel

        
    # Remove padding
    if padding > 0:
        result = padded_result[padding:-padding, padding:-padding]
    else:
        result = padded_result

    return result

def conv3d(volume, kernel, padding=0):
    """
    Perform 3D convolution on a volumr using a kernel.

    Arguments:
    volume -- 3D numpy array, representing the input volume.
    kernel -- 3D numpy array, representing the convolution kernel.
    padding -- integer, amount of padding to be added.

    Returns:
    result -- 3D numpy array, result of the 3D convolution
    """
    volume_depth, volume_height, volume_width = volume.shape
    kernel_depth, kernel_height, kernel_width = kernel.shape

    # Calculate the output dimensions
    output_depth = volume_depth - kernel_depth + 1 + 2*padding
    output_height = volume_height - kernel_height + 1 +2*padding
    output_width = volume_width - kernel_width + 1 + 2*padding

    # Initialize the result
    result = np.zeros(output_depth, output_height, output_width)

    # Add padding to the volume
    padded_volume = np.pad(volume, pad_width=padding, mode='constant', constant_values=0)

    # Perform the 3D convolution
    for d in range(output_depth):
        for h in range(output_height):
            for w in range(output_width):
                result[d, h, w] = np.sum(padded_volume[d:d+kernel_depth, h:h+kernel_height, w:w+kernel_width] * kernel)

    return result
 
def transposed_conv3d(volume, kernel, stride=1, padding=0):
    """
    Perform 3D transposed convolution on a volume using a kernel.

    Arguments:
    volume -- 3D numpy array, representing the input volume.
    kernel -- 3D numpy array, representing the transposed convolution kernel.
    stride -- integer, stride of the transposed convolution
    padding -- integer, amount of padding to add

    Returns:
    result -- 3D numpy array, result of the 3D transposed convolution
    """
    volume_depth, volume_height, volume_width = volume.shape
    kernel_depth, kernel_height, kernel_width = kernel.shape

    # Calculate the output dimensions
    output_depth = (volume_depth - 1) * stride + kernel_depth - 2 * padding
    output_height = (volume_height - 1) * stride + kernel_height - 2 * padding
    output_width = (volume_width - 1) * stride + kernel_width - 2 * padding

    # Initialize the result
    result = np.zeros(output_depth, output_height, output_width)

    # Add padding to the result
    padded_result = np.pad(result, pad_width=padding, mode='constant', constant_values=0)

    # Perform the 3D transposed convolution
    for d in range(volume_depth):
        for h in range(volume_height):
            for w in range(volume_width):
                padded_result[d*stride:d*stride+kernel_depth,
                               h*stride:h*stride+kernel_height,
                               w*stride:w*stride+kernel_width] += volume[d, h, w] * kernel
                
    # Remove padding
    if padding > 0:
        result = padded_result[padding:-padding, padding:-padding, padding:-padding]
    else:
        result = padded_result

    return result

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def step(x):
    return np.where(x >= 0, 1, 0)

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def arctan(x):
    return np.arctan(x)

def relu(x):
    return np.where(x > 0, x, 0)

def gaussian(x):
    return np.exp(-(x**2))

def softplus(x): 
    return np.log2(1 + np.exp(x))