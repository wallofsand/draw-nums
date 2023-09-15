import numpy as np
import struct

def dataset():
    # MNIST digit set:
    FILE_DIGIT_TRAIN_IMG = 'MNIST/train-images-idx3-ubyte'
    FILE_DIGIT_TRAIN_LBL = 'MNIST/train-labels-idx1-ubyte'
    FILE_DIGIT_TEST_IMG = 'MNIST/t10k-images-idx3-ubyte'
    FILE_DIGIT_TEST_LBL = 'MNIST/t10k-labels-idx1-ubyte'
    FILE_FASHION_TRAIN_IMG = 'MNIST-FASHION/train-images-idx3-ubyte'
    FILE_FASHION_TRAIN_LBL = 'MNIST-FASHION/train-labels-idx1-ubyte'
    FILE_FASHION_TEST_IMG = 'MNIST-FASHION/t10k-images-idx3-ubyte'
    FILE_FASHION_TEST_LBL = 'MNIST-FASHION/t10k-labels-idx1-ubyte'
    FILE_TRAIN_IMG = FILE_DIGIT_TRAIN_IMG
    FILE_TRAIN_LBL = FILE_DIGIT_TRAIN_LBL
    FILE_TEST_IMG = FILE_DIGIT_TEST_IMG
    FILE_TEST_LBL = FILE_DIGIT_TEST_LBL
    # open the training file
    with open(FILE_TRAIN_IMG, 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        train_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_data = np.reshape(train_data, (size, nrows, ncols))
    # open the label file for validation
    with open(FILE_TRAIN_LBL,'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        train_labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    # open the testing file
    with open(FILE_TEST_IMG, 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        test_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_data = np.reshape(test_data, (size, nrows, ncols))
    # open the testing label file for validation
    with open(FILE_TEST_LBL,'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        test_labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    return train_data, train_labels, test_data, test_labels

class Layer:
    def forward_prop(self, image):
        pass
    def backward_prop(self, dE_dY, alpha=0.05):
        pass
    def save(self):
        pass

class ConvolutionalLayer(Layer):
    def __init__(self, kernel_num, kernel_size, kernel_depth):
        rng = np.random.default_rng()
        self.kernel_num = kernel_num
        self.kernel_size = kernel_size
        self.kernel_depth = kernel_depth
        self.kernels = rng.standard_normal((kernel_num, kernel_depth, kernel_size, kernel_size))/(kernel_size**2)
    def save(self):
        data = np.array(['cvl',
                self.kernel_num,
                self.kernel_size,
                self.kernel_depth,
                self.kernels], dtype=object)
        return data
    def patches_generator(self, image):
        """
        Divide the input image in patches to be used during convolution.
        Yields the tuples containing the patches and their coordinates.
        """
        # The number of patches, given an fxf filter is h-f+1 for height and w-f+1 for width
        # For our 3x3 filter and our 28x28 image there are
        # 28 - 3 + 1 = 26 for height and width so 26 * 26 = 676 patches
        for h in range(image.shape[0] - self.kernel_size + 1):
            for w in range(image.shape[1] - self.kernel_size + 1):
                patch = image[h:(h + self.kernel_size), w:(w + self.kernel_size)]
                yield patch, h, w
    def forward_prop(self, image):
        """
        Perform forward propagation for the convolutional layer.
        """
        self.CL_in = image # stored for backprop
        # Initialize the convolution output volume of the correct size
        cl_output = np.zeros((image.shape[0]-self.kernel_size+1, image.shape[1]-self.kernel_size+1, self.kernel_num))
        # Unpack the generator
        for patch, h, w in self.patches_generator(self.CL_in):
            # Perform convolution for each patch
            for channel in range(image.shape[2]):
                img = patch[:,:,channel]
                cl_output[h,w] += np.sum(img*self.kernels[:,channel,:,:], axis=(1,2))
        return cl_output
    def backward_prop(self, dE_dY, alpha=0.05):
        """
        Takes the gradient of the loss function with respect to the output and
        computes the gradients of the loss function with respect to the kernels' weights.
        dE_dY comes from the following layer, typically max pooling layer.
        It updates the kernels' weights
        """
        # Initialize gradient of the loss function with respect to the kernel weights
        dE_dk = np.zeros(self.kernels.shape)
        for patch, h, w in self.patches_generator(self.CL_in):
            for f in range(self.kernel_num):
                for channel in range(patch.shape[2]):
                    dE_dk[f,channel] += patch[:,:,channel] * dE_dY[h, w, f]
        # Update the parameters
        self.kernels -= alpha*dE_dk
        # print('CvL dE_dk', dE_dk.T.shape)
        return dE_dk.T

class MaxPoolingLayer(Layer):
    def __init__(self, kernel_size):
        """
        Constructor takes as input kernel size (n of an nxn kernel)
        """
        self.n = kernel_size
    def save(self):
        data = np.array(['mpl', self.n], dtype=object)
        return data
    def patches_generator(self, image):
        """
        Divide the input image in patches to be used during convolution.
        Yields the tuples containing the patches and their coordinates.
        """
        for h in range(image.shape[0]//self.n):
            for w in range(image.shape[1]//self.n):
                patch = image[h*self.n:(h+1)*self.n, w*self.n:(w+1)*self.n]
                yield patch, h, w
    def forward_prop(self, image):
        # Store for backprop
        self.MPL_in = image
        image_h, image_w, num_k = self.MPL_in.shape
        output = np.zeros((image_h//self.n, image_w//self.n, num_k))
        for patch, h, w in self.patches_generator(self.MPL_in):
            output[h, w] = np.amax(patch, axis=(0, 1))
        return output
    def backward_prop(self, dE_dY, alpha=0.05):
        """
        Takes the gradient of the loss function with respect to the output and
        computes the gradients of the loss function with respect to the kernels' weights.
        dE_dY comes from the following layer, typically softmax.
        There are no weights to update; output is used to update the weights of the previous layer.
        """
        dE_dk = np.zeros(self.MPL_in.shape)
        # print('MPL_in', self.MPL_in.shape)
        for patch, h, w in self.patches_generator(self.MPL_in):
            image_h, image_w, num_k = patch.shape
            max_val = np.amax(patch, axis=(0,1))
            # print('patch', patch.shape)
            # print('dE_dk', dE_dk.shape)
            # print('dE_dY', dE_dY.shape)
            for idx_h in range(image_h):
                for idx_w in range(image_w):
                    for idx_k in range(num_k):
                        if patch[idx_h, idx_w, idx_k] == max_val[idx_k]:
                            dE_dk[h*self.n+idx_h, w*self.n+idx_w, idx_k] = np.sum(dE_dY[h, w, idx_k])
            return dE_dk

class SoftmaxLayer(Layer):
    def __init__(self, input_units, output_units):
        """
        Initiallize weights and biases.
        """
        rng = np.random.default_rng()
        self.w = rng.standard_normal((input_units, output_units))/input_units
        self.b = np.zeros(output_units)
    def save(self):
        data = np.array(['sml',
                self.w,
                self.b], dtype=object)
        return data
    def forward_prop(self, image):
        # Store for backprop
        self.input_shape = image.shape
        self.input_flat = image.flatten()
        # W * x + b
        self.output = np.dot(self.input_flat, self.w) + self.b
        softmax_activation = np.exp(self.output) / np.sum(np.exp(self.output), axis=0)
        return softmax_activation
    def backward_prop(self, dE_dY, alpha=0.05):
        # loss (dE_dY) will be 0 for most cases but...
        for i, gradient in enumerate(dE_dY):
            # ... will be some float when i is the label index
            if gradient == 0:
                continue
            # Compute exponential of output ie. partial softmax activation
            trans_eq = np.exp(self.output)
            S_total = np.sum(trans_eq)
            # Compute gradients with respect to output (Z)
            dY_dZ = -trans_eq[i]*trans_eq / (S_total**2)
            dY_dZ[i] = trans_eq[i]*(S_total - trans_eq[i]) / (S_total**2)
            # Compute gradients of Z with respect to weights, bias, input
            dZ_dw = self.input_flat
            dZ_db = 1
            dZ_dX = self.w
            # Gradient of loss with respect to output...
            dE_dZ = gradient * dY_dZ
            dE_dw = dZ_dw[np.newaxis].T @ dE_dZ[np.newaxis]
            dE_db = dE_dZ * dZ_db
            dE_dX = dZ_dX @ dE_dZ
            # Update parameters
            self.w -= dE_dw * alpha
            self.b -= dE_db * alpha
            return dE_dX.reshape(self.input_shape)

class ReLuLayer(Layer):
    """
    Take an arbitrary sized input and return the same size output.
    """
    def save(self):
        data = np.array(['relu'], dtype=object)
        return data
    def forward_prop(self, input):
        zeros = np.zeros(input.shape)
        return np.maximum(input, zeros)
    def backward_prop(self, dE_dY, alpha=0.05):
        # 1 if y > 0 else 0
        return dE_dY > 0

class Network:
    def __init__(self, layers:list):
        self.layers = layers
    def load(self, filename):
        data = np.load(filename, allow_pickle=True)
        lst = data.files
        for item in lst:
            l = data[item]
            newlayer = Layer()
            if l[0] == 'cvl':
                newlayer = ConvolutionalLayer(
                    kernel_num=l[1],
                    kernel_size=l[2],
                    kernel_depth=l[3]
                )
                newlayer.kernels = l[4]
                print('loaded convolutional layer')
            elif l[0] == 'mpl':
                newlayer = MaxPoolingLayer(
                    kernel_size=l[1]
                )
                print('loaded max pooling layer')
            elif l[0] == 'sml':
                newlayer = SoftmaxLayer(0,0)
                newlayer.w = l[1]
                newlayer.b = l[2]
                print('loaded softmax layer')
            elif l[0] == 'relu':
                newlayer = ReLuLayer()
                print('loaded ReLu layer')
            self.layers.append(newlayer)
    def save(self, filename):
        data = np.empty(len(self.layers), dtype=np.ndarray)
        for i, layer in enumerate(self.layers):
            data[i] = layer.save()
        np.savez(filename, *data)
    def forward_pass(self, image, label=None):
        """
        Pass an image through the model without training.
        """
        output = image[:,:,np.newaxis]/255
        for layer in self.layers:
            output = layer.forward_prop(output)
        if label == None:
            return output
        # Compute loss and accuracy
        loss = -np.log(output[label])
        accuracy = 1 if np.argmax(output) == label else 0
        return output, loss, accuracy
    def backprop(self, gradient):
        gradient_back = gradient
        for layer in reversed(self.layers):
            gradient_back = layer.backward_prop(gradient_back)
        return gradient_back
    def train(self, image, label):
        output, loss, accuracy = self.forward_pass(image, label)
        gradient = np.zeros(10)
        gradient[label] = -1/output[label]
        gradient_back = self.backprop(gradient)
        return loss, accuracy
    def test(self, testX, testY):
        """
        Test the model on a set of data.
        Doesn't train the model.
        """
        loss = 0
        accuracy = 0
        permutation = np.random.permutation(1000)
        X = testX[permutation]
        Y = testY[permutation]
        print("Begin test . . .")
        for i, (image, label) in enumerate(zip(X, Y)):
            output, loss_1, accuracy_1 = self.forward_pass(image, label)
            loss += loss_1
            accuracy += accuracy_1
        lossval = loss/len(Y)
        accval = 100*accuracy/len(Y)
        print('Test average loss {:.3f}, accuracy {}%'.format(lossval, accval))
    def CNN_training(self, trainX, trainY):
        for epoch in range(4):
            print('Epoch {} ->'.format(epoch+1))
            # Shuffle training data
            permutation = np.random.permutation(500)
            X = trainX[permutation]
            Y = trainY[permutation]
            loss = 0
            accuracy = 0
            for i, (image, label) in enumerate(zip(X, Y)):
                # print a snapshot of the training
                if i % 100 == 99:
                    print("Step {}. For the last 100 steps: average loss {:.3f}, accuracy {}".format(i+1, loss/100, accuracy))
                    loss = 0
                    accuracy = 0
                loss_1, accuracy_1 = self.train(image, label)
                loss += loss_1
                accuracy += accuracy_1
