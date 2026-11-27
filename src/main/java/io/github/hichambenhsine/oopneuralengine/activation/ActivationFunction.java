package io.github.hichambenhsine.oopneuralengine.activation;

/**
 * Strategy interface encapsulating an activation function used by neurons.
 */
public interface ActivationFunction {

    /**
     * Computes the activation output for a single neuron's weighted sum.
     *
     * @param input the weighted sum (including bias)
     * @return the activated scalar value
     */
    double activate(double input);

    /**
     * Applies the activation function across an entire layer's output vector.
     * Defaults to element-wise activation.
     *
     * @param inputs the raw weighted sums for all neurons in a layer
     * @return the activated output vector
     */
    default double[] activateLayer(double[] inputs) {
        if (inputs == null) {
            throw new IllegalArgumentException("Input vector must not be null");
        }
        double[] outputs = new double[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            outputs[i] = activate(inputs[i]);
        }
        return outputs;
    }
}
