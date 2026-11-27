package io.github.hichambenhsine.oopneuralengine.activation;


/**
 * SoftMax activation strategy.
 * Provides numerically stable probability distribution normalization across a layer vector,
 * and falls back to exponential scaling when evaluated on a single scalar.
 */
public class SoftMaxActivation implements ActivationFunction {

    @Override
    public double activate(double input) {
        return Math.exp(input);
    }

    @Override
    public double[] activateLayer(double[] inputs) {
        if (inputs == null) {
            throw new IllegalArgumentException("Input vector must not be null");
        }
        if (inputs.length == 0) {
            return new double[0];
        }

        double max = inputs[0];
        for (int i = 1; i < inputs.length; i++) {
            if (inputs[i] > max) {
                max = inputs[i];
            }
        }

        double[] expValues = new double[inputs.length];
        double sum = 0.0;
        for (int i = 0; i < inputs.length; i++) {
            expValues[i] = Math.exp(inputs[i] - max);
            sum += expValues[i];
        }

        for (int i = 0; i < expValues.length; i++) {
            expValues[i] /= sum;
        }
        return expValues;
    }
}