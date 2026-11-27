package io.github.hichambenhsine.oopneuralengine.activation;

/**
 * Sigmoid activation strategy: f(x) = 1 / (1 + e^(-x)).
 */
public class SigmoidActivation implements ActivationFunction {

    @Override
    public double activate(double input) {
        return 1.0 / (1.0 + Math.exp(-input));
    }
}