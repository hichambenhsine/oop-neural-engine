package io.github.hichambenhsine.oopneuralengine.activation;

/**
 * Identity (linear) activation strategy: f(x) = x.
 */
public class IdentityActivation implements ActivationFunction {

    @Override
    public double activate(double input) {
        return input;
    }
}