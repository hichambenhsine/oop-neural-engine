package io.github.hichambenhsine.oopneuralengine.activation;

/**
 * Legacy hyperbolic tangent implementation following {@link LegacyActivationFunction}.
 */
public class LegacyTanhActivation implements LegacyActivationFunction {

    @Override
    public double compute(double value) {
        return Math.tanh(value);
    }
}