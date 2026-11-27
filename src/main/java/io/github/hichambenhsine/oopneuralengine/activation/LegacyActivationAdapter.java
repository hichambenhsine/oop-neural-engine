package io.github.hichambenhsine.oopneuralengine.activation;

import java.util.Objects;

/**
 * Object Adapter allowing any {@link LegacyActivationFunction} to be used
 * wherever an {@link ActivationFunction} is expected.
 */
public class LegacyActivationAdapter implements ActivationFunction {

    private final LegacyActivationFunction legacyFunction;

    public LegacyActivationAdapter(LegacyActivationFunction legacyFunction) {
        this.legacyFunction = Objects.requireNonNull(legacyFunction, "Legacy activation function must not be null");
    }

    @Override
    public double activate(double input) {
        return legacyFunction.compute(input);
    }
}