package io.github.hichambenhsine.oopneuralengine.activation;

/**
 * Legacy activation interface with a contract incompatible with {@link ActivationFunction}.
 */
public interface LegacyActivationFunction {

    /**
     * Evaluates the legacy activation function.
     *
     * @param value the raw input value
     * @return the computed output
     */
    double compute(double value);
}