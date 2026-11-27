package io.github.hichambenhsine.oopneuralengine.layer;

import io.github.hichambenhsine.oopneuralengine.activation.ActivationFunction;
import io.github.hichambenhsine.oopneuralengine.model.Neuron;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Factory Method class responsible for instantiating typed {@link Layer} instances
 * populated with configured neurons.
 */
public class LayerFactory {

    public Layer createLayer(LayerType type, int neuronCount, ActivationFunction activationFunction) {
        Objects.requireNonNull(type, "Layer type must not be null");
        Objects.requireNonNull(activationFunction, "Activation function must not be null");
        if (neuronCount <= 0) {
            throw new IllegalArgumentException("Neuron count must be positive, got: " + neuronCount);
        }

        List<Neuron> neurons = new ArrayList<>(neuronCount);
        for (int i = 1; i <= neuronCount; i++) {
            neurons.add(new Neuron(type.name() + "-" + i, activationFunction));
        }

        return switch (type) {
            case INPUT -> new InputLayer(neurons);
            case HIDDEN -> new HiddenLayer(neurons);
            case OUTPUT -> new OutputLayer(neurons);
        };
    }
}