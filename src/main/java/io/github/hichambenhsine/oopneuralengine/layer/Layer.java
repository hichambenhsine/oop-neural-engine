package io.github.hichambenhsine.oopneuralengine.layer;

import io.github.hichambenhsine.oopneuralengine.model.Neuron;

import java.util.List;

/**
 * Component interface representing a neural network layer.
 * Serves as both the Factory product and the Decorator component contract.
 */
public interface Layer {

    LayerType getType();

    List<Neuron> getNeurons();

    /**
     * Propagates an input vector through this layer and returns the activated output vector.
     *
     * @param inputs the signal vector entering this layer
     * @return the activated output vector produced by this layer
     */
    double[] forward(double[] inputs);

    default int size() {
        return getNeurons().size();
    }
}