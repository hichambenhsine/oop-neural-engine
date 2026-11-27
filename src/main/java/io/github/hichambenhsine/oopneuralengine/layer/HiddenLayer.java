package io.github.hichambenhsine.oopneuralengine.layer;

import io.github.hichambenhsine.oopneuralengine.model.Neuron;

import java.util.List;

/**
 * Intermediate hidden layer performing weighted feature transformations.
 */
public class HiddenLayer extends AbstractLayer {

    public HiddenLayer(List<Neuron> neurons) {
        super(LayerType.HIDDEN, neurons);
    }

    @Override
    public double[] forward(double[] inputs) {
        return forwardConnected(inputs);
    }
}