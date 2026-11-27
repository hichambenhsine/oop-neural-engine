package io.github.hichambenhsine.oopneuralengine.layer;

import io.github.hichambenhsine.oopneuralengine.model.Neuron;

import java.util.List;

/**
 * Final layer producing the neural network's prediction vector.
 */
public class OutputLayer extends AbstractLayer {

    public OutputLayer(List<Neuron> neurons) {
        super(LayerType.OUTPUT, neurons);
    }

    @Override
    public double[] forward(double[] inputs) {
        return forwardConnected(inputs);
    }
}