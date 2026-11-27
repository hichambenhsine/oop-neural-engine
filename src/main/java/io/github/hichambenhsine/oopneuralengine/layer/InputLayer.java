package io.github.hichambenhsine.oopneuralengine.layer;

import io.github.hichambenhsine.oopneuralengine.model.Neuron;

import java.util.List;

/**
 * Entry layer receiving external input features.
 */
public class InputLayer extends AbstractLayer {

    public InputLayer(List<Neuron> neurons) {
        super(LayerType.INPUT, neurons);
    }

    @Override
    public double[] forward(double[] inputs) {
        List<Neuron> neurons = getNeurons();
        if (inputs == null || inputs.length != neurons.size()) {
            throw new IllegalArgumentException(
                    "Input vector length must match input layer size (" + neurons.size() + ")"
            );
        }
        double[] outputs = new double[neurons.size()];
        for (int i = 0; i < neurons.size(); i++) {
            outputs[i] = neurons.get(i).activateExternalInput(inputs[i]);
        }
        return outputs;
    }
}