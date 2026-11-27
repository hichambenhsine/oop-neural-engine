package io.github.hichambenhsine.oopneuralengine.layer;

import io.github.hichambenhsine.oopneuralengine.activation.ActivationFunction;
import io.github.hichambenhsine.oopneuralengine.activation.SoftMaxActivation;
import io.github.hichambenhsine.oopneuralengine.model.Neuron;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Base implementation shared by concrete layer types.
 */
public abstract class AbstractLayer implements Layer {

    private final LayerType type;
    private final List<Neuron> neurons;

    protected AbstractLayer(LayerType type, List<Neuron> neurons) {
        this.type = Objects.requireNonNull(type, "Layer type must not be null");
        Objects.requireNonNull(neurons, "Neurons list must not be null");
        if (neurons.isEmpty()) {
            throw new IllegalArgumentException("Layer must contain at least one neuron");
        }
        this.neurons = new ArrayList<>(neurons);
    }

    @Override
    public LayerType getType() {
        return type;
    }

    @Override
    public List<Neuron> getNeurons() {
        return Collections.unmodifiableList(neurons);
    }

    /**
     * Shared forward propagation logic for connected layers (Hidden / Output).
     * Uses the explicit input vector when aligned with incoming connections so upstream
     * layer decorators transparently affect downstream computations, while falling back
     * to source neuron outputs when no vector is provided.
     */
    protected double[] forwardConnected(double[] inputs) {
        double[] weightedSums = new double[neurons.size()];
        for (int i = 0; i < neurons.size(); i++) {
            Neuron neuron = neurons.get(i);
            if (inputs != null && neuron.getIncomingConnections().size() == inputs.length) {
                weightedSums[i] = neuron.computeWeightedSum(inputs);
            } else {
                weightedSums[i] = neuron.computeWeightedSum();
            }
        }

        double[] outputs = new double[neurons.size()];
        ActivationFunction firstActivation = neurons.get(0).getActivationFunction();
        if (firstActivation instanceof SoftMaxActivation) {
            outputs = firstActivation.activateLayer(weightedSums);
            for (int i = 0; i < neurons.size(); i++) {
                neurons.get(i).setOutput(outputs[i]);
            }
        } else {
            for (int i = 0; i < neurons.size(); i++) {
                Neuron neuron = neurons.get(i);
                double activated = neuron.getActivationFunction().activate(weightedSums[i]);
                neuron.setOutput(activated);
                outputs[i] = activated;
            }
        }
        return outputs;
    }
}