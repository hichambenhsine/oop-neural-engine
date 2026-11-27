package io.github.hichambenhsine.oopneuralengine.layer.decorator;

import io.github.hichambenhsine.oopneuralengine.layer.Layer;
import io.github.hichambenhsine.oopneuralengine.layer.LayerType;
import io.github.hichambenhsine.oopneuralengine.model.Neuron;

import java.util.List;
import java.util.Objects;

/**
 * Base Decorator wrapping a {@link Layer} to attach additional dynamic behaviors
 * before or after forward propagation.
 */
public abstract class LayerDecorator implements Layer {

    protected final Layer decoratedLayer;

    protected LayerDecorator(Layer decoratedLayer) {
        this.decoratedLayer = Objects.requireNonNull(decoratedLayer, "Decorated layer must not be null");
    }

    public Layer getDecoratedLayer() {
        return decoratedLayer;
    }

    @Override
    public LayerType getType() {
        return decoratedLayer.getType();
    }

    @Override
    public List<Neuron> getNeurons() {
        return decoratedLayer.getNeurons();
    }

    @Override
    public double[] forward(double[] inputs) {
        return decoratedLayer.forward(inputs);
    }
}