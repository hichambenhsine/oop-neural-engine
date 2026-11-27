package io.github.hichambenhsine.oopneuralengine.layer;

import io.github.hichambenhsine.oopneuralengine.activation.IdentityActivation;
import io.github.hichambenhsine.oopneuralengine.activation.SoftMaxActivation;
import io.github.hichambenhsine.oopneuralengine.model.Neuron;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class LayerFactoryTest {

    private static final double DELTA = 1e-6;

    @Test
    void factoryShouldCreateEachConcreteLayerType() {
        LayerFactory factory = new LayerFactory();

        Layer input = factory.createLayer(LayerType.INPUT, 3, new IdentityActivation());
        Layer hidden = factory.createLayer(LayerType.HIDDEN, 5, new IdentityActivation());
        Layer output = factory.createLayer(LayerType.OUTPUT, 2, new SoftMaxActivation());

        assertInstanceOf(InputLayer.class, input);
        assertInstanceOf(HiddenLayer.class, hidden);
        assertInstanceOf(OutputLayer.class, output);
        assertEquals(3, input.size());
        assertEquals(5, hidden.size());
        assertEquals(2, output.size());
    }

    @Test
    void factoryShouldRejectInvalidNeuronCount() {
        LayerFactory factory = new LayerFactory();
        assertThrows(IllegalArgumentException.class,
                () -> factory.createLayer(LayerType.HIDDEN, 0, new IdentityActivation()));
    }

    @Test
    void inputAndOutputLayersShouldPropagateSignalThroughConnections() {
        LayerFactory factory = new LayerFactory();
        Layer input = factory.createLayer(LayerType.INPUT, 2, new IdentityActivation());
        Layer output = factory.createLayer(LayerType.OUTPUT, 2, new SoftMaxActivation());

        for (Neuron src : input.getNeurons()) {
            for (Neuron dst : output.getNeurons()) {
                src.connectTo(dst, 0.5);
            }
        }

        double[] inputSignals = input.forward(new double[]{2.0, 4.0});
        double[] predictions = output.forward(inputSignals);

        assertEquals(2, predictions.length);
        assertEquals(1.0, predictions[0] + predictions[1], DELTA);
        assertEquals(0.5, predictions[0], DELTA);
        assertEquals(0.5, predictions[1], DELTA);
    }
}