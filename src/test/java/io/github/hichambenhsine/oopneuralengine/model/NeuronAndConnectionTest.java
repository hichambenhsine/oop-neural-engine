package io.github.hichambenhsine.oopneuralengine.model;

import io.github.hichambenhsine.oopneuralengine.activation.IdentityActivation;
import io.github.hichambenhsine.oopneuralengine.activation.SigmoidActivation;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class NeuronAndConnectionTest {

    private static final double DELTA = 1e-6;

    @Test
    void connectionShouldValidateWeightBounds() {
        Neuron src = new Neuron("N1", new IdentityActivation());
        Neuron dst = new Neuron("N2", new IdentityActivation());

        Connection valid = new Connection(src, dst, 0.5);
        assertEquals(0.5, valid.getWeight(), DELTA);

        assertThrows(IllegalArgumentException.class, () -> new Connection(src, dst, -0.1));
        assertThrows(IllegalArgumentException.class, () -> new Connection(src, dst, 1.1));
        assertThrows(IllegalArgumentException.class, () -> new Connection(src, dst, Double.NaN));
    }

    @Test
    void neuronShouldComputeWeightedSumPlusBiasAndActivate() {
        Neuron input1 = new Neuron("X1", new IdentityActivation());
        Neuron input2 = new Neuron("X2", new IdentityActivation());
        Neuron target = new Neuron("H1", 0.5, new IdentityActivation());

        input1.activateExternalInput(2.0);
        input2.activateExternalInput(4.0);

        input1.connectTo(target, 0.5); // 2.0 * 0.5 = 1.0
        input2.connectTo(target, 0.25); // 4.0 * 0.25 = 1.0

        double output = target.forward(); // 1.0 + 1.0 + 0.5 = 2.5
        assertEquals(2.5, output, DELTA);
        assertEquals(2.5, target.getOutput(), DELTA);
    }

    @Test
    void neuronShouldAllowSwappingActivationStrategyAtRuntime() {
        Neuron neuron = new Neuron("N1", 0.0, new IdentityActivation());
        assertEquals(0.0, neuron.activateExternalInput(0.0), DELTA);

        neuron.setActivationFunction(new SigmoidActivation());
        assertEquals(0.5, neuron.activateExternalInput(0.0), DELTA);
    }
}