package io.github.hichambenhsine.oopneuralengine.activation;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ActivationFunctionTest {

    private static final double DELTA = 1e-6;

    @Test
    void sigmoidShouldComputeExpectedValues() {
        ActivationFunction sigmoid = new SigmoidActivation();

        assertEquals(0.5, sigmoid.activate(0.0), DELTA);
        assertTrue(sigmoid.activate(10.0) > 0.9999);
        assertTrue(sigmoid.activate(-10.0) < 0.0001);
    }

    @Test
    void identityShouldReturnInputUnchanged() {
        ActivationFunction identity = new IdentityActivation();

        assertEquals(-3.5, identity.activate(-3.5), DELTA);
        assertEquals(0.0, identity.activate(0.0), DELTA);
        assertEquals(42.25, identity.activate(42.25), DELTA);
    }

    @Test
    void softMaxLayerShouldProduceValidProbabilityDistribution() {
        ActivationFunction softMax = new SoftMaxActivation();
        double[] outputs = softMax.activateLayer(new double[]{1.0, 2.0, 3.0});

        double sum = 0.0;
        for (double output : outputs) {
            assertTrue(output > 0.0 && output < 1.0);
            sum += output;
        }
        assertEquals(1.0, sum, DELTA);
        assertTrue(outputs[2] > outputs[1] && outputs[1] > outputs[0]);
    }

    @Test
    void softMaxLayerShouldHandleLargeInputsWithoutOverflow() {
        ActivationFunction softMax = new SoftMaxActivation();
        double[] outputs = softMax.activateLayer(new double[]{1000.0, 1001.0});

        assertFalse(Double.isNaN(outputs[0]));
        assertFalse(Double.isNaN(outputs[1]));
        assertEquals(1.0, outputs[0] + outputs[1], DELTA);
    }

    @Test
    void softMaxLayerShouldRejectNullAndHandleEmptyArray() {
        ActivationFunction softMax = new SoftMaxActivation();

        assertThrows(IllegalArgumentException.class, () -> softMax.activateLayer(null));
        assertArrayEquals(new double[0], softMax.activateLayer(new double[0]), DELTA);
    }

    @Test
    void legacyAdapterShouldDelegateToLegacyTanhFunction() {
        LegacyActivationFunction legacyTanh = new LegacyTanhActivation();
        ActivationFunction adapted = new LegacyActivationAdapter(legacyTanh);

        assertEquals(0.0, adapted.activate(0.0), DELTA);
        assertEquals(Math.tanh(1.5), adapted.activate(1.5), DELTA);
        assertEquals(Math.tanh(-2.0), adapted.activate(-2.0), DELTA);
    }

    @Test
    void legacyAdapterShouldRejectNullDelegate() {
        assertThrows(NullPointerException.class, () -> new LegacyActivationAdapter(null));
    }
}