package io.github.hichambenhsine.oopneuralengine.builder;

import io.github.hichambenhsine.oopneuralengine.activation.IdentityActivation;
import io.github.hichambenhsine.oopneuralengine.activation.SigmoidActivation;
import io.github.hichambenhsine.oopneuralengine.activation.SoftMaxActivation;
import io.github.hichambenhsine.oopneuralengine.aspect.SecurityContext;
import io.github.hichambenhsine.oopneuralengine.layer.decorator.NormalizationLayerDecorator;
import io.github.hichambenhsine.oopneuralengine.model.NeuralNetwork;
import io.github.hichambenhsine.oopneuralengine.observer.NetworkDashboard;
import io.github.hichambenhsine.oopneuralengine.state.NetworkStateType;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class NeuralNetworkBuilderTest {

    private static final double DELTA = 1e-6;

    @BeforeEach
    void authenticate() {
        SecurityContext.login("test-admin", "ADMIN");
    }

    @AfterEach
    void cleanup() {
        SecurityContext.clear();
    }

    @Test
    void builderShouldConstructFigure11_1Topology3_5_2AndPredict() {
        NetworkDashboard dashboard = new NetworkDashboard();

        NeuralNetwork network = new NeuralNetworkBuilder("Figure-11.1-Net")
                .inputLayer(3)
                .hiddenLayer(5, new SigmoidActivation())
                .decorateLastLayer(NormalizationLayerDecorator::new)
                .outputLayer(2, new SoftMaxActivation())
                .withUniformWeight(0.5)
                .withDefaultBias(0.1)
                .withObserver(dashboard)
                .withTrainingEpochs(3)
                .build();

        assertEquals(NetworkStateType.CONSTRUCTION, network.getCurrentStateType());
        assertEquals(3, network.getLayers().size());
        assertEquals(10, network.getTotalNeurons());       // 3 + 5 + 2
        assertEquals(25, network.getTotalConnections());   // (3*5) + (5*2)

        network.train();
        assertEquals(NetworkStateType.TRAINED, network.getCurrentStateType());
        assertEquals(2, dashboard.getTransitionCount());

        double[] probabilities = network.predict(new double[]{1.0, 0.5, 0.25});
        assertEquals(2, probabilities.length);
        assertEquals(1.0, probabilities[0] + probabilities[1], DELTA);
    }

    @Test
    void builderShouldSupportDeterministicRandomWeights() {
        NeuralNetwork netA = new NeuralNetworkBuilder()
                .inputLayer(2)
                .outputLayer(1, new IdentityActivation())
                .withRandomWeights(123L)
                .build();

        NeuralNetwork netB = new NeuralNetworkBuilder()
                .inputLayer(2)
                .outputLayer(1, new IdentityActivation())
                .withRandomWeights(123L)
                .build();

        netA.fit();
        netB.fit();

        assertArrayEquals(
                netA.predict(new double[]{2.0, 3.0}),
                netB.predict(new double[]{2.0, 3.0}),
                DELTA
        );
    }

    @Test
    void builderShouldEnforceValidLayerOrderingAndSingleBuild() {
        assertThrows(IllegalStateException.class, () ->
                new NeuralNetworkBuilder().hiddenLayer(4, new IdentityActivation()));

        assertThrows(IllegalStateException.class, () ->
                new NeuralNetworkBuilder().inputLayer(2).build());

        NeuralNetworkBuilder builder = new NeuralNetworkBuilder()
                .inputLayer(2)
                .outputLayer(1, new IdentityActivation());
        builder.build();

        assertThrows(IllegalStateException.class, builder::build);
    }
}