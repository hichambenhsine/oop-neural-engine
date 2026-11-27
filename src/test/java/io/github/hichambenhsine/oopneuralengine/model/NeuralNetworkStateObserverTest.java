package io.github.hichambenhsine.oopneuralengine.model;

import io.github.hichambenhsine.oopneuralengine.activation.IdentityActivation;
import io.github.hichambenhsine.oopneuralengine.aspect.SecurityContext;
import io.github.hichambenhsine.oopneuralengine.layer.Layer;
import io.github.hichambenhsine.oopneuralengine.layer.LayerFactory;
import io.github.hichambenhsine.oopneuralengine.layer.LayerType;
import io.github.hichambenhsine.oopneuralengine.observer.NetworkDashboard;
import io.github.hichambenhsine.oopneuralengine.observer.NetworkEvent;
import io.github.hichambenhsine.oopneuralengine.state.NetworkStateType;
import io.github.hichambenhsine.oopneuralengine.state.TrainingState;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class NeuralNetworkStateObserverTest {

    private static final double DELTA = 1e-6;

    @BeforeEach
    void authenticate() {
        SecurityContext.login("test-admin", "ADMIN");
    }

    @AfterEach
    void cleanup() {
        SecurityContext.clear();
    }

    private NeuralNetwork createSimpleNetwork() {
        LayerFactory factory = new LayerFactory();
        Layer input = factory.createLayer(LayerType.INPUT, 2, new IdentityActivation());
        Layer output = factory.createLayer(LayerType.OUTPUT, 1, new IdentityActivation());

        input.getNeurons().get(0).connectTo(output.getNeurons().get(0), 0.5);
        input.getNeurons().get(1).connectTo(output.getNeurons().get(0), 0.5);

        NeuralNetwork network = new NeuralNetwork("TestNet");
        network.addLayer(input);
        network.addLayer(output);
        return network;
    }

    @Test
    void networkShouldStartInConstructionStateAndBlockPredict() {
        NeuralNetwork network = createSimpleNetwork();

        assertEquals(NetworkStateType.CONSTRUCTION, network.getCurrentStateType());
        assertThrows(IllegalStateException.class, () -> network.predict(new double[]{1.0, 2.0}));
    }

    @Test
    void trainShouldTransitionConstructionToTrainingThenTrainedAndNotifyDashboard() {
        NeuralNetwork network = createSimpleNetwork();
        NetworkDashboard dashboard = new NetworkDashboard();
        network.addObserver(dashboard);

        network.train();

        assertEquals(NetworkStateType.TRAINED, network.getCurrentStateType());
        List<NetworkEvent> events = dashboard.getEventHistory();
        assertEquals(2, events.size());

        assertEquals(NetworkStateType.CONSTRUCTION, events.get(0).previousState());
        assertEquals(NetworkStateType.TRAINING, events.get(0).newState());

        assertEquals(NetworkStateType.TRAINING, events.get(1).previousState());
        assertEquals(NetworkStateType.TRAINED, events.get(1).newState());
        assertEquals(2, events.get(1).layerCount());
        assertEquals(3, events.get(1).totalNeurons());
        assertEquals(2, events.get(1).totalConnections());

        double[] prediction = network.predict(new double[]{2.0, 4.0});
        assertArrayEquals(new double[]{3.0}, prediction, DELTA);
        assertEquals(1L, network.getPredictionCount());
    }

    @Test
    void fitShouldTransitionDirectlyFromConstructionToTrained() {
        NeuralNetwork network = createSimpleNetwork();
        NetworkDashboard dashboard = new NetworkDashboard();
        network.addObserver(dashboard);

        network.fit();

        assertEquals(NetworkStateType.TRAINED, network.getCurrentStateType());
        assertEquals(1, dashboard.getTransitionCount());
        assertEquals(NetworkStateType.CONSTRUCTION, dashboard.getLatestEvent().previousState());
        assertEquals(NetworkStateType.TRAINED, dashboard.getLatestEvent().newState());

        double[] prediction = network.predict(new double[]{4.0, 6.0});
        assertArrayEquals(new double[]{5.0}, prediction, DELTA);
    }

    @Test
    void trainedAndTrainingStatesShouldRejectInvalidOperations() {
        NeuralNetwork network = createSimpleNetwork();
        Layer extraLayer = new LayerFactory().createLayer(LayerType.HIDDEN, 2, new IdentityActivation());

        network.fit();
        assertThrows(IllegalStateException.class, () -> network.addLayer(extraLayer));
        assertThrows(IllegalStateException.class, network::train);
        assertThrows(IllegalStateException.class, network::fit);

        network.setState(new TrainingState());
        assertThrows(IllegalStateException.class, () -> network.predict(new double[]{1.0, 1.0}));
        assertThrows(IllegalStateException.class, network::train);
        assertThrows(IllegalStateException.class, network::fit);
    }
}