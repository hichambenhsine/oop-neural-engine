package io.github.hichambenhsine.oopneuralengine.observer;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Concrete observer displaying and recording neural network lifecycle statistics.
 */
public class NetworkDashboard implements NetworkObserver {

    private static final Logger LOGGER = LoggerFactory.getLogger(NetworkDashboard.class);
    private final List<NetworkEvent> eventHistory = new ArrayList<>();

    @Override
    public void onStateChanged(NetworkEvent event) {
        eventHistory.add(event);
        LOGGER.info("[DASHBOARD] Network='{}' | Transition: {} -> {} | Layers: {} | Neurons: {} | Connections: {} | Predictions: {}",
                event.networkName(),
                event.previousState(),
                event.newState(),
                event.layerCount(),
                event.totalNeurons(),
                event.totalConnections(),
                event.predictionCount());
    }

    public List<NetworkEvent> getEventHistory() {
        return Collections.unmodifiableList(eventHistory);
    }

    public NetworkEvent getLatestEvent() {
        if (eventHistory.isEmpty()) {
            return null;
        }
        return eventHistory.get(eventHistory.size() - 1);
    }

    public int getTransitionCount() {
        return eventHistory.size();
    }
}