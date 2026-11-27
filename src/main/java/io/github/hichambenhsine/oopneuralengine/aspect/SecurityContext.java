package io.github.hichambenhsine.oopneuralengine.aspect;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

/**
 * Thread-bound authentication and role context for the neural network engine.
 */
public final class SecurityContext {

    public record UserPrincipal(String username, Set<String> roles) {
    }

    private static final ThreadLocal<UserPrincipal> CURRENT_USER = new ThreadLocal<>();

    private SecurityContext() {
    }

    public static void login(String username, String... roles) {
        Objects.requireNonNull(username, "Username must not be null");
        if (username.isBlank()) {
            throw new IllegalArgumentException("Username must not be blank");
        }
        Set<String> roleSet = roles == null
                ? Collections.emptySet()
                : Collections.unmodifiableSet(new HashSet<>(Arrays.asList(roles)));
        CURRENT_USER.set(new UserPrincipal(username, roleSet));
    }

    public static Optional<UserPrincipal> getCurrentUser() {
        return Optional.ofNullable(CURRENT_USER.get());
    }

    public static void clear() {
        CURRENT_USER.remove();
    }
}