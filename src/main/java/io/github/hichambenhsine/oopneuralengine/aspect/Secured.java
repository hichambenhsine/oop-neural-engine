package io.github.hichambenhsine.oopneuralengine.aspect;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Marks a method as protected by role-based access control enforced by {@link SecurityAspect}.
 */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.METHOD)
public @interface Secured {

    /**
     * Allowed roles. If empty, any authenticated user is permitted.
     */
    String[] roles() default {};
}