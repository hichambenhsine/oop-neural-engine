package io.github.hichambenhsine.oopneuralengine.aspect;

import org.aspectj.lang.JoinPoint;
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.annotation.Before;

import java.util.Arrays;

/**
 * AspectJ security aspect enforcing authentication and role checks on {@link Secured} methods.
 */
@Aspect
public class SecurityAspect {

    @Before("execution(@io.github.hichambenhsine.oopneuralengine.aspect.Secured * io.github.hichambenhsine.oopneuralengine..*(..)) && @annotation(secured)")
    public void checkAccess(JoinPoint joinPoint, Secured secured) {
        SecurityContext.UserPrincipal principal = SecurityContext.getCurrentUser()
                .orElseThrow(() -> new SecurityException(
                        "Unauthenticated access denied for method: " + joinPoint.getSignature().toShortString()
                ));

        String[] requiredRoles = secured.roles();
        if (requiredRoles.length == 0) {
            return;
        }

        boolean authorized = Arrays.stream(requiredRoles)
                .anyMatch(role -> principal.roles().contains(role));

        if (!authorized) {
            throw new SecurityException(
                    "Access denied for user '" + principal.username() + "' on method "
                            + joinPoint.getSignature().toShortString()
                            + ". Required roles: " + Arrays.toString(requiredRoles)
            );
        }
    }
}