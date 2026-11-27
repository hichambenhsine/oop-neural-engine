package io.github.hichambenhsine.oopneuralengine.aspect;

import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.annotation.Pointcut;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

/**
 * AspectJ logging aspect tracing execution of all domain and builder methods.
 */
@Aspect
public class LoggingAspect {

    private static final Logger LOGGER = LoggerFactory.getLogger(LoggingAspect.class);

    @Pointcut("execution(* io.github.hichambenhsine.oopneuralengine..*(..)) && !within(io.github.hichambenhsine.oopneuralengine.aspect..*)")
    public void applicationMethods() {
    }

    @Around("applicationMethods()")
    public Object logMethodExecution(ProceedingJoinPoint joinPoint) throws Throwable {
        String signature = joinPoint.getSignature().toShortString();
        String formattedArgs = formatArgs(joinPoint.getArgs());
        long startNanos = System.nanoTime();

        LOGGER.info("[AOP-LOG] -> {} args={}", signature, formattedArgs);
        try {
            Object result = joinPoint.proceed();
            double durationMs = (System.nanoTime() - startNanos) / 1_000_000.0;
            LOGGER.info("[AOP-LOG] <- {} ({} ms)", signature, String.format("%.3f", durationMs));
            return result;
        } catch (Throwable ex) {
            double durationMs = (System.nanoTime() - startNanos) / 1_000_000.0;
            LOGGER.warn("[AOP-LOG] !! {} threw {} ({} ms): {}",
                    signature, ex.getClass().getSimpleName(), String.format("%.3f", durationMs), ex.getMessage());
            throw ex;
        }
    }

    private String formatArgs(Object[] args) {
        if (args == null || args.length == 0) {
            return "[]";
        }
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < args.length; i++) {
            if (i > 0) {
                sb.append(", ");
            }
            if (args[i] instanceof double[] doubleArray) {
                sb.append(Arrays.toString(doubleArray));
            } else {
                sb.append(args[i]);
            }
        }
        sb.append("]");
        return sb.toString();
    }
}