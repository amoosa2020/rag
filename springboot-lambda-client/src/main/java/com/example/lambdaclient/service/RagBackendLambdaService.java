package com.example.lambdaclient.service;

import com.example.lambdaclient.dto.RephraseResponse;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import software.amazon.awssdk.core.SdkBytes;
import software.amazon.awssdk.services.lambda.LambdaClient;
import software.amazon.awssdk.services.lambda.model.InvokeRequest;
import software.amazon.awssdk.services.lambda.model.InvokeResponse;
import software.amazon.awssdk.services.lambda.model.LambdaException;

import java.nio.charset.StandardCharsets;

/**
 * Invokes the rag_backend AWS Lambda and parses its response.
 *
 * The Lambda handler ({@code lambda_function.lambda_handler}) accepts either a
 * plain JSON event or an API Gateway proxy event. This client sends a plain
 * JSON event: {"statement": "..."}.
 */
@Service
public class RagBackendLambdaService {

    private static final Logger log = LoggerFactory.getLogger(RagBackendLambdaService.class);

    private final LambdaClient lambdaClient;
    private final ObjectMapper objectMapper;

    @Value("${aws.lambda.function-name}")
    private String functionName;

    public RagBackendLambdaService(LambdaClient lambdaClient, ObjectMapper objectMapper) {
        this.lambdaClient = lambdaClient;
        this.objectMapper = objectMapper;
    }

    /**
     * Invoke the rag_backend Lambda to rephrase the given statement.
     *
     * @param statement the statement to rephrase
     * @return the parsed rephrase response from the Lambda
     * @throws LambdaException if the Lambda invocation fails
     */
    public RephraseResponse rephrase(String statement) {
        // Build the plain JSON event the Lambda handler expects.
        String payload = "{\"statement\": " + toJsonString(statement) + "}";

        InvokeRequest request = InvokeRequest.builder()
                .functionName(functionName)
                .payload(SdkBytes.fromUtf8String(payload))
                .build();

        log.info("Invoking Lambda '{}' with statement: {}", functionName, statement);
        InvokeResponse response = lambdaClient.invoke(request);

        String responseBody = response.payload() != null
                ? response.payload().asUtf8String()
                : "";

        // Check for a Lambda function error (e.g. unhandled exception).
        if (response.functionError() != null) {
            log.error("Lambda '{}' returned function error '{}': {}",
                    functionName, response.functionError(), responseBody);
            throw LambdaException.builder()
                    .message("Lambda function error: " + response.functionError()
                            + " -> " + responseBody)
                    .build();
        }

        return parseLambdaResponse(responseBody);
    }

    /**
     * The Lambda returns an API Gateway-style envelope:
     *   {"statusCode": 200, "headers": {...}, "body": "{\"rephrased_statement\":...}"}
     * Unwrap it and parse the inner body into a RephraseResponse.
     */
    private RephraseResponse parseLambdaResponse(String responseBody) {
        try {
            JsonNode envelope = objectMapper.readTree(responseBody);

            int statusCode = envelope.path("statusCode").asInt(200);
            if (statusCode >= 400) {
                String error = envelope.path("body").asText(responseBody);
                log.error("Lambda returned status {}: {}", statusCode, error);
                throw LambdaException.builder()
                        .message("Lambda returned status " + statusCode + ": " + error)
                        .build();
            }

            String innerBody = envelope.path("body").asText();
            if (innerBody == null || innerBody.isBlank()) {
                // Fall back to treating the whole response as the payload.
                innerBody = responseBody;
            }

            return objectMapper.readValue(innerBody, RephraseResponse.class);
        } catch (LambdaException e) {
            throw e;
        } catch (Exception e) {
            log.error("Failed to parse Lambda response: {}", responseBody, e);
            throw LambdaException.builder()
                    .message("Failed to parse Lambda response: " + e.getMessage())
                    .cause(e)
                    .build();
        }
    }

    private String toJsonString(String value) {
        try {
            return objectMapper.writeValueAsString(value);
        } catch (Exception e) {
            // Fallback: escape manually (should never happen for a String).
            return "\"" + value.replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
        }
    }
}
