package com.example.lambdaclient.controller;

import com.example.lambdaclient.dto.ErrorResponse;
import com.example.lambdaclient.dto.RephraseRequest;
import com.example.lambdaclient.dto.RephraseResponse;
import com.example.lambdaclient.service.RagBackendLambdaService;
import jakarta.validation.Valid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import software.amazon.awssdk.services.lambda.model.LambdaException;

/**
 * REST endpoint that invokes the rag_backend AWS Lambda.
 *
 * POST /rephrase
 *   Body: {"statement": "The dealer profile needs updating."}
 */
@RestController
@RequestMapping("/api")
public class RephraseController {

    private static final Logger log = LoggerFactory.getLogger(RephraseController.class);

    private final RagBackendLambdaService lambdaService;

    public RephraseController(RagBackendLambdaService lambdaService) {
        this.lambdaService = lambdaService;
    }

    @PostMapping("/rephrase")
    public ResponseEntity<RephraseResponse> rephrase(@Valid @RequestBody RephraseRequest request) {
        log.info("Received rephrase request for statement: {}", request.getStatement());
        RephraseResponse result = lambdaService.rephrase(request.getStatement());
        return ResponseEntity.ok(result);
    }

    @ExceptionHandler(LambdaException.class)
    public ResponseEntity<ErrorResponse> handleLambdaException(LambdaException e) {
        log.error("Lambda invocation failed", e);
        return ResponseEntity.status(HttpStatus.BAD_GATEWAY)
                .body(new ErrorResponse(e.getMessage()));
    }

    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<ErrorResponse> handleValidation(MethodArgumentNotValidException e) {
        String message = e.getBindingResult().getFieldErrors().stream()
                .findFirst()
                .map(fe -> fe.getField() + ": " + fe.getDefaultMessage())
                .orElse("Invalid request");
        return ResponseEntity.badRequest().body(new ErrorResponse(message));
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity<ErrorResponse> handleGeneric(Exception e) {
        log.error("Unexpected error", e);
        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(new ErrorResponse(e.getMessage()));
    }
}
