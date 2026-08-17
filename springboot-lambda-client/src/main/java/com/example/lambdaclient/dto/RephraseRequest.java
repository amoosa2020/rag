package com.example.lambdaclient.dto;

import jakarta.validation.constraints.NotBlank;

/**
 * Request body for the /rephrase endpoint.
 */
public class RephraseRequest {

    @NotBlank(message = "statement must not be blank")
    private String statement;

    public RephraseRequest() {
    }

    public RephraseRequest(String statement) {
        this.statement = statement;
    }

    public String getStatement() {
        return statement;
    }

    public void setStatement(String statement) {
        this.statement = statement;
    }
}
