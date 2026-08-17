package com.example.lambdaclient.dto;

import java.util.List;

/**
 * Response returned by the rag_backend Lambda (parsed from its JSON body).
 */
public class RephraseResponse {

    private String rephrased_statement;
    private List<String> sources;

    public RephraseResponse() {
    }

    public RephraseResponse(String rephrased_statement, List<String> sources) {
        this.rephrased_statement = rephrased_statement;
        this.sources = sources;
    }

    public String getRephrased_statement() {
        return rephrased_statement;
    }

    public void setRephrased_statement(String rephrased_statement) {
        this.rephrased_statement = rephrased_statement;
    }

    public List<String> getSources() {
        return sources;
    }

    public void setSources(List<String> sources) {
        this.sources = sources;
    }
}
