package com.example.lambdaclient;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * Spring Boot entry point for the REST client that invokes the rag_backend
 * AWS Lambda.
 */
@SpringBootApplication
public class LambdaClientApplication {

    public static void main(String[] args) {
        SpringApplication.run(LambdaClientApplication.class, args);
    }
}
