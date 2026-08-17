# Spring Boot Lambda Client

A Spring Boot REST service that invokes the `rag_backend` AWS Lambda and returns
the rephrased statement.

## Endpoint

```
POST /api/rephrase
Content-Type: application/json

{
  "statement": "The dealer profile needs updating."
}
```

### Success response (200)

```json
{
  "rephrased_statement": "...",
  "sources": ["..."]
}
```

### Error responses

- `400` — missing/blank `statement`
- `502` — Lambda invocation failed (LambdaException)
- `500` — unexpected error

## Configuration

Edit [`src/main/resources/application.yml`](src/main/resources/application.yml):

```yaml
aws:
  region: ap-south-1          # region where the Lambda is deployed
  lambda:
    function-name: rag_backend # name or ARN of the Lambda function
```

## AWS Credentials

The client uses the AWS SDK **default credential provider chain**:

1. Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`)
2. Java system properties
3. `~/.aws/credentials` profile file
4. ECS/EC2 instance role (when running on AWS)

The IAM user/role invoking the Lambda needs the `lambda:InvokeFunction` permission
on the target function.

## Run

```bash
# from the springboot-lambda-client directory
mvn spring-boot:run
```

The service starts on `http://localhost:8080`.

## Test with curl

```bash
curl -X POST http://localhost:8080/api/rephrase \
  -H "Content-Type: application/json" \
  -d '{"statement": "The dealer profile needs updating."}'
```
