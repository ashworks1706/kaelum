# Deployment Guide for KaelumAI

This guide covers deploying KaelumAI in various environments.

## Quick Start with Docker

### 1. Local Docker Deployment

```bash
# Clone the repository
git clone https://github.com/ashworks1706/KaelumAI.git
cd KaelumAI

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys

# Build and run with Docker
docker build -t kaelum .
docker run -p 8000:8000 --env-file .env kaelum
```

### 2. Docker Compose

```bash
# Start the service
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the service
docker-compose down
```

## Cloud Deployment

### AWS ECS / Fargate

1. **Build and push Docker image**:
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker build -t kaelum .
docker tag kaelum:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/kaelum:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/kaelum:latest
```

2. **Create ECS Task Definition** with:
   - Container port: 8000
   - Environment variables: OPENAI_API_KEY, etc.
   - Health check: `/health` endpoint

3. **Create ECS Service** with:
   - Load balancer (ALB)
   - Auto-scaling policies
   - CloudWatch logging

### Google Cloud Run

```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT-ID/kaelum
gcloud run deploy kaelum \
  --image gcr.io/PROJECT-ID/kaelum \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars OPENAI_API_KEY=your-key
```

### Azure Container Instances

```bash
# Create container instance
az container create \
  --resource-group myResourceGroup \
  --name kaelum \
  --image <registry>/kaelum:latest \
  --dns-name-label kaelum-api \
  --ports 8000 \
  --environment-variables OPENAI_API_KEY=your-key
```

## Kubernetes Deployment

### 1. Create Kubernetes Manifests

**deployment.yaml**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kaelum
spec:
  replicas: 3
  selector:
    matchLabels:
      app: kaelum
  template:
    metadata:
      labels:
        app: kaelum
    spec:
      containers:
      - name: kaelum
        image: kaelum:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: kaelum-secrets
              key: openai-api-key
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
```

**service.yaml**:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: kaelum-service
spec:
  type: LoadBalancer
  selector:
    app: kaelum
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
```

**secret.yaml**:
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: kaelum-secrets
type: Opaque
stringData:
  openai-api-key: your-api-key-here
```

### 2. Deploy to Kubernetes

```bash
# Create secret
kubectl apply -f secret.yaml

# Deploy application
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# Check status
kubectl get pods
kubectl get services

# View logs
kubectl logs -f deployment/kaelum
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for LLM access |
| `ANTHROPIC_API_KEY` | No | Anthropic API key (if using Claude) |
| `LOG_LEVEL` | No | Logging level (default: INFO) |
| `ENVIRONMENT` | No | Environment name (development/production) |

## Scaling Considerations

### Horizontal Scaling
- KaelumAI is stateless and can be scaled horizontally
- Use load balancer to distribute traffic
- Consider using Redis for shared caching in multi-instance deployments

### Vertical Scaling
- Memory: 512MB minimum, 1-2GB recommended
- CPU: 0.5 cores minimum, 1-2 cores recommended
- Adjust based on traffic and LLM response times

### Performance Optimization
1. **API Rate Limits**: Implement rate limiting at load balancer level
2. **Caching**: Add Redis for caching frequent queries
3. **Async Processing**: For high-volume scenarios, use message queues
4. **Model Selection**: Use faster models (gpt-3.5-turbo) for verifier/reflector

## Monitoring

### Health Checks
- Endpoint: `GET /health`
- Should return 200 OK when healthy

### Metrics
- Endpoint: `GET /metrics`
- Track verification rates, confidence scores, latency

### Logging
- Logs are written to stdout/stderr
- Integrate with CloudWatch, Stackdriver, or Azure Monitor

### Recommended Monitoring Tools
- **Prometheus + Grafana**: For metrics visualization
- **Sentry**: For error tracking
- **DataDog / New Relic**: For APM

## Security

1. **API Keys**: Store in secrets management (AWS Secrets Manager, etc.)
2. **HTTPS**: Always use TLS/SSL in production
3. **Authentication**: Add API key/JWT authentication for endpoints
4. **Rate Limiting**: Implement to prevent abuse
5. **Network**: Use VPC/private networks when possible

## Troubleshooting

### Container won't start
- Check API keys are set correctly
- Verify Docker image was built successfully
- Check logs: `docker logs <container-id>`

### High latency
- Check LLM API response times
- Consider using faster models for verification
- Reduce `max_reflection_iterations` in config

### Out of memory
- Increase container memory limits
- Check for memory leaks in logs
- Reduce concurrent requests

## Support

For issues and questions:
- GitHub Issues: https://github.com/ashworks1706/KaelumAI/issues
- Email: ashworks1706@gmail.com
