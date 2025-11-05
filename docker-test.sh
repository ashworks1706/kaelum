#!/bin/bash

echo "=========================================="
echo "Running Kaelum Tests in Docker"
echo "=========================================="
echo ""

echo "1. Quick test..."
docker-compose exec kaelum python -c "from kaelum import enhance; result = enhance('What is 2+2?'); print(result)"

echo ""
echo "=========================================="
echo ""

echo "2. Running full test suite..."
docker-compose exec kaelum python run.py

echo ""
echo "=========================================="
echo ""

echo "3. Testing metrics and active learning..."
docker-compose exec kaelum python test_features.py

echo ""
echo "=========================================="
echo "✓ All tests completed"
echo "=========================================="
